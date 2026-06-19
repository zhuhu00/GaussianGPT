"""Scan a Gaussian-splat dataset and write per-scene stats to CSV.

Each scene directory under --data-root is expected to contain a Gaussian
checkpoint: either an anchor/offset payload (.pth/.pt with "anchor"/"offset"
tensors) or an INRIA-style binary .ply. Optionally, per-scene camera transforms
under --transforms-root contribute view counts.

The CSV (one row per scene) is the canonical output; it is consumed by
dataset_split_from_quick_stats.py to build train/val splits. Aggregate stats are
printed to stdout, and an optional histogram PNG is written alongside the CSV.

Example:
    python scripts/dataset_quick_stats.py \
        --data-root /path/to/gaussians \
        --transforms-root /path/to/transforms \
        --output logs/vfront_houses_quick_stats.csv \
        --z-stats
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

DEFAULT_CKPT_GLOBS = ("**/ckpt.pth", "**/ckpt.pt", "**/point_cloud_30000.ply")
DEFAULT_TRANSFORMS_NAMES = (
    "transforms.json",
    "transforms_train.json",
    "transforms_test.json",
)
DETECT_CKPT_MAX_SCENES = 25  # scenes sampled to auto-detect the ckpt rel-path
FRAMES_NUM_SCAN_BYTES = 65536  # transforms.json prefix scanned for "frames_num"
INTERMEDIATE_EVERY = 250
CSV_FIELDS = [
    "scene_id",
    "status",  # ok | missing | corrupt
    "num_views",
    "num_gaussians",
    "extent_x",
    "extent_y",
    "extent_z",
    "max_abs_x",
    "max_abs_y",
    "max_abs_z",
    "floor_z",
    "error",
]


def _compute_xyz_stats(coords: np.ndarray, floor_percentile: Optional[float]) -> dict:
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    coord_min = np.array([x.min(), y.min(), z.min()], dtype=np.float32)
    coord_max = np.array([x.max(), y.max(), z.max()], dtype=np.float32)
    max_abs = np.array(
        [np.abs(x).max(), np.abs(y).max(), np.abs(z).max()], dtype=np.float32
    )
    floor_z = (
        float(np.percentile(z, floor_percentile))
        if floor_percentile is not None
        else None
    )
    return {
        "num_gaussians": int(coords.shape[0]),
        "coord_min": coord_min,
        "coord_max": coord_max,
        "max_abs": max_abs,
        "floor_z": floor_z,
    }


def load_ply_stats(path: Path, floor_percentile: Optional[float]) -> dict:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading header: {path}")
            line_str = line.decode("ascii").strip()
            header_lines.append(line_str)
            if line_str == "end_header":
                break
        data_start = f.tell()

    if header_lines[0] != "ply":
        raise RuntimeError(f"Not a PLY file: {path}")

    format_line = next((ln for ln in header_lines if ln.startswith("format ")), None)
    if format_line != "format binary_little_endian 1.0":
        raise RuntimeError(f"Unsupported PLY format in {path}: {format_line}")

    vertex_line = next(
        (ln for ln in header_lines if ln.startswith("element vertex ")), None
    )
    if vertex_line is None:
        raise RuntimeError(f"No vertex element found in {path}")
    num_vertices = int(vertex_line.split()[-1])

    property_lines = [ln for ln in header_lines if ln.startswith("property ")]
    property_names = [ln.split()[-1] for ln in property_lines]
    if not {"x", "y", "z"}.issubset(property_names):
        raise RuntimeError(f"Missing x/y/z properties in {path}")

    num_properties = len(property_names)
    raw = np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        offset=data_start,
        shape=(num_vertices, num_properties),
    )
    x_idx = property_names.index("x")
    y_idx = property_names.index("y")
    z_idx = property_names.index("z")
    coords = np.stack([raw[:, x_idx], raw[:, y_idx], raw[:, z_idx]], axis=1)
    return _compute_xyz_stats(coords, floor_percentile)


def _to_numpy(data) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def load_pth_stats(
    path: Path, floor_percentile: Optional[float], voxel_size: float
) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    anchor = _to_numpy(data["anchor"]).astype(np.float32, copy=False)
    offset = _to_numpy(data["offset"]).astype(np.float32, copy=False)
    coords = anchor + 1.5 * voxel_size * np.tanh(offset)
    return _compute_xyz_stats(coords, floor_percentile)


def load_ckpt_stats(
    path: Path, floor_percentile: Optional[float], voxel_size: float
) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return load_ply_stats(path, floor_percentile)
    if suffix in {".pth", ".pt"}:
        return load_pth_stats(path, floor_percentile, voxel_size)
    raise RuntimeError(f"Unsupported checkpoint format '{suffix}' for {path}")


def resolve_transforms_path(
    scene_id: str, transforms_root: Optional[Path], transforms_names: tuple[str, ...]
) -> Optional[Path]:
    if transforms_root is None:
        return None
    for name in transforms_names:
        path = transforms_root / scene_id / name
        if path.exists():
            return path
    return None


def detect_ckpt_rel_path(
    scene_dirs: list[Path],
    ckpt_globs: tuple[str, ...],
    preferred_rel_path: Optional[Path],
) -> Optional[Path]:
    """Sample a few scenes to find the relative ckpt path shared across the set.

    A shared rel-path lets resolve_ckpt_path() skip the per-scene glob walk.
    """
    for scene_dir in scene_dirs[:DETECT_CKPT_MAX_SCENES]:
        if preferred_rel_path is not None and (scene_dir / preferred_rel_path).exists():
            return preferred_rel_path
        for pattern in ckpt_globs:
            matches = sorted(scene_dir.glob(pattern))
            if matches:
                return matches[0].relative_to(scene_dir)
    return None


def resolve_ckpt_path(
    scene_dir: Path, preferred_rel_path: Optional[Path], ckpt_globs: tuple[str, ...]
) -> Optional[Path]:
    if preferred_rel_path is not None:
        path = scene_dir / preferred_rel_path
        if path.exists():
            return path
    for pattern in ckpt_globs:
        matches = sorted(scene_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


_FRAMES_NUM_PATTERN = re.compile(r'"frames_num"\s*:\s*(\d+)')
_TEST_FRAMES_NUM_PATTERN = re.compile(r'"test_frames_num"\s*:\s*(\d+)')


def read_num_views(transforms_path: Path, include_test_frames: bool) -> int:
    with transforms_path.open("r", encoding="utf-8") as f:
        head = f.read(FRAMES_NUM_SCAN_BYTES)
    frames_match = _FRAMES_NUM_PATTERN.search(head)
    test_frames_match = _TEST_FRAMES_NUM_PATTERN.search(head)
    if frames_match is not None:
        num_views = int(frames_match.group(1))
        if include_test_frames and test_frames_match is not None:
            num_views += int(test_frames_match.group(1))
        return num_views

    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    num_views = len(transforms.get("frames", []))
    if include_test_frames:
        num_views += len(transforms.get("test_frames", []))
    return num_views


def process_scene(
    scene_dir: Path,
    ckpt_rel_path: Optional[Path],
    ckpt_globs: tuple[str, ...],
    transforms_root: Optional[Path],
    transforms_names: tuple[str, ...],
    include_test_frames: bool,
    floor_percentile: Optional[float],
    voxel_size: float,
) -> dict:
    ckpt_path = resolve_ckpt_path(scene_dir, ckpt_rel_path, ckpt_globs)
    if ckpt_path is None:
        return {"scene_id": scene_dir.name, "status": "missing"}

    try:
        stats = load_ckpt_stats(ckpt_path, floor_percentile, voxel_size)
    except Exception as exc:
        return {
            "scene_id": scene_dir.name,
            "status": "corrupt",
            "error": f"{type(exc).__name__}: {exc}",
        }

    transforms_path = resolve_transforms_path(
        scene_dir.name, transforms_root, transforms_names
    )
    num_views = (
        read_num_views(transforms_path, include_test_frames)
        if transforms_path is not None
        else -1
    )
    return {
        "scene_id": scene_dir.name,
        "status": "ok",
        "num_views": num_views,
        "num_gaussians": stats["num_gaussians"],
        "coord_min": stats["coord_min"],
        "coord_max": stats["coord_max"],
        "max_abs": stats["max_abs"],
        "floor_z": stats["floor_z"],
    }


def _csv_row(result: dict) -> dict:
    status = result["status"]
    if status != "ok":
        return {
            "scene_id": result["scene_id"],
            "status": status,
            "error": result.get("error", ""),
        }
    extent = result["coord_max"] - result["coord_min"]
    max_abs = result["max_abs"]
    num_views = result["num_views"]
    return {
        "scene_id": result["scene_id"],
        "status": "ok",
        "num_views": "" if num_views < 0 else num_views,
        "num_gaussians": result["num_gaussians"],
        "extent_x": f"{extent[0]:.6f}",
        "extent_y": f"{extent[1]:.6f}",
        "extent_z": f"{extent[2]:.6f}",
        "max_abs_x": f"{max_abs[0]:.6f}",
        "max_abs_y": f"{max_abs[1]:.6f}",
        "max_abs_z": f"{max_abs[2]:.6f}",
        "floor_z": "" if result["floor_z"] is None else f"{result['floor_z']:.6f}",
    }


def _print_aggregates(
    ok_results: list[dict], floor_percentile: Optional[float]
) -> None:
    counts = [r["num_gaussians"] for r in ok_results]
    extents = np.stack([r["coord_max"] - r["coord_min"] for r in ok_results], axis=0)
    max_extents = extents.max(axis=1)
    max_abs = np.stack([r["max_abs"] for r in ok_results], axis=0)
    views = [r["num_views"] for r in ok_results if r["num_views"] >= 0]

    print("Aggregate stats (ok scenes):")
    print(
        f"- num_gaussians min/mean/max: "
        f"{min(counts)} / {np.mean(counts):.1f} / {max(counts)}"
    )
    print(
        f"- max_extent   min/mean/max: "
        f"{max_extents.min():.4f} / {max_extents.mean():.4f} / {max_extents.max():.4f}"
    )
    mean_max_abs = max_abs.mean(axis=0)
    global_max_abs = max_abs.max(axis=0)
    print(
        "- max_abs_coord xyz mean: "
        f"[{mean_max_abs[0]:.4f}, {mean_max_abs[1]:.4f}, {mean_max_abs[2]:.4f}]"
    )
    print(
        "- max_abs_coord xyz max:  "
        f"[{global_max_abs[0]:.4f}, {global_max_abs[1]:.4f}, {global_max_abs[2]:.4f}]"
    )
    if floor_percentile is not None:
        floor = np.asarray([r["floor_z"] for r in ok_results], dtype=np.float32)
        print(
            f"- floor_z (p{floor_percentile:g} of z) min/mean/max/std: "
            f"{floor.min():.4f} / {floor.mean():.4f} / {floor.max():.4f} / {floor.std():.4f}"
        )
    if views:
        views_arr = np.asarray(views, dtype=np.float32)
        print(
            f"- num_views min/mean/max/std: "
            f"{int(views_arr.min())} / {views_arr.mean():.1f} / "
            f"{int(views_arr.max())} / {views_arr.std():.1f}"
        )


def write_histogram(ok_results: list[dict], output_path: Path, bins: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Skipping histogram (matplotlib not installed).")
        return

    counts = [r["num_gaussians"] for r in ok_results]
    extents = np.stack([r["coord_max"] - r["coord_min"] for r in ok_results], axis=0)
    views = [r["num_views"] for r in ok_results if r["num_views"] >= 0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    if views:
        axes[0, 0].hist(views, bins=bins)
        axes[0, 0].set_title("Num Views")
        axes[0, 0].set_xlabel("views")
        axes[0, 0].set_ylabel("count")
    else:
        axes[0, 0].text(0.5, 0.5, "No view counts", ha="center", va="center")
        axes[0, 0].set_title("Num Views")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
    axes[0, 1].hist(counts, bins=bins)
    axes[0, 1].set_title("Num Points")
    axes[0, 1].set_xlabel("points")
    axes[0, 1].set_ylabel("count")
    axes[0, 2].axis("off")
    for col, label in enumerate("xyz"):
        axes[1, col].hist(extents[:, col], bins=bins)
        axes[1, col].set_title(f"Extent {label.upper()}")
        axes[1, col].set_xlabel(f"extent_{label}")
        axes[1, col].set_ylabel("count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote histogram to {output_path}")


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root dir with one subdir per scene.",
    )
    parser.add_argument(
        "--transforms-root",
        type=Path,
        default=None,
        help="Optional root dir with <scene_id>/transforms.json for view counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (one row per scene).",
    )
    parser.add_argument(
        "--ckpt-glob",
        action="append",
        default=None,
        dest="ckpt_globs",
        help=(
            "Glob (relative to a scene dir) for the Gaussian ckpt; repeatable. "
            f"Default: {', '.join(DEFAULT_CKPT_GLOBS)}"
        ),
    )
    parser.add_argument(
        "--ckpt-rel-path",
        type=Path,
        default=None,
        help="Exact ckpt rel-path to try before globbing (auto-detected if omitted).",
    )
    parser.add_argument(
        "--transforms-name",
        action="append",
        default=None,
        dest="transforms_names",
        help=(
            "transforms filename to look for under each scene; repeatable. "
            f"Default: {', '.join(DEFAULT_TRANSFORMS_NAMES)}"
        ),
    )
    parser.add_argument("--voxel-size", type=float, default=0.025)
    parser.add_argument(
        "--n-scenes",
        type=int,
        default=None,
        help="Process only the first N scenes (alphabetical).",
    )
    parser.add_argument(
        "--only-numeric",
        action="store_true",
        help="Keep only scene dirs whose name is all digits.",
    )
    parser.add_argument(
        "--include-test-frames",
        action="store_true",
        help="Count test frames toward num_views.",
    )
    parser.add_argument(
        "--z-stats",
        action="store_true",
        help="Compute the floor-z percentile per scene.",
    )
    parser.add_argument(
        "--floor-percentile",
        type=float,
        default=1.0,
        help="Lower z percentile used as floor estimate when --z-stats is set.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--hist-output",
        default=None,
        help="Histogram PNG path. Defaults to <output stem>_hist.png; "
        "pass an empty string to disable.",
    )
    parser.add_argument("--hist-bins", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _build_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"--data-root not found: {args.data_root}")
    if args.transforms_root is not None and not args.transforms_root.exists():
        raise FileNotFoundError(f"--transforms-root not found: {args.transforms_root}")

    ckpt_globs = tuple(args.ckpt_globs) if args.ckpt_globs else DEFAULT_CKPT_GLOBS
    transforms_names = (
        tuple(args.transforms_names)
        if args.transforms_names
        else DEFAULT_TRANSFORMS_NAMES
    )
    floor_percentile = args.floor_percentile if args.z_stats else None

    scene_dirs = [
        d
        for d in sorted(args.data_root.iterdir(), key=lambda p: p.name)
        if d.is_dir() and not d.name.startswith(".")
    ]
    if args.only_numeric:
        scene_dirs = [d for d in scene_dirs if d.name.isdigit()]
    if args.n_scenes is not None:
        scene_dirs = scene_dirs[: args.n_scenes]
    if not scene_dirs:
        print("No scene directories found.")
        return

    ckpt_rel_path = detect_ckpt_rel_path(scene_dirs, ckpt_globs, args.ckpt_rel_path)
    if ckpt_rel_path is None:
        print(
            "WARNING: could not auto-detect a shared ckpt rel-path; globbing per scene."
        )
    else:
        print(f"Resolved ckpt rel-path: {ckpt_rel_path}")

    process_fn = partial(
        process_scene,
        ckpt_rel_path=ckpt_rel_path,
        ckpt_globs=ckpt_globs,
        transforms_root=args.transforms_root,
        transforms_names=transforms_names,
        include_test_frames=args.include_test_frames,
        floor_percentile=floor_percentile,
        voxel_size=args.voxel_size,
    )

    if args.num_workers <= 1:
        results_iter = (process_fn(d) for d in scene_dirs)
        pool = None
    else:
        pool = ThreadPoolExecutor(max_workers=args.num_workers)
        results_iter = pool.map(process_fn, scene_dirs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok_results = []
    total_missing = 0
    total_corrupt = 0
    try:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for idx, result in enumerate(
                tqdm(results_iter, total=len(scene_dirs), desc="Processing scenes"),
                start=1,
            ):
                writer.writerow(_csv_row(result))
                if result["status"] == "ok":
                    ok_results.append(result)
                elif result["status"] == "missing":
                    total_missing += 1
                else:
                    total_corrupt += 1
                if idx % INTERMEDIATE_EVERY == 0:
                    tqdm.write(
                        f"[{idx}/{len(scene_dirs)}] ok={len(ok_results)} "
                        f"missing={total_missing} corrupt={total_corrupt}"
                    )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    print(
        f"Wrote {len(scene_dirs)} rows to {args.output} "
        f"(ok={len(ok_results)} missing={total_missing} corrupt={total_corrupt})"
    )
    if not ok_results:
        return
    _print_aggregates(ok_results, floor_percentile)

    if args.hist_output is None:
        write_histogram(
            ok_results,
            args.output.with_name(f"{args.output.stem}_hist.png"),
            args.hist_bins,
        )
    elif args.hist_output:  # empty string disables
        write_histogram(ok_results, Path(args.hist_output), args.hist_bins)


if __name__ == "__main__":
    main()
