"""Build train/val splits from a dataset_quick_stats.py CSV.

Reads the per-scene CSV produced by dataset_quick_stats.py, keeps the scenes that
pass the (optional) view-count / point-count / extent filters, shuffles them, and
writes newline-separated scene-id lists for training and validation.

All filters default to "off" (no bound), so the tool is dataset-agnostic. The
filters used for the VFront houses split, as a reference, were:
    --min-images 100 --max-points 5000000 \
    --min-extent-x 3.2 --max-extent-x 25 \
    --min-extent-y 3.2 --max-extent-y 25 --max-extent-z 4

Example:
    python scripts/dataset_split_from_quick_stats.py \
        --input logs/vfront_houses_quick_stats.csv \
        --train-split data_splits/vfront_houses_train.txt \
        --val-split data_splits/vfront_houses_val.txt \
        --min-images 100 --max-points 5000000 \
        --min-extent-x 3.2 --max-extent-x 25 \
        --min-extent-y 3.2 --max-extent-y 25 --max-extent-z 4
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Optional


def read_scene_rows(csv_path: Path) -> list[dict]:
    """Read ok-status rows from a quick-stats CSV into typed dicts."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Stats CSV not found: {csv_path}")

    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for record in csv.DictReader(f):
            if record.get("status") != "ok":
                continue
            num_views_raw = record.get("num_views", "")
            extent = [
                float(record["extent_x"]),
                float(record["extent_y"]),
                float(record["extent_z"]),
            ]
            rows.append(
                {
                    "scene_id": record["scene_id"],
                    "num_views": int(num_views_raw) if num_views_raw else -1,
                    "num_gaussians": int(record["num_gaussians"]),
                    "extent_xyz": extent,
                    "max_extent": max(extent),
                }
            )
    return rows


def write_accepted_histogram(rows: list[dict], output_path: Path, bins: int) -> None:
    if not rows:
        print("Skipping accepted-scene histogram (no accepted scenes).")
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Skipping accepted-scene histogram (matplotlib not installed).")
        return

    view_counts = [r["num_views"] for r in rows if r["num_views"] >= 0]
    point_counts = [r["num_gaussians"] for r in rows]
    extent_x = [r["extent_xyz"][0] for r in rows]
    extent_y = [r["extent_xyz"][1] for r in rows]
    extent_z = [r["extent_xyz"][2] for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    if view_counts:
        axes[0, 0].hist(view_counts, bins=bins)
        axes[0, 0].set_title("Num Views (Accepted)")
        axes[0, 0].set_xlabel("views")
        axes[0, 0].set_ylabel("count")
    else:
        axes[0, 0].text(0.5, 0.5, "No view counts", ha="center", va="center")
        axes[0, 0].set_title("Num Views (Accepted)")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
    axes[0, 1].hist(point_counts, bins=bins)
    axes[0, 1].set_title("Num Points (Accepted)")
    axes[0, 1].set_xlabel("points")
    axes[0, 1].set_ylabel("count")
    axes[0, 2].axis("off")
    for col, (label, values) in enumerate(zip("xyz", (extent_x, extent_y, extent_z))):
        axes[1, col].hist(values, bins=bins)
        axes[1, col].set_title(f"Extent {label.upper()} (Accepted)")
        axes[1, col].set_xlabel(f"extent_{label}")
        axes[1, col].set_ylabel("count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote accepted-scene histogram to {output_path}")


def _within(value: float, lo: Optional[float], hi: Optional[float]) -> bool:
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", type=Path, required=True, help="quick-stats CSV path."
    )
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--min-images", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-points", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument(
        "--min-extent", type=float, default=None, help="Min over max(x,y,z) extent."
    )
    parser.add_argument(
        "--max-extent", type=float, default=None, help="Max over max(x,y,z) extent."
    )
    parser.add_argument("--min-extent-x", type=float, default=None)
    parser.add_argument("--max-extent-x", type=float, default=None)
    parser.add_argument("--min-extent-y", type=float, default=None)
    parser.add_argument("--max-extent-y", type=float, default=None)
    parser.add_argument("--min-extent-z", type=float, default=None)
    parser.add_argument("--max-extent-z", type=float, default=None)

    parser.add_argument(
        "--hist-output",
        default=None,
        help="Accepted-scene histogram PNG. Defaults to <train-split stem>_hist.png; "
        "pass an empty string to disable.",
    )
    parser.add_argument("--hist-bins", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _build_args()
    rows = read_scene_rows(args.input)
    if not rows:
        raise RuntimeError(f"No ok-status scene rows found in {args.input}.")

    image_filter = args.min_images is not None or args.max_images is not None
    point_filter = args.min_points is not None or args.max_points is not None
    extent_filter = any(
        v is not None
        for v in [
            args.min_extent,
            args.max_extent,
            args.min_extent_x,
            args.max_extent_x,
            args.min_extent_y,
            args.max_extent_y,
            args.min_extent_z,
            args.max_extent_z,
        ]
    )

    image_pass = image_known = image_unknown = 0
    point_pass = extent_pass = 0
    accepted_rows = []
    for row in rows:
        num_views = row["num_views"]
        extent_x, extent_y, extent_z = row["extent_xyz"]

        meets_images = True
        if image_filter:
            if num_views < 0:
                meets_images = False
                image_unknown += 1
            else:
                image_known += 1
                meets_images = _within(num_views, args.min_images, args.max_images)
            if meets_images:
                image_pass += 1

        meets_points = True
        if point_filter:
            meets_points = _within(
                row["num_gaussians"], args.min_points, args.max_points
            )
            if meets_points:
                point_pass += 1

        meets_extent = True
        if extent_filter:
            meets_extent = (
                _within(row["max_extent"], args.min_extent, args.max_extent)
                and _within(extent_x, args.min_extent_x, args.max_extent_x)
                and _within(extent_y, args.min_extent_y, args.max_extent_y)
                and _within(extent_z, args.min_extent_z, args.max_extent_z)
            )
            if meets_extent:
                extent_pass += 1

        if meets_images and meets_points and meets_extent:
            accepted_rows.append(row)

    accepted_ids = [r["scene_id"] for r in accepted_rows]
    random.Random(args.seed).shuffle(accepted_ids)
    if len(accepted_ids) <= 1:
        train_ids, val_ids = accepted_ids, []
    else:
        split_idx = int(round(len(accepted_ids) * args.train_ratio))
        split_idx = max(1, min(len(accepted_ids) - 1, split_idx))
        train_ids, val_ids = accepted_ids[:split_idx], accepted_ids[split_idx:]

    args.train_split.parent.mkdir(parents=True, exist_ok=True)
    args.val_split.parent.mkdir(parents=True, exist_ok=True)
    args.train_split.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    args.val_split.write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    total = len(rows)
    print(f"Loaded {total} ok scene rows from {args.input}")
    if image_filter:
        pct = 100.0 * image_pass / image_known if image_known else 0.0
        print(f"images: {image_pass}/{image_known} = {pct:.2f}%")
        if image_unknown:
            print(f"unknown num_views (excluded): {image_unknown}")
    if point_filter:
        print(f"points: {point_pass}/{total} = {100.0 * point_pass / total:.2f}%")
    if extent_filter:
        print(f"extent: {extent_pass}/{total} = {100.0 * extent_pass / total:.2f}%")
    print(f"accepted scenes: {len(accepted_ids)}")
    print(
        f"Wrote splits to {args.train_split} ({len(train_ids)}) and "
        f"{args.val_split} ({len(val_ids)})"
    )

    if args.hist_output is None:
        write_accepted_histogram(
            accepted_rows,
            args.train_split.with_name(f"{args.train_split.stem}_hist.png"),
            args.hist_bins,
        )
    elif args.hist_output:  # empty string disables
        write_accepted_histogram(accepted_rows, Path(args.hist_output), args.hist_bins)


if __name__ == "__main__":
    main()
