from __future__ import annotations

import argparse
from pathlib import Path

from data.photoshape import convert_gaussian_pt_to_inria_ply


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Gaussian .pt/.pth payload to an INRIA-style .ply file "
            "compatible with data.photoshape.load_inria_data."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input Gaussian payload (.pt/.pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .ply path.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.025,
        help=(
            "Voxel size used only for sparse ckpt payloads "
            "(anchor/offset/scale/rotation/f_dc format)."
        ),
    )
    parser.add_argument(
        "--max-height-quantile",
        type=float,
        default=None,
        help=(
            "If set, remove Gaussians with z-height above this quantile (range [0, 1]). "
            "Example: 0.99 removes points above the 99th percentile of z."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _build_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".ply":
        raise ValueError(f"Output path must end with .ply, got {output_path}.")

    n_points = convert_gaussian_pt_to_inria_ply(
        input_path=str(input_path),
        output_path=str(output_path),
        voxel_size=float(args.voxel_size),
        max_height_quantile=args.max_height_quantile,
    )
    print(f"Wrote {n_points} points to {output_path}")


if __name__ == "__main__":
    main()
