#!/usr/bin/env python3

"""
How to use:

python3 crop_pace_pt.py \
    --input data/roa_sim/chirp_data.pt \
    --output data/roa_sim/chirp_data_Hip.pt \
    --start-sec 0.0 \
    --end-sec 68.0 
    
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


REQUIRED_KEYS = (
    "time",
    "dof_pos",
    "des_dof_pos",
)


def load_dataset(path: Path) -> dict[str, Any]:
    dataset = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(dataset, dict):
        raise TypeError(
            f"Dataset must be a dictionary, got {type(dataset).__name__}"
        )

    missing_keys = [
        key for key in REQUIRED_KEYS
        if key not in dataset
    ]

    if missing_keys:
        raise KeyError(
            f"Missing required dataset keys: {missing_keys}"
        )

    time = dataset["time"]
    dof_pos = dataset["dof_pos"]
    des_dof_pos = dataset["des_dof_pos"]

    if not isinstance(time, torch.Tensor):
        raise TypeError("'time' must be a torch.Tensor.")

    if not isinstance(dof_pos, torch.Tensor):
        raise TypeError("'dof_pos' must be a torch.Tensor.")

    if not isinstance(des_dof_pos, torch.Tensor):
        raise TypeError("'des_dof_pos' must be a torch.Tensor.")

    if time.ndim != 1:
        raise ValueError(
            f"time must have shape [T], got {tuple(time.shape)}"
        )

    if dof_pos.ndim != 2:
        raise ValueError(
            f"dof_pos must have shape [T, DOF], "
            f"got {tuple(dof_pos.shape)}"
        )

    if des_dof_pos.shape != dof_pos.shape:
        raise ValueError(
            "dof_pos and des_dof_pos shape mismatch: "
            f"{tuple(dof_pos.shape)} vs "
            f"{tuple(des_dof_pos.shape)}"
        )

    if time.shape[0] != dof_pos.shape[0]:
        raise ValueError(
            "Time and position sample count mismatch: "
            f"{time.shape[0]} vs {dof_pos.shape[0]}"
        )

    if time.numel() == 0:
        raise ValueError("Dataset is empty.")

    return dataset


def crop_dataset(
    dataset: dict[str, Any],
    start_sec: float,
    end_sec: float,
    reset_time: bool,
) -> dict[str, Any]:
    time = dataset["time"]

    # start_sec 이상, end_sec 이하인 샘플을 선택한다.
    mask = (
        (time >= start_sec)
        & (time <= end_sec)
    )

    selected_count = int(mask.sum().item())

    if selected_count == 0:
        raise ValueError(
            "No samples exist in the requested interval: "
            f"{start_sec:.6f} ~ {end_sec:.6f} sec"
        )

    cropped_time = time[mask].clone()
    cropped_dof_pos = dataset["dof_pos"][mask].clone()
    cropped_des_dof_pos = dataset["des_dof_pos"][mask].clone()

    if reset_time:
        cropped_time = cropped_time - cropped_time[0]

    cropped_dataset = dict(dataset)

    cropped_dataset["time"] = cropped_time
    cropped_dataset["dof_pos"] = cropped_dof_pos
    cropped_dataset["des_dof_pos"] = cropped_des_dof_pos

    return cropped_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop a PACE .pt dataset using a time interval."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input PACE .pt file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output cropped .pt file.",
    )

    parser.add_argument(
        "--start-sec",
        type=float,
        required=True,
        help="Start time in seconds, inclusive.",
    )

    parser.add_argument(
        "--end-sec",
        type=float,
        required=True,
        help="End time in seconds, inclusive.",
    )

    parser.add_argument(
        "--keep-original-time",
        action="store_true",
        help=(
            "Keep the original time values. "
            "By default, the first cropped sample becomes 0 seconds."
        ),
    )

    args = parser.parse_args()

    if args.start_sec < 0.0:
        parser.error("--start-sec must be >= 0.")

    if args.end_sec <= args.start_sec:
        parser.error(
            "--end-sec must be greater than --start-sec."
        )

    if Path(args.input).suffix != ".pt":
        parser.error("--input must end with .pt.")

    if Path(args.output).suffix != ".pt":
        parser.error("--output must end with .pt.")

    return args


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    dataset = load_dataset(input_path)

    original_time = dataset["time"]
    original_count = int(original_time.shape[0])
    original_duration = float(
        original_time[-1].item()
        - original_time[0].item()
    )

    cropped_dataset = crop_dataset(
        dataset=dataset,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        reset_time=not args.keep_original_time,
    )

    cropped_time = cropped_dataset["time"]
    cropped_count = int(cropped_time.shape[0])
    cropped_duration = float(
        cropped_time[-1].item()
        - cropped_time[0].item()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        cropped_dataset,
        output_path,
    )

    print("========== PACE PT CROP ==========")
    print(f"[INPUT]  {input_path}")
    print(f"[OUTPUT] {output_path}")
    print(
        f"[RANGE]  "
        f"{args.start_sec:.6f} ~ "
        f"{args.end_sec:.6f} sec"
    )
    print(
        f"[TIME]   reset to zero: "
        f"{not args.keep_original_time}"
    )
    print(
        f"[BEFORE] samples={original_count}, "
        f"duration={original_duration:.6f} sec"
    )
    print(
        f"[AFTER]  samples={cropped_count}, "
        f"duration={cropped_duration:.6f} sec"
    )
    print(
        f"[SHAPE]  time={tuple(cropped_dataset['time'].shape)}"
    )
    print(
        f"[SHAPE]  dof_pos="
        f"{tuple(cropped_dataset['dof_pos'].shape)}"
    )
    print(
        f"[SHAPE]  des_dof_pos="
        f"{tuple(cropped_dataset['des_dof_pos'].shape)}"
    )
    print("==================================")


if __name__ == "__main__":
    main()