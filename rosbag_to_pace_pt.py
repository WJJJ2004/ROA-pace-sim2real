#!/usr/bin/env python3
"""
Convert an ROA calibration rosbag directly to a PACE-compatible PyTorch dataset.

PACE output convention
----------------------
The output file contains the same three required keys as the PACE simulator
data-generation script:

    {
        "time":        torch.Tensor [T],
        "dof_pos":     torch.Tensor [T, 12],
        "des_dof_pos": torch.Tensor [T, 12],
    }

The 12-DOF column order MUST match ROAPaceCfg.joint_order:

    0  left_hip_pitch
    1  left_hip_roll
    2  left_hip_yaw
    3  left_knee_pitch
    4  left_ankle_pitch
    5  left_ankle_roll
    6  right_hip_pitch
    7  right_hip_roll
    8  right_hip_yaw
    9  right_knee_pitch
    10 right_ankle_pitch
    11 right_ankle_roll

Coordinate-space rules
----------------------
Non-RSU joints:
    Target: /hardware_interface/command position, motor IDs 10~17
    State:  /hardware_interface/state position, motor IDs 10~17

    IMPORTANT:
    IDs 10~13 already contain the hip mapper output in actuator/motor space.
    Do not apply the hip mapper or an inverse mapper in this converter.

RSU ankle joints:
    Target: /rsu/target virtual roll/pitch
    State:  /rsu/state q virtual roll/pitch

    Do not use actuator-level commands for motor IDs 18~21.
    Do not use /rsu/imp_solution as PACE ankle position data.

Synchronization
---------------
/hardware_interface/command is used as the reference timeline. For each command
message, the latest message at or before the command timestamp is selected from:

    /hardware_interface/state
    /rsu/target
    /rsu/state

A row is kept only when all required values are finite, all required messages
are within --max-dt-sec, and RSU state is feasible unless
--allow-rsu-infeasible is specified.

Example
-------
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/setup.bash

python3 rosbag_to_pace_pt.py \
    --bag ~/pace_chirp_20260727_010203 \
    --out ~/pace-sim2real/pace_sim2real/data/roa_sim/chirp_data.pt \
    --max-dt-sec 0.02
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    import yaml
except Exception:
    yaml = None

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except Exception as exc:
    print("[ERROR] Failed to import ROS 2 Python modules.", file=sys.stderr)
    print("        Source ROS 2 and the workspace before running:", file=sys.stderr)
    print("        source /opt/ros/humble/setup.bash", file=sys.stderr)
    print("        source ~/colcon_ws/install/setup.bash", file=sys.stderr)
    print(f"        Original error: {exc}", file=sys.stderr)
    raise


# -----------------------------------------------------------------------------
# PACE dataset convention
# -----------------------------------------------------------------------------

ROA_JOINT_ORDER: List[str] = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
]

EXPECTED_DOF = 12

# Non-RSU entries are read directly from the hardware command/state topics.
#
# IMPORTANT:
# IDs 10~13 are the already-mapped hip motor angles produced by the sample
# generator. The PACE USD/config joint slots use hip_pitch/hip_roll names, but
# this converter must not run the hip mapper again.
NON_RSU_JOINT_TO_MOTOR_ID: Dict[str, int] = {
    "left_hip_pitch": 10,
    "left_hip_roll": 12,
    "left_hip_yaw": 14,
    "left_knee_pitch": 16,
    "right_hip_pitch": 11,
    "right_hip_roll": 13,
    "right_hip_yaw": 15,
    "right_knee_pitch": 17,
}

RSU_JOINTS = {
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_ankle_pitch",
    "right_ankle_roll",
}


# -----------------------------------------------------------------------------
# Parsed message containers
# -----------------------------------------------------------------------------

@dataclass
class MotorCommandSample:
    t: float
    by_id: Dict[int, float]


@dataclass
class MotorStateSample:
    t: float
    by_id: Dict[int, float]


@dataclass
class RsuTargetSample:
    t: float
    left_roll: float
    left_pitch: float
    right_roll: float
    right_pitch: float


@dataclass
class RsuStateSample:
    t: float
    left_roll: float
    left_pitch: float
    right_roll: float
    right_pitch: float
    feasible: bool


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def stamp_to_sec(stamp: Any) -> float:
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
    except Exception:
        return 0.0


def select_message_time(msg: Any, bag_time_sec: float) -> float:
    """Prefer a valid ROS header timestamp and otherwise use bag record time."""
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        header_time = stamp_to_sec(msg.header.stamp)
        if header_time > 0.0 and math.isfinite(header_time):
            return header_time
    return bag_time_sec


def detect_storage_id(bag_path: Path) -> str:
    metadata_path = bag_path / "metadata.yaml"

    if metadata_path.exists() and yaml is not None:
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = yaml.safe_load(file)
            storage_id = (
                metadata.get("rosbag2_bagfile_information", {})
                .get("storage_identifier")
            )
            if storage_id:
                return str(storage_id)
        except Exception:
            pass

    if any(bag_path.glob("*.mcap")):
        return "mcap"
    return "sqlite3"


def make_reader(
    bag_path: Path,
    storage_id: str,
) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(bag_path),
            storage_id=storage_id,
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    return reader


def latest_before(
    samples: Sequence[Any],
    reference_time: float,
    max_dt: float,
    start_index: int,
) -> Tuple[Optional[Any], int]:
    """Return the latest sample satisfying sample.t <= reference_time."""
    if not samples:
        return None, 0

    index = max(0, min(start_index, len(samples) - 1))

    while index + 1 < len(samples) and samples[index + 1].t <= reference_time:
        index += 1

    sample = samples[index]
    dt = reference_time - sample.t

    if sample.t <= reference_time and 0.0 <= dt <= max_dt:
        return sample, index

    return None, index


def describe_intervals(times: Sequence[float]) -> Dict[str, float]:
    if len(times) < 2:
        return {
            "count": float(len(times)),
            "duration_sec": 0.0,
            "mean_dt_sec": float("nan"),
            "mean_rate_hz": float("nan"),
            "min_dt_sec": float("nan"),
            "max_dt_sec": float("nan"),
        }

    time_tensor = torch.tensor(times, dtype=torch.float64)
    dt = time_tensor[1:] - time_tensor[:-1]
    positive_dt = dt[dt > 0.0]

    if positive_dt.numel() == 0:
        mean_dt = min_dt = max_dt = float("nan")
        mean_rate = float("nan")
    else:
        mean_dt = float(positive_dt.mean())
        min_dt = float(positive_dt.min())
        max_dt = float(positive_dt.max())
        mean_rate = 1.0 / mean_dt if mean_dt > 0.0 else float("nan")

    return {
        "count": float(len(times)),
        "duration_sec": float(times[-1] - times[0]),
        "mean_dt_sec": mean_dt,
        "mean_rate_hz": mean_rate,
        "min_dt_sec": min_dt,
        "max_dt_sec": max_dt,
    }


# -----------------------------------------------------------------------------
# Message parsing
# -----------------------------------------------------------------------------

def parse_motor_command(msg: Any, t: float) -> MotorCommandSample:
    by_id: Dict[int, float] = {}

    for command in getattr(msg, "commands", []):
        motor_id = int(command.motor_id)
        by_id[motor_id] = finite_float(command.position)

    return MotorCommandSample(t=t, by_id=by_id)


def parse_motor_state(msg: Any, t: float) -> MotorStateSample:
    by_id: Dict[int, float] = {}

    for state in getattr(msg, "states", []):
        motor_id = int(state.motor_id)
        by_id[motor_id] = finite_float(state.position)

    return MotorStateSample(t=t, by_id=by_id)


def parse_rsu_target(msg: Any, t: float) -> RsuTargetSample:
    return RsuTargetSample(
        t=t,
        left_roll=finite_float(msg.left_roll),
        left_pitch=finite_float(msg.left_pitch),
        right_roll=finite_float(msg.right_roll),
        right_pitch=finite_float(msg.right_pitch),
    )


def parse_rsu_state(msg: Any, t: float) -> RsuStateSample:
    q = msg.q

    return RsuStateSample(
        t=t,
        left_roll=finite_float(q.left_rsu_roll),
        left_pitch=finite_float(q.left_rsu_pitch),
        right_roll=finite_float(q.right_rsu_roll),
        right_pitch=finite_float(q.right_rsu_pitch),
        feasible=bool(getattr(msg, "feasible", False)),
    )


# -----------------------------------------------------------------------------
# PACE vector construction
# -----------------------------------------------------------------------------

def build_desired_vector(
    motor_command: MotorCommandSample,
    rsu_target: RsuTargetSample,
) -> List[float]:
    rsu_values = {
        "left_ankle_pitch": rsu_target.left_pitch,
        "left_ankle_roll": rsu_target.left_roll,
        "right_ankle_pitch": rsu_target.right_pitch,
        "right_ankle_roll": rsu_target.right_roll,
    }

    vector: List[float] = []

    for joint_name in ROA_JOINT_ORDER:
        if joint_name in NON_RSU_JOINT_TO_MOTOR_ID:
            motor_id = NON_RSU_JOINT_TO_MOTOR_ID[joint_name]
            vector.append(
                motor_command.by_id.get(motor_id, float("nan"))
            )
        elif joint_name in RSU_JOINTS:
            vector.append(rsu_values[joint_name])
        else:
            raise KeyError(f"Unhandled PACE joint: {joint_name}")

    return vector


def build_measured_vector(
    motor_state: MotorStateSample,
    rsu_state: RsuStateSample,
) -> List[float]:
    rsu_values = {
        "left_ankle_pitch": rsu_state.left_pitch,
        "left_ankle_roll": rsu_state.left_roll,
        "right_ankle_pitch": rsu_state.right_pitch,
        "right_ankle_roll": rsu_state.right_roll,
    }

    vector: List[float] = []

    for joint_name in ROA_JOINT_ORDER:
        if joint_name in NON_RSU_JOINT_TO_MOTOR_ID:
            motor_id = NON_RSU_JOINT_TO_MOTOR_ID[joint_name]
            vector.append(
                motor_state.by_id.get(motor_id, float("nan"))
            )
        elif joint_name in RSU_JOINTS:
            vector.append(rsu_values[joint_name])
        else:
            raise KeyError(f"Unhandled PACE joint: {joint_name}")

    return vector


def all_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


# -----------------------------------------------------------------------------
# Conversion
# -----------------------------------------------------------------------------

def convert(args: argparse.Namespace) -> None:
    bag_path = Path(args.bag).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    if not bag_path.is_dir():
        raise NotADirectoryError(
            f"--bag must point to a rosbag directory: {bag_path}"
        )

    storage_id = (
        args.storage_id
        if args.storage_id != "auto"
        else detect_storage_id(bag_path)
    )

    print("========== ROSBAG -> PACE PT ==========")
    print(f"[INFO] bag                : {bag_path}")
    print(f"[INFO] output             : {output_path}")
    print(f"[INFO] storage_id         : {storage_id}")
    print(f"[INFO] max_dt_sec         : {args.max_dt_sec:.6f}")
    print(f"[INFO] require feasible   : {not args.allow_rsu_infeasible}")
    print("[INFO] PACE joint order:")
    for index, joint_name in enumerate(ROA_JOINT_ORDER):
        source = (
            f"motor ID {NON_RSU_JOINT_TO_MOTOR_ID[joint_name]}"
            if joint_name in NON_RSU_JOINT_TO_MOTOR_ID
            else "RSU virtual joint"
        )
        print(f"  [{index:2d}] {joint_name:<24} <- {source}")

    reader = make_reader(bag_path, storage_id)
    topic_types = {
        item.name: item.type
        for item in reader.get_all_topics_and_types()
    }

    requested_topics = {
        args.motor_command_topic,
        args.motor_state_topic,
        args.rsu_target_topic,
        args.rsu_state_topic,
    }

    missing_topics = sorted(requested_topics.difference(topic_types))
    if missing_topics:
        available = "\n".join(f"  - {name}" for name in sorted(topic_types))
        missing = "\n".join(f"  - {name}" for name in missing_topics)
        raise RuntimeError(
            "Required topics are missing from the rosbag.\n"
            f"Missing:\n{missing}\n"
            f"Available:\n{available}"
        )

    message_type_cache: Dict[str, Any] = {}

    motor_commands: List[MotorCommandSample] = []
    motor_states: List[MotorStateSample] = []
    rsu_targets: List[RsuTargetSample] = []
    rsu_states: List[RsuStateSample] = []

    counts = {topic: 0 for topic in requested_topics}

    while reader.has_next():
        topic, serialized_data, timestamp_ns = reader.read_next()

        if topic not in requested_topics:
            continue

        if topic not in message_type_cache:
            message_type_cache[topic] = get_message(topic_types[topic])

        msg = deserialize_message(
            serialized_data,
            message_type_cache[topic],
        )
        bag_time_sec = float(timestamp_ns) * 1.0e-9
        t = select_message_time(msg, bag_time_sec)
        counts[topic] += 1

        if topic == args.motor_command_topic:
            motor_commands.append(parse_motor_command(msg, t))
        elif topic == args.motor_state_topic:
            motor_states.append(parse_motor_state(msg, t))
        elif topic == args.rsu_target_topic:
            rsu_targets.append(parse_rsu_target(msg, t))
        elif topic == args.rsu_state_topic:
            rsu_states.append(parse_rsu_state(msg, t))

    motor_commands.sort(key=lambda sample: sample.t)
    motor_states.sort(key=lambda sample: sample.t)
    rsu_targets.sort(key=lambda sample: sample.t)
    rsu_states.sort(key=lambda sample: sample.t)

    print("[INFO] rosbag message counts:")
    for topic in sorted(requested_topics):
        print(f"  {topic:<32}: {counts[topic]}")

    if not motor_commands:
        raise RuntimeError("No motor command messages were parsed.")

    idx_motor_state = 0
    idx_rsu_target = 0
    idx_rsu_state = 0

    accepted_times: List[float] = []
    measured_rows: List[List[float]] = []
    desired_rows: List[List[float]] = []

    rejected = {
        "missing_motor_state": 0,
        "missing_rsu_target": 0,
        "missing_rsu_state": 0,
        "rsu_infeasible": 0,
        "missing_or_nonfinite_value": 0,
        "non_monotonic_time": 0,
    }

    last_accepted_time: Optional[float] = None

    for motor_command in motor_commands:
        reference_time = motor_command.t

        motor_state, idx_motor_state = latest_before(
            motor_states,
            reference_time,
            args.max_dt_sec,
            idx_motor_state,
        )
        if motor_state is None:
            rejected["missing_motor_state"] += 1
            continue

        rsu_target, idx_rsu_target = latest_before(
            rsu_targets,
            reference_time,
            args.max_dt_sec,
            idx_rsu_target,
        )
        if rsu_target is None:
            rejected["missing_rsu_target"] += 1
            continue

        rsu_state, idx_rsu_state = latest_before(
            rsu_states,
            reference_time,
            args.max_dt_sec,
            idx_rsu_state,
        )
        if rsu_state is None:
            rejected["missing_rsu_state"] += 1
            continue

        if not args.allow_rsu_infeasible and not rsu_state.feasible:
            rejected["rsu_infeasible"] += 1
            continue

        desired = build_desired_vector(motor_command, rsu_target)
        measured = build_measured_vector(motor_state, rsu_state)

        if (
            len(desired) != EXPECTED_DOF
            or len(measured) != EXPECTED_DOF
            or not all_finite(desired)
            or not all_finite(measured)
        ):
            rejected["missing_or_nonfinite_value"] += 1
            continue

        if (
            last_accepted_time is not None
            and reference_time <= last_accepted_time
        ):
            rejected["non_monotonic_time"] += 1
            continue

        accepted_times.append(reference_time)
        desired_rows.append(desired)
        measured_rows.append(measured)
        last_accepted_time = reference_time

    if not accepted_times:
        rejection_text = "\n".join(
            f"  {name}: {count}"
            for name, count in rejected.items()
        )
        raise RuntimeError(
            "No valid synchronized samples were produced.\n"
            f"Rejected rows:\n{rejection_text}"
        )

    # absolute_time = torch.tensor(accepted_times, dtype=torch.float32)
    # time_tensor = absolute_time - absolute_time[0]

    # dof_pos = torch.tensor(measured_rows, dtype=torch.float32)
    # des_dof_pos = torch.tensor(desired_rows, dtype=torch.float32)
    # 큰 ROS epoch 시간을 먼저 Python float 단계에서 제거한다.
    start_time = accepted_times[0]
    relative_times = [
        float(timestamp - start_time)
        for timestamp in accepted_times
    ]

    # 시간은 float64로 유지한다.
    time_tensor = torch.tensor(
        relative_times,
        dtype=torch.float64,
    )

    dof_pos = torch.tensor(
        measured_rows,
        dtype=torch.float32,
    )
    des_dof_pos = torch.tensor(
        desired_rows,
        dtype=torch.float32,
    )

    # 동일하거나 역전된 timestamp가 남아 있으면,
    # 해당 시간과 대응하는 position 행을 함께 제거한다.
    if time_tensor.numel() > 1:
        time_diff = time_tensor[1:] - time_tensor[:-1]

        keep_mask = torch.ones(
            time_tensor.shape[0],
            dtype=torch.bool,
        )
        keep_mask[1:] = time_diff > 0.0

        removed_count = int((~keep_mask).sum().item())

        if removed_count > 0:
            print(
                f"[WARN] Removing {removed_count} samples with "
                "duplicate/non-increasing timestamps."
            )

            time_tensor = time_tensor[keep_mask]
            dof_pos = dof_pos[keep_mask]
            des_dof_pos = des_dof_pos[keep_mask]

    # 필터링 이후 첫 시간을 정확히 0으로 맞춘다.
    time_tensor = time_tensor - time_tensor[0]
    if time_tensor.ndim != 1:
        raise RuntimeError(
            f"time must be 1-D, got shape={tuple(time_tensor.shape)}"
        )
    if dof_pos.shape != des_dof_pos.shape:
        raise RuntimeError(
            "dof_pos/des_dof_pos shape mismatch: "
            f"{tuple(dof_pos.shape)} vs {tuple(des_dof_pos.shape)}"
        )
    if dof_pos.ndim != 2 or dof_pos.shape[1] != EXPECTED_DOF:
        raise RuntimeError(
            "PACE position tensor must have shape [T, 12], got "
            f"{tuple(dof_pos.shape)}"
        )
    if dof_pos.shape[0] != time_tensor.shape[0]:
        raise RuntimeError(
            "Time/position sample count mismatch: "
            f"{time_tensor.shape[0]} vs {dof_pos.shape[0]}"
        )
    if not torch.isfinite(time_tensor).all():
        raise RuntimeError("time contains NaN or Inf.")
    if not torch.isfinite(dof_pos).all():
        raise RuntimeError("dof_pos contains NaN or Inf.")
    if not torch.isfinite(des_dof_pos).all():
        raise RuntimeError("des_dof_pos contains NaN or Inf.")
    if time_tensor.numel() > 1:
        time_diff = time_tensor[1:] - time_tensor[:-1]

        if not torch.all(time_diff > 0.0):
            bad_count = int((time_diff <= 0.0).sum().item())
            min_dt = float(time_diff.min().item())

            raise RuntimeError(
                "time is still not strictly increasing after filtering: "
                f"bad_count={bad_count}, min_dt={min_dt:.12f}"
            )

        print(
            f"[CHECK] time is strictly increasing | "
            f"min_dt={float(time_diff.min().item()):.9f} sec | "
            f"max_dt={float(time_diff.max().item()):.9f} sec"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the saved dictionary exactly compatible with the original
    # PACE simulator data-generation script.
    dataset = {
        "time": time_tensor.cpu(),
        "dof_pos": dof_pos.cpu(),
        "des_dof_pos": des_dof_pos.cpu(),
    }

    torch.save(dataset, output_path)

    timing = describe_intervals(accepted_times)
    total_commands = len(motor_commands)
    accepted = len(accepted_times)

    print("=============== SUMMARY ===============")
    print(f"[WRITE] {output_path}")
    print(f"[SUMMARY] command rows       : {total_commands}")
    print(f"[SUMMARY] accepted rows      : {accepted}")
    print(f"[SUMMARY] accepted ratio     : {accepted / total_commands:.4f}")
    print(f"[SUMMARY] tensor time        : {tuple(time_tensor.shape)}")
    print(f"[SUMMARY] tensor dof_pos     : {tuple(dof_pos.shape)}")
    print(f"[SUMMARY] tensor des_dof_pos : {tuple(des_dof_pos.shape)}")
    print(f"[SUMMARY] duration           : {timing['duration_sec']:.6f} sec")
    print(f"[SUMMARY] mean dt            : {timing['mean_dt_sec']:.6f} sec")
    print(f"[SUMMARY] mean rate          : {timing['mean_rate_hz']:.3f} Hz")
    print("[SUMMARY] rejected rows:")
    for name, count in rejected.items():
        print(f"  {name:<28}: {count}")

    print("[CHECK] First desired sample:")
    for index, joint_name in enumerate(ROA_JOINT_ORDER):
        print(
            f"  [{index:2d}] {joint_name:<24} "
            f"target={des_dof_pos[0, index].item():+.6f} "
            f"state={dof_pos[0, index].item():+.6f}"
        )
    print("=======================================")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an ROA chirp rosbag directly to a PACE-compatible .pt file."
        )
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to the rosbag directory.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output .pt path, for example data/roa_sim/chirp_data.pt.",
    )
    parser.add_argument(
        "--storage-id",
        default="auto",
        choices=["auto", "sqlite3", "mcap"],
        help="Rosbag storage backend. Default: auto.",
    )
    parser.add_argument(
        "--max-dt-sec",
        type=float,
        default=0.02,
        help=(
            "Maximum allowed latest-before synchronization gap. "
            "Default: 0.02 sec."
        ),
    )
    parser.add_argument(
        "--allow-rsu-infeasible",
        action="store_true",
        help="Keep rows even when /rsu/state feasible is false.",
    )
    parser.add_argument(
        "--motor-command-topic",
        default="/hardware_interface/command",
    )
    parser.add_argument(
        "--motor-state-topic",
        default="/hardware_interface/state",
    )
    parser.add_argument(
        "--rsu-target-topic",
        default="/rsu/target",
    )
    parser.add_argument(
        "--rsu-state-topic",
        default="/rsu/state",
    )

    args = parser.parse_args()

    if args.max_dt_sec <= 0.0:
        parser.error("--max-dt-sec must be > 0.")

    if Path(args.out).suffix != ".pt":
        parser.error("--out must end with .pt.")

    return args


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()