# © 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

"""Replay a real-robot target trajectory in Isaac Lab.

The same target trajectory is compared against:
1. Real robot joint state
2. Isaac Sim joint state

python scripts/pace/replay_traj.py \
  --task Isaac-Pace-ROA-v0 \
  --config data/roa_sim/chirp_data.pt \
  --communication_delay_ms 14 \
  --output outputs/real_vs_sim.pt \
  --plot_dir outputs/final \
  --headless


"""
import argparse
from pathlib import Path
# from source.pace_sim2real.pace_sim2real.tasks.manager_based.pace.assets.roa import ROA_JOINT_ORDER

from isaaclab.app import AppLauncher
from typing import Optional
# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

ROA_JOINT_ORDER = [
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


parser = argparse.ArgumentParser(
    description="Replay recorded targets in Isaac Lab and compare real/sim states."
)

parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Pace-ROA-v0",
    help="Isaac Lab task name.",
)

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to PACE config.pt containing time, target and real state.",
)

parser.add_argument(
    "--output",
    type=str,
    default="replay_result.pt",
    help="Output file for replay results.",
)

parser.add_argument(
    "--plot_dir",
    type=str,
    default="replay_plots",
    help="Directory for plots.",
)

parser.add_argument(
    "--initial_hold_sec",
    type=float,
    default=0.0,
    help="Hold the first target before replay.",
)

parser.add_argument(
    "--active_threshold",
    type=float,
    default=1.0e-4,
    help=(
        "Target change threshold [rad/sample] used to detect "
        "the active excitation interval."
    ),
)

parser.add_argument(
    "--active_padding_before_sec",
    type=float,
    default=1.0,
    help="Seconds included before the detected target activity.",
)

parser.add_argument(
    "--active_padding_after_sec",
    type=float,
    default=2.0,
    help="Seconds included after the detected target activity.",
)

parser.add_argument(
    "--active_merge_gap_sec",
    type=float,
    default=0.2,
    help=(
        "Inactive gaps shorter than this value are filled when "
        "detecting the active interval."
    ),
)

parser.add_argument(
    "--active_min_duration_sec",
    type=float,
    default=0.5,
    help="Minimum accepted active interval duration.",
)

parser.add_argument(
    "--communication_delay_ms",
    type=int,
    default=0,
    help=(
        "Communication delay from joint target "
        "to the PD controller [ms]."
    ),
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Isaac Lab imports
# -----------------------------------------------------------------------------

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import pace_sim2real.tasks  # noqa: F401
def load_replay_data(dataset_path: str) -> dict:
    """Load target and real trajectories from the rosbag-converted dataset."""

    path = Path(dataset_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file does not exist: {path}"
        )

    dataset = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = [
        "time",
        "des_dof_pos",
        "dof_pos",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in dataset
    ]

    if missing_keys:
        raise KeyError(
            f"Missing keys in dataset: {missing_keys}\n"
            f"Available keys: {list(dataset.keys())}"
        )

    time = torch.as_tensor(
        dataset["time"],
        dtype=torch.float64,
    )

    target = torch.as_tensor(
        dataset["des_dof_pos"],
        dtype=torch.float32,
    )

    real = torch.as_tensor(
        dataset["dof_pos"],
        dtype=torch.float32,
    )

    joint_order = list(
        dataset.get(
            "joint_order",
            ROA_JOINT_ORDER,
        )
    )

    if time.ndim != 1:
        raise ValueError(
            f"time must have shape [T], got {tuple(time.shape)}"
        )

    if target.ndim != 2:
        raise ValueError(
            "des_dof_pos must have shape [T, J], "
            f"got {tuple(target.shape)}"
        )

    if real.ndim != 2:
        raise ValueError(
            "dof_pos must have shape [T, J], "
            f"got {tuple(real.shape)}"
        )

    if target.shape != real.shape:
        raise ValueError(
            "Target/real shape mismatch: "
            f"{tuple(target.shape)} vs {tuple(real.shape)}"
        )

    if target.shape[0] != time.shape[0]:
        raise ValueError(
            "Time/sample count mismatch: "
            f"time={time.shape[0]}, target={target.shape[0]}"
        )

    if target.shape[1] != len(joint_order):
        raise ValueError(
            "Joint count mismatch: "
            f"data={target.shape[1]}, "
            f"joint_order={len(joint_order)}"
        )

    if not torch.isfinite(time).all():
        raise ValueError("time contains NaN or Inf.")

    if not torch.isfinite(target).all():
        raise ValueError("des_dof_pos contains NaN or Inf.")

    if not torch.isfinite(real).all():
        raise ValueError("dof_pos contains NaN or Inf.")

    if time.numel() > 1:
        time_diff = time[1:] - time[:-1]

        if not torch.all(time_diff > 0.0):
            raise ValueError(
                "Dataset time must be strictly increasing."
            )

    # 상대 시간 0초부터 시작
    time = time - time[0]

    print("========== REPLAY DATASET ==========")
    print(f"Path        : {path}")
    print(f"Time shape  : {tuple(time.shape)}")
    print(f"Target shape: {tuple(target.shape)}")
    print(f"Real shape  : {tuple(real.shape)}")
    print(f"Duration    : {time[-1].item():.6f} sec")

    if time.numel() > 1:
        dt = time[1:] - time[:-1]

        print(f"Mean dt     : {dt.mean().item():.9f} sec")
        print(f"Mean rate   : {1.0 / dt.mean().item():.3f} Hz")
        print(f"Min dt      : {dt.min().item():.9f} sec")
        print(f"Max dt      : {dt.max().item():.9f} sec")

    print("Joint order:")

    for index, joint_name in enumerate(joint_order):
        print(f"  [{index:2d}] {joint_name}")

    print("====================================")

    return {
        "time": time,
        "target": target,
        "real": real,
        "joint_order": joint_order,
    }

def zero_order_hold_resample(
    source_time: torch.Tensor,
    source_value: torch.Tensor,
    destination_time: torch.Tensor,
) -> torch.Tensor:
    """Resample target command using zero-order hold."""

    indices = torch.searchsorted(
        source_time,
        destination_time,
        right=True,
    ) - 1

    indices = torch.clamp(
        indices,
        min=0,
        max=len(source_time) - 1,
    )

    return source_value[indices]


def linear_resample(
    source_time: torch.Tensor,
    source_value: torch.Tensor,
    destination_time: torch.Tensor,
) -> torch.Tensor:
    """Linearly resample measured states."""

    source_time_np = source_time.numpy()
    destination_time_np = destination_time.numpy()
    source_value_np = source_value.numpy()

    output = np.empty(
        (
            len(destination_time_np),
            source_value_np.shape[1],
        ),
        dtype=np.float32,
    )

    for joint_index in range(source_value_np.shape[1]):
        output[:, joint_index] = np.interp(
            destination_time_np,
            source_time_np,
            source_value_np[:, joint_index],
        )

    return torch.from_numpy(output)

def calculate_active_metrics(
    sim: torch.Tensor,
    real: torch.Tensor,
    joint_order: list[str],
    active_intervals: dict[str, Optional[tuple[int, int]]],
) -> dict:
    """Calculate metrics independently over each joint's active interval."""

    joint_count = len(joint_order)

    rmse = torch.full(
        (joint_count,),
        float("nan"),
        dtype=torch.float32,
    )

    mae = torch.full_like(rmse, float("nan"))
    max_error = torch.full_like(rmse, float("nan"))

    for joint_index, joint_name in enumerate(joint_order):
        interval = active_intervals.get(joint_name)

        if interval is None:
            continue

        start_index, end_index = interval

        error = (
            sim[start_index : end_index + 1, joint_index]
            - real[start_index : end_index + 1, joint_index]
        )

        rmse[joint_index] = torch.sqrt(
            torch.mean(error.square())
        )

        mae[joint_index] = torch.mean(
            torch.abs(error)
        )

        max_error[joint_index] = torch.max(
            torch.abs(error)
        )

    return {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
    }




def fill_short_false_gaps(
    mask: torch.Tensor,
    max_gap_samples: int,
) -> torch.Tensor:
    """Fill short inactive gaps between active regions."""

    if mask.ndim != 1:
        raise ValueError(
            f"mask must be 1-D, got shape={tuple(mask.shape)}"
        )

    output = mask.clone()

    if max_gap_samples <= 0 or output.numel() == 0:
        return output

    active_indices = torch.nonzero(
        output,
        as_tuple=False,
    ).squeeze(-1)

    if active_indices.numel() < 2:
        return output

    for index in range(active_indices.numel() - 1):
        current_index = int(active_indices[index].item())
        next_index = int(active_indices[index + 1].item())

        gap_length = next_index - current_index - 1

        if 0 < gap_length <= max_gap_samples:
            output[current_index : next_index + 1] = True

    return output


def find_active_interval(
    time: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    padding_before_sec: float,
    padding_after_sec: float,
    merge_gap_sec: float,
    min_duration_sec: float,
) -> Optional[tuple[int, int]]:
    """Detect the active interval from target sample-to-sample changes.

    Returns:
        Tuple ``(start_index, end_index)``. The end index is inclusive.
        Returns None when no meaningful active interval is detected.
    """

    if time.ndim != 1:
        raise ValueError(
            f"time must be 1-D, got shape={tuple(time.shape)}"
        )

    if target.ndim != 1:
        raise ValueError(
            f"target must be 1-D, got shape={tuple(target.shape)}"
        )

    if len(time) != len(target):
        raise ValueError(
            f"time/target length mismatch: {len(time)} vs {len(target)}"
        )

    if len(time) < 2:
        return None

    if threshold <= 0.0:
        raise ValueError("threshold must be greater than zero.")

    dt = time[1:] - time[:-1]
    median_dt = float(torch.median(dt).item())

    if median_dt <= 0.0:
        raise ValueError("Invalid time interval.")

    # 변화량의 길이를 target/time과 맞추기 위해 첫 원소에 0을 추가한다.
    target_delta = torch.zeros_like(target)
    target_delta[1:] = torch.abs(target[1:] - target[:-1])

    active_mask = target_delta > threshold

    # 높은 주파수 chirp에서도 순간적으로 변화량이 threshold 아래로
    # 내려갈 수 있으므로 짧은 inactive gap은 채운다.
    merge_gap_samples = max(
        0,
        int(round(merge_gap_sec / median_dt)),
    )

    active_mask = fill_short_false_gaps(
        active_mask,
        merge_gap_samples,
    )

    active_indices = torch.nonzero(
        active_mask,
        as_tuple=False,
    ).squeeze(-1)

    if active_indices.numel() == 0:
        return None

    raw_start_index = int(active_indices[0].item())
    raw_end_index = int(active_indices[-1].item())

    raw_duration = float(
        time[raw_end_index] - time[raw_start_index]
    )

    if raw_duration < min_duration_sec:
        return None

    start_time = max(
        float(time[0].item()),
        float(time[raw_start_index].item()) - padding_before_sec,
    )

    end_time = min(
        float(time[-1].item()),
        float(time[raw_end_index].item()) + padding_after_sec,
    )

    start_index = int(
        torch.searchsorted(
            time,
            torch.tensor(start_time, dtype=time.dtype),
            right=False,
        ).item()
    )

    end_index = int(
        torch.searchsorted(
            time,
            torch.tensor(end_time, dtype=time.dtype),
            right=True,
        ).item()
    ) - 1

    start_index = max(0, start_index)
    end_index = min(len(time) - 1, end_index)

    if end_index <= start_index:
        return None

    return start_index, end_index

def save_plots(
    time: torch.Tensor,
    target: torch.Tensor,
    real: torch.Tensor,
    sim: torch.Tensor,
    joint_order: list[str],
    active_intervals: dict[str, Optional[tuple[int, int]]],
    plot_dir: str,
    plot_rate_hz: float
) -> None:
    """Save target, real and sim trajectories over each active interval."""

    output_dir = Path(plot_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    time_np = time.cpu().numpy()
    target_np = target.cpu().numpy()
    real_np = real.cpu().numpy()
    sim_np = sim.cpu().numpy()

    for joint_index, joint_name in enumerate(joint_order):
        interval = active_intervals.get(joint_name)

        if interval is None:
            print(
                f"[WARN] Plot skipped because no active interval was "
                f"detected: {joint_name}"
            )
            continue

        start_index, end_index = interval

        sim_dt = float(
            torch.median(time[1:] - time[:-1]).item()
        )
        sim_rate_hz = 1.0 / sim_dt

        plot_stride = max(
            1,
            int(round(sim_rate_hz / plot_rate_hz)),
        )

        plot_indices = torch.arange(
            start_index,
            end_index + 1,
            plot_stride,
            dtype=torch.long,
        )

        plot_time = time[plot_indices].cpu().numpy()
        relative_time = plot_time - plot_time[0]

        figure = plt.figure(figsize=(11, 5))

        plot_indices_np = plot_indices.cpu().numpy()



        plt.plot(
            relative_time,
            sim_np[plot_indices_np, joint_index],
            label="Sim state",
            linewidth=1.1,
            alpha=1.0,
        )

        # plt.plot(
        #     relative_time,
        #     target_np[plot_indices_np, joint_index],
        #     label="Target",
        #     linestyle="--",
        #     linewidth=1.4,
        # )
        
        plt.plot(
            relative_time,
            real_np[plot_indices_np, joint_index],
            label="Real state",
            linewidth=1.3,
        )
        original_start_time = time_np[start_index]
        original_end_time = time_np[end_index]

        plt.title(
            f"Target-to-state response: {joint_name}\n"
            f"Original interval: "
            f"{original_start_time:.3f}–{original_end_time:.3f} s"
        )

        plt.xlabel("Time from detected interval start [s]")
        plt.ylabel("Joint position [rad]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        safe_name = (
            joint_name
            .replace("/", "_")
            .replace(" ", "_")
        )

        figure.savefig(
            output_dir / f"{joint_index:02d}_{safe_name}.png",
            dpi=160,
        )

        plt.close(figure)

def main() -> None:
    data = load_replay_data(args_cli.config)

    source_time = data["time"]
    source_target = data["target"]
    source_real = data["real"]
    joint_order = data["joint_order"]

    action_target = source_target.clone()

    # -------------------------------------------------------------------------
    # Create Isaac Lab environment
    # -------------------------------------------------------------------------

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode=None,
    )

    try:
        env.reset()

        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        device = base_env.device

        control_dt = float(base_env.step_dt)
        control_rate = 1.0 / control_dt

        if args_cli.communication_delay_ms < 0:
            raise ValueError(
                "--communication_delay_ms must be greater than or equal to zero."
            )

        requested_delay_sec = (
            args_cli.communication_delay_ms / 1000.0
        )

        communication_delay_steps = int(
            round(requested_delay_sec / control_dt)
        )

        applied_delay_sec = (
            communication_delay_steps * control_dt
        )

        applied_delay_ms = (
            applied_delay_sec * 1000.0
        )

        print("=" * 80)
        print(f"Task             : {args_cli.task}")
        print(f"Device           : {device}")
        print(f"Physics dt       : {base_env.physics_dt:.9f} s")
        print(f"Control dt       : {control_dt:.9f} s")
        print(f"Control rate     : {control_rate:.3f} Hz")
        print(f"Recorded samples : {len(source_time)}")
        print(f"Recorded duration: {source_time[-1].item():.3f} s")
        print("=" * 80)
        print(
            f"Requested delay  : "
            f"{args_cli.communication_delay_ms} ms"
        )
        print(
            f"Delay steps      : "
            f"{communication_delay_steps} steps"
        )
        print(
            f"Applied delay    : "
            f"{applied_delay_ms:.3f} ms"
        )

        # ---------------------------------------------------------------------
        # Generate exact simulation control time
        # ---------------------------------------------------------------------

        duration = float(source_time[-1].item())

        number_of_steps = (
            int(np.floor(duration / control_dt)) + 1
        )

        replay_time = (
            torch.arange(
                number_of_steps,
                dtype=torch.float64,
            )
            * control_dt
        )

        # Target is a command, so use zero-order hold.
        replay_target = zero_order_hold_resample(
            source_time,
            source_target,
            replay_time,
        )
        
        delayed_replay_target = torch.empty_like(
            replay_target
        )

        for step_index in range(number_of_steps):
            delayed_step_index = max(
                0,
                step_index - communication_delay_steps,
            )

            delayed_replay_target[step_index] = (
                replay_target[delayed_step_index]
            )        

        # Real state is only used for comparison.
        replay_real = linear_resample(
            source_time,
            source_real,
            replay_time,
        )

        # ---------------------------------------------------------------------
        # Joint mapping
        # ---------------------------------------------------------------------

        sim_joint_names = list(robot.data.joint_names)

        missing_joints = [
            joint_name
            for joint_name in joint_order
            if joint_name not in sim_joint_names
        ]

        if missing_joints:
            raise ValueError(
                f"These dataset joints do not exist in simulation: "
                f"{missing_joints}"
            )

        sim_joint_ids = torch.tensor(
            [
                sim_joint_names.index(joint_name)
                for joint_name in joint_order
            ],
            dtype=torch.long,
            device=device,
        )

        print("\nJoint mapping")

        for data_index, joint_name in enumerate(joint_order):
            print(
                f"data[{data_index:2d}] "
                f"-> sim[{sim_joint_ids[data_index].item():2d}] "
                f"{joint_name}"
            )

        action_from_dataset_ids = torch.empty(
            len(joint_order),
            dtype=torch.long,
        )                
        for dataset_index, simulation_index in enumerate(
            sim_joint_ids.cpu().tolist()
        ):
            action_from_dataset_ids[simulation_index] = dataset_index

        print("\nAction reorder mapping")

        for action_index, dataset_index in enumerate(
            action_from_dataset_ids.tolist()
        ):
            print(
                f"action[{action_index:2d}] "
                f"<- data[{dataset_index:2d}] "
                f"{joint_order[dataset_index]}"
            )        
        
        # ---------------------------------------------------------------------
        # Initialize simulation from the first real state
        # ---------------------------------------------------------------------

        initial_position = robot.data.default_joint_pos.clone()
        initial_velocity = torch.zeros_like(
            robot.data.default_joint_vel
        )

        initial_position[:, sim_joint_ids] = (
            replay_real[0]
            .to(device)
            .unsqueeze(0)
        )

        robot.write_joint_state_to_sim(
            initial_position,
            initial_velocity,
        )

        robot.reset()

        # ---------------------------------------------------------------------
        # Optional initial hold
        # ---------------------------------------------------------------------

        hold_steps = int(
            round(args_cli.initial_hold_sec / control_dt)
        )

        first_target = (
            action_target[0]
            .to(device)
            .unsqueeze(0)
        )

        for _ in range(hold_steps):
            env.step(first_target)

        # ---------------------------------------------------------------------
        # Replay
        # ---------------------------------------------------------------------

        sim_state = torch.empty(
            (
                number_of_steps,
                len(joint_order),
            ),
            dtype=torch.float32,
        )

        sim_velocity = torch.empty_like(sim_state)
        # action_target = replay_target[:, action_from_dataset_ids]
        
        action_target = delayed_replay_target[
            :,
            action_from_dataset_ids,
        ]
        
        print("\nStarting target replay")

        for step_index in range(number_of_steps):
            action = (
                action_target[step_index]
                .to(device)
                .unsqueeze(0)
            )

            env.step(action)

            sim_state[step_index] = (
                robot.data.joint_pos[
                    0,
                    sim_joint_ids,
                ]
                .detach()
                .cpu()
            )

            sim_velocity[step_index] = (
                robot.data.joint_vel[
                    0,
                    sim_joint_ids,
                ]
                .detach()
                .cpu()
            )

            if step_index % max(1, int(control_rate)) == 0:
                print(
                    f"{replay_time[step_index].item():8.3f} / "
                    f"{duration:8.3f} s"
                )

        active_intervals = {}

        print("\nDetected active intervals")
        print("-" * 90)

        for joint_index, joint_name in enumerate(joint_order):
            interval = find_active_interval(
                time=replay_time,
                target=replay_target[:, joint_index],
                threshold=args_cli.active_threshold,
                padding_before_sec=args_cli.active_padding_before_sec,
                padding_after_sec=args_cli.active_padding_after_sec,
                merge_gap_sec=args_cli.active_merge_gap_sec,
                min_duration_sec=args_cli.active_min_duration_sec,
            )

            active_intervals[joint_name] = interval

            if interval is None:
                print(
                    f"{joint_name:25s}: "
                    "no active interval detected"
                )
                continue

            start_index, end_index = interval

            print(
                f"{joint_name:25s}: "
                f"{replay_time[start_index].item():8.3f} ~ "
                f"{replay_time[end_index].item():8.3f} sec "
                f"({end_index - start_index + 1} samples)"
            )

        print("-" * 90)

        # ---------------------------------------------------------------------
        # Compare results
        # ---------------------------------------------------------------------

        metrics = calculate_active_metrics(
            sim=sim_state,
            real=replay_real,
            joint_order=joint_order,
            active_intervals=active_intervals,
        )

        print("\nSimulation-to-real response error")
        print("-" * 90)

        for joint_index, joint_name in enumerate(joint_order):
            print(
                f"{joint_name:30s} "
                f"RMSE={metrics['rmse'][joint_index].item():.6f}  "
                f"MAE={metrics['mae'][joint_index].item():.6f}  "
                f"MAX={metrics['max_error'][joint_index].item():.6f}"
            )

        print("-" * 90)

        result = {
            "time": replay_time.float(),
            "target": replay_target,
            "delayed_target": delayed_replay_target,
            "target": replay_target,
            "real_state": replay_real,
            "sim_state": sim_state,
            "sim_velocity": sim_velocity,
            "joint_order": joint_order,
            "control_dt": control_dt,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "max_error": metrics["max_error"],
        }

        output_path = (
            Path(args_cli.output)
            .expanduser()
            .resolve()
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(result, output_path)


        recorded_dt = torch.median(
            source_time[1:] - source_time[:-1]
        )

        recorded_rate_hz = 1.0 / float(
            recorded_dt.item()
        )

        save_plots(
            time=replay_time,
            target=replay_target,
            real=replay_real,
            sim=sim_state,
            joint_order=joint_order,
            active_intervals=active_intervals,
            plot_dir=args_cli.plot_dir,
            plot_rate_hz=recorded_rate_hz,
        )

        print(f"Result saved: {output_path}")
        print(
            f"Plots saved : "
            f"{Path(args_cli.plot_dir).resolve()}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()