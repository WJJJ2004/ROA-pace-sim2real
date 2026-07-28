# © 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

"""Replay a real-robot target trajectory in Isaac Lab.

The same target trajectory is compared against:
1. Real robot joint state
2. Isaac Sim joint state
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

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


def load_replay_data(config_path: str) -> dict:
    """Load target and real trajectories from config.pt."""

    path = Path(config_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    config = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = [
        "time",
        "des_dof_pos",
        "dof_pos",
        "joint_order",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in config
    ]

    if missing_keys:
        raise KeyError(
            f"Missing keys in config.pt: {missing_keys}\n"
            f"Available keys: {list(config.keys())}"
        )

    time = torch.as_tensor(
        config["time"],
        dtype=torch.float64,
    )

    target = torch.as_tensor(
        config["des_dof_pos"],
        dtype=torch.float32,
    )

    real = torch.as_tensor(
        config["dof_pos"],
        dtype=torch.float32,
    )

    joint_order = list(config["joint_order"])

    if time.ndim != 1:
        raise ValueError(
            f"time must be [N], got {tuple(time.shape)}"
        )

    if target.ndim != 2:
        raise ValueError(
            f"des_dof_pos must be [N, J], got {tuple(target.shape)}"
        )

    if real.ndim != 2:
        raise ValueError(
            f"dof_pos must be [N, J], got {tuple(real.shape)}"
        )

    if target.shape != real.shape:
        raise ValueError(
            f"Target/Real shape mismatch: "
            f"{tuple(target.shape)} vs {tuple(real.shape)}"
        )

    if len(time) != target.shape[0]:
        raise ValueError(
            f"Time/sample mismatch: "
            f"time={len(time)}, trajectory={target.shape[0]}"
        )

    if len(joint_order) != target.shape[1]:
        raise ValueError(
            f"Joint count mismatch: "
            f"joint_order={len(joint_order)}, "
            f"trajectory={target.shape[1]}"
        )

    # Start replay time from zero.
    time = time - time[0]

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


def calculate_metrics(
    sim: torch.Tensor,
    real: torch.Tensor,
) -> dict:
    """Calculate simulation-to-real position error."""

    error = sim - real

    return {
        "rmse": torch.sqrt(
            torch.mean(error.square(), dim=0)
        ),
        "mae": torch.mean(
            torch.abs(error),
            dim=0,
        ),
        "max_error": torch.max(
            torch.abs(error),
            dim=0,
        ).values,
    }


def save_plots(
    time: torch.Tensor,
    target: torch.Tensor,
    real: torch.Tensor,
    sim: torch.Tensor,
    joint_order: list[str],
    plot_dir: str,
) -> None:
    """Save target, real and sim trajectories."""

    output_dir = Path(plot_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    time_np = time.numpy()
    target_np = target.numpy()
    real_np = real.numpy()
    sim_np = sim.numpy()

    for joint_index, joint_name in enumerate(joint_order):
        figure = plt.figure(figsize=(11, 5))

        plt.plot(
            time_np,
            target_np[:, joint_index],
            label="Target",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

        plt.plot(
            time_np,
            real_np[:, joint_index],
            label="Real state",
            linewidth=2.0,
        )

        plt.plot(
            time_np,
            sim_np[:, joint_index],
            label="Sim state",
            linewidth=2.0,
        )

        plt.title(f"Target-to-state response: {joint_name}")
        plt.xlabel("Time [s]")
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

    # -------------------------------------------------------------------------
    # Create Isaac Lab environment
    # -------------------------------------------------------------------------

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
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

        print("=" * 80)
        print(f"Task             : {args_cli.task}")
        print(f"Device           : {device}")
        print(f"Physics dt       : {base_env.physics_dt:.9f} s")
        print(f"Control dt       : {control_dt:.9f} s")
        print(f"Control rate     : {control_rate:.3f} Hz")
        print(f"Recorded samples : {len(source_time)}")
        print(f"Recorded duration: {source_time[-1].item():.3f} s")
        print("=" * 80)

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
            replay_target[0]
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

        print("\nStarting target replay")

        for step_index in range(number_of_steps):
            action = (
                replay_target[step_index]
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

        # ---------------------------------------------------------------------
        # Compare results
        # ---------------------------------------------------------------------

        metrics = calculate_metrics(
            sim_state,
            replay_real,
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

        save_plots(
            time=replay_time,
            target=replay_target,
            real=replay_real,
            sim=sim_state,
            joint_order=joint_order,
            plot_dir=args_cli.plot_dir,
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