# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass
from .assets.roa.roa import ROA_CFG
from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch

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

ROA_RS03_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[
        ".*_hip_roll",
        ".*_hip_yaw",
    ],
    saturation_effort=60.0,
    effort_limit=42.0,
    velocity_limit=18.849,
    stiffness={
        ".*_hip_roll": 200.0,
        ".*_hip_yaw": 100.0,
    },
    damping={
        ".*_hip_roll": 26.387,
        ".*_hip_yaw": 3.419,
    },
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},          # static friction coefficient (Nm)
    dynamic_friction={".*": 0.0},  # dynamic friction coefficient (Nm)
    viscous_friction={".*": 0.0},  # viscous friction coefficient (Nm s/rad)
    max_delay=10,                  # max delay in simulation steps
)

ROA_RS04_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[
        ".*_hip_pitch",
        ".*_knee_pitch",
    ],
    saturation_effort=120.0,
    effort_limit=84.0,
    velocity_limit=17.488,
    stiffness={
        ".*_hip_pitch": 150.0,
        ".*_knee_pitch": 150.0,
    },
    damping={
        ".*_hip_pitch": 24.722,
        ".*_knee_pitch": 8.654,
    },
    armature={
        ".*": 0.0004,
    },
    encoder_bias={
        ".*": 0.0,
    },
    friction={
        ".*": 0.0,
    },
    dynamic_friction={
        ".*": 0.0,
    },
    viscous_friction={
        ".*": 0.0,
    },
    max_delay=10,
)

# TODO 정확한 값으로 업데이트 필요 (액추에이터 -> 가상관절 토크 전파)
ROA_RSU_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[
        ".*_ankle_pitch",
        ".*_ankle_roll",
    ],
    saturation_effort=11.9,
    effort_limit=11.9,
    velocity_limit=5.0,
    stiffness={
        ".*_ankle_pitch": 30.0,
        ".*_ankle_roll": 30.0 * 1.37,
    },
    damping={
        ".*_ankle_pitch": 2.5,
        ".*_ankle_roll": 2.5 * 1.37,
    },
    armature={
        ".*": 0.02,
    },
    encoder_bias={
        ".*": 0.0,
    },
    friction={
        ".*": 0.0,
    },
    dynamic_friction={
        ".*": 0.0,
    },
    viscous_friction={
        ".*": 0.0,
    },
    max_delay=10,
)

@configclass
class ROAPaceCfg(PaceCfg):
    """Pace configuration for ROA robot."""   
    robot_name: str = "roa_sim"
    data_dir: str = "roa_sim/chirp_data.pt"  # located in pace_sim2real/data/roa_sim/chirp_data.pt
    joint_order: list[str] = ROA_JOINT_ORDER

    bounds_params: torch.Tensor = torch.zeros((49, 2))

    def __post_init__(self):
        n = len(self.joint_order)

        armature_slice = slice(0, n)
        viscous_slice = slice(n, 2 * n)
        friction_slice = slice(2 * n, 3 * n)
        bias_slice = slice(3 * n, 4 * n)
        delay_idx = 4 * n

        # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[armature_slice, 0] = 1.0e-5
        self.bounds_params[armature_slice, 1] = 1.0

        # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[viscous_slice, 0] = 0.0
        self.bounds_params[viscous_slice, 1] = 7.0

        # friction between 0.0 - 0.5
        self.bounds_params[friction_slice, 0] = 0.0
        self.bounds_params[friction_slice, 1] = 0.5

        # bias between -0.1 - 0.1 [rad]
        self.bounds_params[bias_slice, 0] = -0.1
        self.bounds_params[bias_slice, 1] = 0.1

        # delay between 0.0 - 10.0 [sim steps]
        self.bounds_params[delay_idx, 0] = 0.0
        self.bounds_params[delay_idx, 1] = 10.0

@configclass
class ROAPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for the ROA robot in the PACE environment."""

    robot: ArticulationCfg = ROA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        actuators={
            "robstride_03": ROA_RS03_PACE_ACTUATOR_CFG,
            "robstride_04": ROA_RS04_PACE_ACTUATOR_CFG,
            "rsu": ROA_RSU_PACE_ACTUATOR_CFG,
        },
    )


@configclass
class ROAPaceEnvCfg(PaceSim2realEnvCfg):
    """PACE system-identification environment for ROA."""

    scene: ROAPaceSceneCfg = ROAPaceSceneCfg()
    sim2real: ROAPaceCfg = ROAPaceCfg()

    def __post_init__(self):
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.decimation = 1  # 400Hz control

