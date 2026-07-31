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

GLOBAL_STATIC_FRICTION = 0.0
GLOBAL_DYNAMIC_FRICTION = 0.0
GLOBAL_VISCOUS_FRICTION = 0.0

# GLOBAL_STATIC_FRICTION = 0.075
# GLOBAL_DYNAMIC_FRICTION = 0.01
# GLOBAL_VISCOUS_FRICTION = 0.8
GLOBAL_ARMATURE_SCALE = 1.0 # float type, scale factor for armature inertia (unit kgm2)

GLOBAL_MAX_DELAY = 4 # int type, max Torque delay in simulation steps (unit 2.5 milliseconds) 

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
    armature={
        ".*": 0.04 *GLOBAL_ARMATURE_SCALE,
    },
    encoder_bias={".*": 0.0},
    friction={".*": GLOBAL_STATIC_FRICTION},          # static friction coefficient (Nm)
    dynamic_friction={".*": GLOBAL_DYNAMIC_FRICTION},  # dynamic friction coefficient (Nm)
    viscous_friction={".*": GLOBAL_VISCOUS_FRICTION},  # viscous friction coefficient (Nm s/rad)
    max_delay=GLOBAL_MAX_DELAY,                  # max delay in simulation steps
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
        ".*": 0.04 * GLOBAL_ARMATURE_SCALE,
    },
    encoder_bias={
        ".*": 0.0,
    },
    friction={
        ".*": GLOBAL_STATIC_FRICTION,
    },
    dynamic_friction={
        ".*": GLOBAL_DYNAMIC_FRICTION,
    },
    viscous_friction={
        ".*": GLOBAL_VISCOUS_FRICTION,
    },
    max_delay=GLOBAL_MAX_DELAY,
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
        ".*": 0.02 * GLOBAL_ARMATURE_SCALE,
    },
    encoder_bias={
        ".*": 0.0,
    },
    friction={
        ".*": GLOBAL_STATIC_FRICTION,
    },
    dynamic_friction={
        ".*": GLOBAL_DYNAMIC_FRICTION,
    },
    viscous_friction={
        ".*": GLOBAL_VISCOUS_FRICTION,
    },
    max_delay=GLOBAL_MAX_DELAY,
)

def print_roa_pace_config(env_cfg) -> None:
    """현재 적용된 ROA PACE 주요 설정을 콘솔에 출력한다."""

    separator = "=" * 78
    subsection = "-" * 78

    print(f"\n{separator}")
    print("                     ROA PACE CONFIGURATION")
    print(separator)

    print("[Simulation]")
    print(f"  Simulation dt       : {env_cfg.sim.dt:.6f} s")
    print(f"  Simulation frequency: {1.0 / env_cfg.sim.dt:.2f} Hz")
    print(f"  Decimation          : {env_cfg.decimation}")
    print(
        f"  Control frequency   : "
        f"{1.0 / (env_cfg.sim.dt * env_cfg.decimation):.2f} Hz"
    )

    print(subsection)
    print("[Global Actuator Parameters]")
    print(f"  Static friction     : {GLOBAL_STATIC_FRICTION:.6f} Nm")
    print(f"  Dynamic friction    : {GLOBAL_DYNAMIC_FRICTION:.6f} Nm")
    print(f"  Viscous friction    : {GLOBAL_VISCOUS_FRICTION:.6f} Nm·s/rad")
    print(f"  Armature scale      : {GLOBAL_ARMATURE_SCALE:.6f}")
    print(f"  Maximum delay       : {GLOBAL_MAX_DELAY} sim step(s)")
    print(
        f"  Maximum delay time  : "
        f"{GLOBAL_MAX_DELAY * env_cfg.sim.dt * 1000.0:.3f} ms"
    )

    actuator_configs = {
        "RobStride RS03": ROA_RS03_PACE_ACTUATOR_CFG,
        "RobStride RS04": ROA_RS04_PACE_ACTUATOR_CFG,
        "RSU": ROA_RSU_PACE_ACTUATOR_CFG,
    }

    for actuator_name, actuator_cfg in actuator_configs.items():
        print(subsection)
        print(f"[{actuator_name}]")

        print(f"  Joint expressions   : {actuator_cfg.joint_names_expr}")
        print(f"  Saturation effort   : {actuator_cfg.saturation_effort}")
        print(f"  Effort limit        : {actuator_cfg.effort_limit}")
        print(f"  Velocity limit      : {actuator_cfg.velocity_limit}")

        print(f"  Stiffness           : {actuator_cfg.stiffness}")
        print(f"  Damping             : {actuator_cfg.damping}")
        print(f"  Armature            : {actuator_cfg.armature}")
        print(f"  Static friction     : {actuator_cfg.friction}")
        print(f"  Dynamic friction    : {actuator_cfg.dynamic_friction}")
        print(f"  Viscous friction    : {actuator_cfg.viscous_friction}")
        print(f"  Encoder bias        : {actuator_cfg.encoder_bias}")
        print(f"  Maximum delay       : {actuator_cfg.max_delay} sim step(s)")

    print(subsection)
    print("[PACE Dataset]")
    print(f"  Robot name          : {env_cfg.sim2real.robot_name}")
    print(f"  Data path           : {env_cfg.sim2real.data_dir}")
    print(f"  Number of joints    : {len(env_cfg.sim2real.joint_order)}")
    print(f"  Joint order         : {env_cfg.sim2real.joint_order}")

    print(separator)
    print("                 ROA PACE CONFIGURATION LOADED")
    print(f"{separator}\n")

@configclass
class ROAPaceCfg(PaceCfg):
    """Pace configuration for ROA robot."""   
    robot_name: str = "roa_sim"
    data_dir: str = "roa_sim/chirp_data_Hip.pt"  # located in pace_sim2real/data/roa_sim/chirp_data.pt
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
        self.bounds_params[delay_idx, 1] = 4.0

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
    
    # 로그 정리 필요    
    # print(f"[INFO]: ROA robot configuration: {robot}")
    # print(f"[INFO]: ROA robot actuator configuration: {robot.actuators}")



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
        print_roa_pace_config(self)