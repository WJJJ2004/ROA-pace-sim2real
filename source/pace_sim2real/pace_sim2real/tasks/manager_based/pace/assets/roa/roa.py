# assets/roa/roa.py
from pathlib import Path
import math

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg

ROA_ASSETS_DIR = Path(__file__).resolve().parent

USD_PATH = f"{ROA_ASSETS_DIR}/roa_deploy/roa_deploy.usd"

# ROA_INIT_JOINT_POS = {
#     "left_hip_pitch": math.radians(-20.0),
#     "left_hip_roll": 0.0,
#     "left_hip_yaw": 0.0,
#     "left_knee_pitch": math.radians(50.0),
#     "left_ankle_pitch": math.radians(-30.0),
#     "left_ankle_roll": 0.0,

#     "right_hip_pitch": math.radians(20.0),
#     "right_hip_roll": 0.0,
#     "right_hip_yaw": 0.0,
#     "right_knee_pitch": math.radians(-50.0),
#     "right_ankle_pitch": math.radians(30.0),
#     "right_ankle_roll": 0.0,
# }

_RSU_KVALUE = 1.37
GLOBAL_STATIC_FRICTION = 0.05
GLOBAL_DYNAMIC_FRICTION = 0.01
GLOBAL_VISCOUS_FRICTION = 0.5
GLOBAL_MIN_DELAY = 0
GLOBAL_MAX_DELAY = 4
"""
NOTE:
RSU Equivalent PD Gain Analysis -> RSU Gain's Kinematic relationship

roll / pitch stiffness ratio = 1.37

roll / pitch damping ratio = 1.37

Robit Notion Link: https://app.notion.com/p/robitkw/RSU-Equivalent-PD-Gain-Analysis-379a551c9cc080ae8dfac45f2c8132a7
"""

_RSU_KVALUE = 1.37


ROA_ACTUATORS = {
    "robstride_03": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_hip_roll",
            ".*_hip_yaw",
        ],
        stiffness={
            ".*_hip_roll": 200.0,
            ".*_hip_yaw": 100.0,
        },
        damping={
            ".*_hip_roll": 26.387,
            ".*_hip_yaw": 3.419,
        },

        # PACE 초기 설정과 동일하게 명시
        armature={
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

        # Actuator model torque limit
        effort_limit={
            ".*": 42.0,
        },

        # Physics solver limits
        effort_limit_sim={
            ".*": 42.0,
        },
        velocity_limit_sim={
            ".*": 18.849,
        },

        min_delay=GLOBAL_MIN_DELAY,
        max_delay=GLOBAL_MAX_DELAY,
    ),

    "robstride_04": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_hip_pitch",
            ".*_knee_pitch",
        ],
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
        friction={
            ".*": GLOBAL_STATIC_FRICTION,
        },
        dynamic_friction={
            ".*": GLOBAL_DYNAMIC_FRICTION,
        },
        viscous_friction={
            ".*": GLOBAL_VISCOUS_FRICTION,
        },

        effort_limit={
            ".*": 84.0,
        },
        effort_limit_sim={
            ".*": 84.0,
        },
        velocity_limit_sim={
            ".*": 17.488,
        },

        min_delay=GLOBAL_MIN_DELAY,
        max_delay=GLOBAL_MAX_DELAY,
    ),

    "rsu": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_ankle_pitch",
            ".*_ankle_roll",
        ],
        stiffness={
            ".*_ankle_pitch": 30.0,
            ".*_ankle_roll": 30.0 * _RSU_KVALUE,
        },
        damping={
            ".*_ankle_pitch": 2.5,
            ".*_ankle_roll": 2.5 * _RSU_KVALUE,
        },
        armature={
            ".*": 0.02,
        },
        friction={
            ".*": GLOBAL_STATIC_FRICTION * 20.0,  # RSU friction is 20x higher than other joints
        },
        dynamic_friction={
            ".*": GLOBAL_DYNAMIC_FRICTION * 20.0,  # RSU friction is 20x higher than other joints
        },
        viscous_friction={
            ".*": GLOBAL_VISCOUS_FRICTION,
        },

        effort_limit={
            ".*": 11.9,
        },
        effort_limit_sim={
            ".*": 11.9,
        },
        velocity_limit_sim={
            ".*": 5.0,
        },

        min_delay=GLOBAL_MIN_DELAY,
        max_delay=GLOBAL_MAX_DELAY,
    ),
}

ROA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1_000.0,
            max_angular_velocity=1_000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        # joint_pos=ROA_INIT_JOINT_POS,
        # joint_vel={".*": 0.0},
    ),
    actuators=ROA_ACTUATORS,
    soft_joint_pos_limit_factor=0.95,
)