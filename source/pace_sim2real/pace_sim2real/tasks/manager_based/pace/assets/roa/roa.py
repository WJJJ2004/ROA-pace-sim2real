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

"""
NOTE:
RSU Equivalent PD Gain Analysis -> RSU Gain's Kinematic relationship

roll / pitch stiffness ratio = 1.37

roll / pitch damping ratio = 1.37

Robit Notion Link: https://app.notion.com/p/robitkw/RSU-Equivalent-PD-Gain-Analysis-379a551c9cc080ae8dfac45f2c8132a7
"""


_JOINT_META = {

    # ────────────── LEFT LEG ──────────────
    "left_hip_pitch": {
        "kp": 150.0,
        "kd": 24.722,
        "torque": 84.0,
        "vmax": 8.0,
        "arm": 0.0004,
    },
    "left_hip_roll": {
        "kp": 200.0,
        "kd": 26.387,
        "torque": 42.0,
        "vmax": 8.0,
        "arm": 0.0004,
    },
    "left_hip_yaw": {
        "kp": 100.0,
        "kd": 3.419,
        "torque": 42.0,
        "vmax": 8.0,
        "arm": 0.0004,
    },
    "left_knee_pitch": {
        "kp": 150.0,
        "kd": 8.654,
        "torque": 84.0,
        "vmax": 8.0,
        "arm": 0.0004,
    },
    "left_ankle_pitch": {
        "kp": 25.0,
        "kd": 1.2,
        "torque": 11.9,
        "vmax": 5.0,
        "arm": 0.02,
    },
    "left_ankle_roll": {
        "kp": 25.0 * _RSU_KVALUE,
        "kd": 1.2 * _RSU_KVALUE,
        "torque": 11.9,
        "vmax": 5.0,
        "arm": 0.02,
    },

    # ────────────── RIGHT LEG ──────────────
    "right_hip_pitch": {
        "kp": 150.0,
        "kd": 24.722,
        "torque": 84.0,
        "vmax": 8.0,
        "arm": 0.0004,
    },
    "right_hip_roll": {
        "kp": 200.0,
        "kd": 26.387,
        "torque": 42.0,
        "vmax": 8.0,
        "arm": 0.0004,
    },
    "right_hip_yaw": {
        "kp": 100.0,
        "kd": 3.419,
        "torque": 42.0,
        "vmax": 8.0,
        "arm": 0.0004,
    },
    "right_knee_pitch": {
        "kp": 150.0,
        "kd": 8.654,
        "torque": 84.0,
        "vmax": 8.0,
        "arm": 0.0004,
    },
    "right_ankle_pitch": {
        "kp": 30.0,
        "kd": 2.5,
        "torque": 11.9,
        "vmax": 5.0,
        "arm": 0.02,
    },
    "right_ankle_roll": {
        "kp": 25.0 * _RSU_KVALUE,
        "kd": 1.2 * _RSU_KVALUE,
        "torque": 11.9,
        "vmax": 5.0,
        "arm": 0.02,
    },
}

ROA_ACTUATORS = {
    jn: DelayedPDActuatorCfg(
        joint_names_expr=[jn],
        effort_limit=meta["torque"],
        velocity_limit=meta["vmax"],
        stiffness={jn: meta["kp"]},
        damping={jn: meta["kd"]},
        armature=meta["arm"],
        min_delay=6,
        max_delay=12,
    )
    for jn, meta in _JOINT_META.items()
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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
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