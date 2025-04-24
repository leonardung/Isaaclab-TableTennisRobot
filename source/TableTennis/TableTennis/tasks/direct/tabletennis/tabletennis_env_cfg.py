# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import (
    SimulationCfg,
    RigidBodyMaterialCfg,
    UsdFileCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.utils import configclass


@configclass
class TabletennisEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 6
    observation_space = 18
    state_space = 0

    # reset
    ball_speed_x_range = (-1, 1)
    ball_speed_y_range = (3.5, 5)
    ball_speed_z_range = (2.0, 2.2)
    ball_pos_x_range = (-0.2, 0.2)

    # ball_speed_x_range = (-0.1, 0.1)
    # ball_speed_y_range = (6, 6)
    # ball_speed_z_range = (1, 1)
    # ball_pos_x_range = (0, 0)

    table_contact_x = (-0.74, 0.74)
    table_contact_y = (-1.35, -0.1)
    table_contact_z = (0.68, 0.735)

    table_not_contact_x = (-0.74, 0.74)
    table_not_contact_y = (0, 1.36)
    table_not_contact_z = (0.68, 0.735)

    rew_scale_y = 0.5
    rew_scale_contact = 1
    rew_scale_table_success = 5
    rew_scale_table_fail = 2
    rew_scale_ball_floor = 3.5

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.8,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=5.0, replicate_physics=True
    )

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(
            usd_path="C:/Users/Leonard/Documents/Python_projects/Robot_RL/custom_usd/UR10_instanceable_pong.usd",
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 2.7, 0.6),
            rot=(0.70711, 0.0, 0.0, -0.70711),
            joint_pos={
                "shoulder_pan_joint": -0.29,
                "shoulder_lift_joint": -1.212,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": -0.33,
                "wrist_3_joint": 1.39,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )

    # table tennis
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=UsdFileCfg(
            usd_path="C:/Users/Leonard/Documents/Python_projects/Robot_RL/custom_usd/Table_tennis.usd",
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    # table tennis
    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=UsdFileCfg(
            usd_path="C:/Users/Leonard/Documents/Python_projects/Robot_RL/custom_usd/Ping_pong_ball.usd",
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -1.4, 1), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
