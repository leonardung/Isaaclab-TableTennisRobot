# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import datetime
import os

from matplotlib import pyplot as plt
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.assets import Articulation
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply
from isaaclab.sensors import TiledCamera, save_images_to_file

from .tabletennis_camera_env_cfg import TabletennisCameraEnvCfg


class TabletennisCameraEnv(DirectRLEnv):
    cfg: TabletennisCameraEnvCfg

    def __init__(
        self, cfg: TabletennisCameraEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )
        self.total_reward = torch.zeros(self.num_envs, device=self.device)
        self.has_touch_paddle = torch.zeros(self.num_envs, device=self.device).bool()
        self.has_first_bouce = torch.zeros(self.num_envs, device=self.device).bool()
        self.has_first_bouce_prev = torch.zeros(
            self.num_envs, device=self.device
        ).bool()
        self.has_touch_own_table = torch.zeros(self.num_envs, device=self.device).bool()
        self.has_touch_own_table_prev = torch.zeros(
            self.num_envs, device=self.device
        ).bool()
        self.reward_vel_prev = torch.zeros(self.num_envs, device=self.device)
        self.rew_table_success = torch.zeros(self.num_envs, device=self.device)
        self.rew_table_fail = torch.zeros(self.num_envs, device=self.device)
        self.rew_ball_to_floor = torch.zeros(self.num_envs, device=self.device)

        # debug
        self.current_obs = []
        self.current_rew = []
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logdir = os.path.join("logs/plots", now)
        os.makedirs(self.logdir, exist_ok=True)
        self.episode_count = 0

    def _setup_scene(self):

        self._robot = Articulation(self.cfg.robot)
        self._table = RigidObject(self.cfg.table)
        self._ball = RigidObject(self.cfg.ball)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["ball"] = self._ball
        self.scene.sensors["tiled_camera"] = self._tiled_camera

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        self._robot.set_joint_position_target(self.actions * 50)

    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        stop = (
            (self.rew_table_success != 0)
            | (self.rew_table_fail != 0)
            | (self.rew_ball_to_floor != 0)
        )
        return stop, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        reward_vel = (
            -self.ball_linvel[:, 1]
            * self.has_touch_paddle.float()
            * self.cfg.rew_scale_y
            * (self.ball_contact == 0).float()
            * torch.logical_not(self.reward_vel_prev)
        )
        # Add bonus reward if contact happened (binary flag).
        reward_contact = self.cfg.rew_scale_contact * self.ball_contact.float()
        self.rew_table_success = (
            self.has_touch_paddle
            * self.has_touch_opponent_table.float()
            * self.cfg.rew_scale_table_success
        )
        rew_ball_pos = (
            -self.rew_table_success * self.ball_pos[:, 1] * self.cfg.rew_scale_ball_pos
        )
        self.rew_table_fail = (
            self.has_first_bouce_prev
            * self.has_touch_own_table.float()
            * self.cfg.rew_scale_table_fail
        )
        ball_y = self.ball_pos[:, 1]  # (N,)
        mask_fail = self.rew_table_fail != 0  # (N,) boolean
        self.rew_table_fail[mask_fail] += ball_y[mask_fail] + 0.1

        ball_z = self.ball_pos[:, 2]
        ball_to_floor = ball_z < 0.65
        self.rew_ball_to_floor = ball_to_floor * self.cfg.rew_scale_ball_floor

        self.total_reward = (
            reward_contact
            + self.rew_table_success
            - self.rew_table_fail
            - self.rew_ball_to_floor
            + reward_vel
            + rew_ball_pos
        )
        still_false = self.reward_vel_prev == 0
        self.reward_vel_prev[still_false] = reward_vel[still_false]

        self.current_rew.append(
            {
                "reward_contact": reward_contact[0].cpu().numpy(),
                "rew_table_success": self.rew_table_success[0].cpu().numpy(),
                "reward_vel": reward_vel[0].cpu().numpy(),
                "rew_ball_pos": rew_ball_pos[0].cpu().numpy(),
                "rew_table_fail": self.rew_table_fail[0].cpu().numpy(),
                "rew_ball_to_floor": self.rew_ball_to_floor[0].cpu().numpy(),
            }
        )
        return self.total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # save plots
        if 0 in env_ids and len(self.current_obs) != 0 and len(self.current_rew) != 0:
            if self.episode_count % 50 == 0:
                self._plot_last_episode(self.current_obs, self.current_rew)
            self.current_obs = []
            self.current_rew = []
            self.episode_count += 1

        # reset the entries in env_ids
        self.total_reward[env_ids] = 0.0
        self.has_touch_paddle[env_ids] = False
        self.has_first_bouce[env_ids] = False
        self.has_first_bouce_prev[env_ids] = False
        self.has_touch_own_table[env_ids] = False
        self.has_touch_own_table_prev[env_ids] = False
        self.reward_vel_prev[env_ids] = 0.0
        self.rew_table_success[env_ids] = 0.0
        self.rew_table_fail[env_ids] = 0.0
        self.rew_ball_to_floor[env_ids] = 0.0

        # robot joints (only those envs)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)

        # ball (only env_ids)
        ball_state = self._ball.data.default_root_state.clone()[env_ids]
        ball_state[:, :3] += self.scene.env_origins[env_ids]
        # random x-noise and velocity
        pos_noise = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.cfg.ball_pos_x_range
        )
        ball_state[:, 0:1] += pos_noise
        v_x = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.cfg.ball_speed_x_range
        )
        v_y = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.cfg.ball_speed_y_range
        )
        v_z = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.cfg.ball_speed_z_range
        )
        lin_vel = torch.cat((v_x, v_y, v_z), dim=1)
        ang_vel = torch.zeros(len(env_ids), 3, device=self.device)
        ball_state[:, 7:] = torch.cat((lin_vel, ang_vel), dim=1)
        self._ball.write_root_pose_to_sim(ball_state[:, :7], env_ids)
        self._ball.write_root_velocity_to_sim(ball_state[:, 7:], env_ids)

        # table (only env_ids)
        table_state = self._table.data.default_root_state.clone()[env_ids]
        table_state[:, :3] += self.scene.env_origins[env_ids]
        self._table.write_root_pose_to_sim(table_state[:, :7], env_ids)
        self._table.write_root_velocity_to_sim(table_state[:, 7:], env_ids)

        self._compute_intermediate_values()

    def _compute_intermediate_values(self):
        # Compute intermediate values for the robot (if needed)
        self.robot_joint_pos = self._robot.data.joint_pos
        self.robot_joint_vel = self._robot.data.joint_vel

        # For ball: extract global pose and then compute local pose using scene.env_origins
        self.ball_global_pos = (
            self._ball.data.root_pos_w
        )  # Global position in simulation
        self.ball_pos = (
            self._ball.data.root_pos_w - self.scene.env_origins
        )  # Local (offset) position
        self.ball_quat = self._ball.data.root_quat_w
        self.ball_vel = self._ball.data.root_vel_w
        self.ball_linvel = self._ball.data.root_lin_vel_w
        self.ball_angvel = self._ball.data.root_ang_vel_w

        # For table: similar approach to get global and local positions
        self.table_global_pos = self._table.data.root_pos_w
        self.table_pos = self._table.data.root_pos_w - self.scene.env_origins
        self.table_quat = self._table.data.root_quat_w
        self.table_vel = self._table.data.root_vel_w
        self.table_linvel = self._table.data.root_lin_vel_w
        self.table_angvel = self._table.data.root_ang_vel_w

        # --- Compute Paddle Position and Contact ---
        paddle_index = -1
        paddle_index = 6
        paddle_pos = self._robot.data.body_pos_w[:, paddle_index, :]
        paddle_quat = self._robot.data.body_quat_w[:, paddle_index, :]
        # 1) Normalize the quaternion (just in case):
        paddle_quat = paddle_quat / paddle_quat.norm(dim=1, keepdim=True)
        # 2) Build the local offset (0, +0.3, 0) and expand to (N,3):
        local_offset = (
            torch.tensor(
                [0.0, 0.265, 0.0],
                device=paddle_pos.device,
                dtype=paddle_pos.dtype,
            )
            .unsqueeze(0)
            .expand_as(paddle_pos)
        )
        rotated_offset: torch.Tensor = quat_apply(paddle_quat, local_offset)
        # 4) Compute your touch point:
        self.paddle_touch_point = paddle_pos + rotated_offset
        # 5) Compute touch reward:
        distance = torch.norm(self.ball_global_pos - self.paddle_touch_point, dim=1)
        contact_score = (
            self.cfg.contact_threshold - distance
        ) / self.cfg.contact_threshold
        self.ball_contact = torch.clamp(contact_score, min=0.0, max=1.0)
        self.ball_contact = self.ball_contact * ~self.has_touch_paddle
        new_hits = contact_score > 0  # Tensor[N] bool
        still_false = ~self.has_touch_paddle  # Tensor[N] bool
        self.has_touch_paddle[still_false] = new_hits[still_false]

        # --- Compute Contact with table ---
        bx, by, bz = (self.ball_pos[:, 0], self.ball_pos[:, 1], self.ball_pos[:, 2])
        # 3) Load your table‐contact bounds from self
        tcx_min, tcx_max = self.cfg.table_contact_x
        tcy_min, tcy_max = self.cfg.table_contact_y
        tcz_min, tcz_max = self.cfg.table_contact_z

        ncx_min, ncx_max = self.cfg.table_not_contact_x
        ncy_min, ncy_max = self.cfg.table_not_contact_y
        ncz_min, ncz_max = self.cfg.table_not_contact_z

        # 4) Build masks
        self.has_touch_opponent_table = (
            (bx >= tcx_min)
            & (bx <= tcx_max)
            & (by >= tcy_min)
            & (by <= tcy_max)
            & (bz >= tcz_min)
            & (bz <= tcz_max)
        )
        has_touch_own_table_just_now = (
            (bx >= ncx_min)
            & (bx <= ncx_max)
            & (by >= ncy_min)
            & (by <= ncy_max)
            & (bz >= ncz_min)
            & (bz <= ncz_max)
            & (~self.has_touch_own_table_prev)
        )
        self.has_touch_own_table_prev = (
            self.has_touch_own_table | has_touch_own_table_just_now
        )
        self.has_touch_own_table = has_touch_own_table_just_now
        self.has_first_bouce_prev = self.has_first_bouce.clone()
        still_false = ~self.has_first_bouce
        self.has_first_bouce[still_false] = self.has_touch_own_table[still_false]

        self.current_obs.append(
            {
                "has_touch_own_table": self.has_touch_own_table[0].cpu().numpy(),
                "has_touch_own_table_prev": self.has_touch_own_table_prev[0]
                .cpu()
                .numpy(),
                "has_touch_opponent_table": self.has_touch_opponent_table[0]
                .cpu()
                .numpy(),
                "has_first_bouce": self.has_first_bouce[0].cpu().numpy(),
                "has_first_bouce_prev": self.has_first_bouce_prev[0].cpu().numpy(),
                "ball_contact": self.ball_contact[0].cpu().numpy(),
                "has_touch_paddle": self.has_touch_paddle[0].cpu().numpy(),
                "has_first_bouce_prev": self.has_first_bouce_prev[0].cpu().numpy(),
            }
        )

    def _get_observations(self) -> dict:
        # Retrieve and (optionally) normalise the RGB image
        rgb = self._tiled_camera.data.output["rgb"] / 255.0  # (N,H,W,3), float32
        mean = torch.mean(rgb, dim=(1, 2), keepdim=True)
        rgb -= mean  # zero-centre helps RL

        if self.cfg.write_image_to_file:
            save_images_to_file(rgb, "tabletennis_rgb.png")

        return {"policy": rgb}

    def _plot_last_episode(self, obs_list, rew_list):
        # keys
        obs_keys = list(obs_list[0].keys())
        rew_keys = list(rew_list[0].keys())
        nrows = max(len(obs_keys), len(rew_keys))

        # time axis
        t_obs = np.arange(len(obs_list))
        t_rew = np.arange(len(rew_list))

        # create subplots: nrows x 2
        fig, axes = plt.subplots(
            nrows,
            2,
            figsize=(12, 2.5 * nrows),
            sharex=True,
            tight_layout=True,
        )

        # ensure axes is always 2D array
        if nrows == 1:
            axes = axes[np.newaxis, :]

        # plot observations in left column
        for i, key in enumerate(obs_keys):
            ax = axes[i, 0]
            series = np.array([s[key] for s in obs_list])
            ax.plot(t_obs, series)
            ax.set_ylabel(key)
        # blank any extra rows if obs < nrows
        for i in range(len(obs_keys), nrows):
            axes[i, 0].axis("off")

        # plot rewards in right column
        for i, key in enumerate(rew_keys):
            ax = axes[i, 1]
            series = np.array([r[key] for r in rew_list])
            ax.plot(t_rew, series)
            ax.set_ylabel(key)
        # blank extra reward rows
        for i in range(len(rew_keys), nrows):
            axes[i, 1].axis("off")

        # common x-label on bottom row
        axes[-1, 0].set_xlabel("Timestep")
        axes[-1, 1].set_xlabel("Timestep")

        fig.suptitle("Last Episode: Observations (left) & Rewards (right)")
        filename = os.path.join(self.logdir, f"episode_{self.episode_count:03d}.png")
        fig.savefig(filename)
        plt.close(fig)
