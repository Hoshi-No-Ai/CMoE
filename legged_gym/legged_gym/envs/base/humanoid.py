# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from collections import OrderedDict, defaultdict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_zyx_to_quaternion, euler_from_quaternion
from legged_gym.utils.humanoid_terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

import time
from tqdm import tqdm
import trimesh

class Humanoid(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]   

        self.left_foot_sample_points_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.left_foot_sample_points_indices, 0:3]
        self.right_foot_sample_points_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.right_foot_sample_points_indices, 0:3]

        self.contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(self.contact, self.last_contacts) 
        self.last_contacts = self.contact    

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self.adaptive_bootstrapping()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)

        self.reset_idx(env_ids)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.disturbance[:, :, :] = 0.0
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_torques[:] = self.torques[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._draw_height_samples()
            self._draw_pred_height_samples()

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """

        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        roll_cutoff = torch.abs(self.roll) > 1.0
        pitch_cutoff = torch.abs(self.pitch) > 1.0
        # make the height cutoff reasonable and only apply it in specific parkour environments (e.g., gaps)
        height_cutoff = self.root_states[:, 2] < 0.5 
        if_gap = self.env_class[:] == 5
        height_cutoff &= if_gap

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff

        self.win_buf = self.time_out_buf
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_contacts[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        # update height measurements
        self.feet_heights = self._get_feet_heights()
        self.left_foot_sample_heights, self.right_foot_sample_heights = self._get_feet_sample_heights()
        if self.cfg.terrain.measure_heights:
            # update noise values
            self.xy_noise[env_ids] = torch.normal(mean=0, std=0.05, size=(self.height_points.shape[0], 2)).to(self.device).unsqueeze(1).repeat(1, self.height_points.shape[1], 1)[env_ids]
            mask = (self.terrain_levels >= 2)[env_ids]
            # environments that satisfy the condition
            if mask.any():
                selected_env_ids = env_ids[mask]
                self.z_noise[selected_env_ids] = (
                    (torch.rand(self.height_points.shape[0], 1).to(self.device) * 2 - 1)
                    .repeat(1, self.height_points.shape[1])[selected_env_ids]
                )
                # environments that do not satisfy the condition (~mask)
            if (~mask).any():
                unselected_env_ids = env_ids[~mask]
                self.z_noise[unselected_env_ids] = 0
            yaw_noise = torch.empty(self.height_points.shape[0]).uniform_(-0.2, 0.2)
            self.noisy_base_quat[env_ids] = torch.stack([torch.zeros(self.height_points.shape[0]), torch.zeros(self.height_points.shape[0]), torch.sin(yaw_noise / 2), torch.cos(yaw_noise / 2)], dim=1).to(self.device)[env_ids]

            self.measured_heights = self._get_heights()
        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), 1), device=self.device)
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_easy_terrain_command_x"] = self.easy_terrain_command_ranges["lin_vel_x"][1]
            self.extras["episode"]["max_hard_terrain_command_x"] = self.hard_terrain_command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0

    def disturb_heights_extreme(self,heights):
        """
        For each row, randomly pick 4 distinct positions:
        - 2 positions replaced with a random value in [max, 2max - min] (upward disturbance)
        - 2 positions replaced with a random value in [2min - max, min] (downward disturbance)

        Args:
            heights (torch.Tensor): input tensor of shape [B, N]

        Returns:
            torch.Tensor: the disturbed heights (modified in place)
        """
        B, N = heights.shape
        n=4
        assert 2 * n <= N, f"disturbance points per row 2n={2*n} cannot exceed number of columns N={N}"
        row_max = heights.max(dim=1, keepdim=True)[0]  # [B, 1]
        row_min = heights.min(dim=1, keepdim=True)[0]  # [B, 1]

        # upward disturbance interval: [max, 2max - min]
        high_upper = 2 * row_max - row_min
        rand_upper = torch.rand((B, n), device=heights.device)
        rand_upper = rand_upper * (high_upper - row_max) + row_max  # [B, n]

        # downward disturbance interval: [2min - max, min]
        low_lower = 2 * row_min - row_max
        rand_lower = torch.rand((B, n), device=heights.device)
        rand_lower = rand_lower * (row_min - low_lower) + low_lower  # [B, n]

        # randomly select 2n distinct positions per row
        all_indices = torch.multinomial(torch.ones((B, N), device=heights.device), num_samples=2 * n, replacement=False)
        upper_indices = all_indices[:, :n]
        lower_indices = all_indices[:, n:]

        # build row indices
        row_indices = torch.arange(B, device=heights.device).unsqueeze(1).expand(-1, n)  # [B, n]

        # replace in place
        heights[row_indices, upper_indices] = rand_upper
        heights[row_indices, lower_indices] = rand_lower
        return heights

    def compute_observations(self):
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,                     #cmd 3
                                    self.base_ang_vel  * self.obs_scales.ang_vel,                   #angv 3
                                    self.projected_gravity,                                         #G 3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,#q 12
                                    self.dof_vel * self.obs_scales.dof_vel,                         #dq 12
                                    self.actions,                                                   #a 12
                                    ),dim=-1)
        # QS: privileged_obs should not be noisy
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1) #v 3
        if self.cfg.env.use_detailed_privileged_obs:
            current_obs = torch.cat((current_obs, self.base_heights.view(-1,1) * self.obs_scales.base_height, self.feet_pos_base.view(-1,12) * self.obs_scales.feet_pos_base), dim=-1)
        current_obs = torch.cat((current_obs, self.disturbance[:, 0, :]), dim=-1)                   #noise
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.measured_heights, -100, 100.) * self.obs_scales.height_measurements
            # Gaussian noise
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 77)]

            # salt-and-pepper noise: replace a few points per row with extreme values
            heights = self.disturb_heights_extreme(heights)

            current_obs = torch.cat((current_obs, heights), dim=-1)

        self.history_obs = torch.where(
            self.episode_length_buf[:, None] <= 1,
            torch.cat([current_obs[:, :self.num_one_step_obs]] * self.obs_history_length, dim=-1), 
            torch.cat((current_obs[:, :self.num_one_step_obs], self.history_obs[:, :-self.num_one_step_obs]), dim=-1)
        )
        self.obs_buf = torch.cat((self.history_obs, heights), dim=-1)
        self.privileged_obs_buf = torch.where(self.episode_length_buf[:, None] <= 1, torch.cat([current_obs[:, :self.num_one_step_privileged_obs]] * self.priv_obs_history_length, dim=-1)  ,
                                        torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1))

    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        if self.cfg.env.use_detailed_privileged_obs:
            current_obs = torch.cat((current_obs, self.base_heights.view(-1,1) * self.obs_scales.base_height, self.feet_pos_base.view(-1,12) * self.obs_scales.feet_pos_base), dim=-1)
        current_obs = torch.cat((current_obs, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.measured_heights, -100, 100.) * self.obs_scales.height_measurements
            # Gaussian noise
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 77)]
            # salt-and-pepper noise: replace a few points per row with extreme values
            heights = self.disturb_heights_extreme(heights)

            current_obs = torch.cat((current_obs, heights), dim=-1)

        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        start =  time.time()
        print("*"*80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
            self.mesh = trimesh.Trimesh(vertices=self.terrain.vertices, faces=self.terrain.triangles)
            self.mesh.vertices[:, :2] -= self.cfg.terrain.border_size
            self.mesh.vertices /= self.cfg.mesh.resolution
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time.time() - start))
        print("*"*80)
        self._create_envs()

    #------------- Callbacks --------------    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            self.heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - self.heading), -1., 1.)

        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        for i in range(len(self.feet_indices)):
            self.feet_pos_base[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            self.feet_vel_base[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])

        self.base_heights = self._get_base_heights()
        self.feet_heights = self._get_feet_heights()
        self.left_foot_sample_heights, self.right_foot_sample_heights = self._get_feet_sample_heights()
        if self.cfg.terrain.measure_heights:
            # history noise: 0.2 probability of not updating each step
            self.measured_heights = torch.where( 
                (torch.rand(self.measured_heights.size(0), 1) > 0.2).to(self.device), 
                self._get_heights(), 
                self.measured_heights
            )

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        self.joint_pos_target = self.default_dof_pos + actions_scaled

        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # TODO: Modify the base velocities after reset
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        if self.cfg.terrain.measure_heights:
            noise_vec = torch.zeros(9 + 3*self.num_actions + 2 + 77, device=self.device)
        else:
            noise_vec = torch.zeros(9 + 3*self.num_actions + 2, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0. # commands
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:(9 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(9 + self.num_actions):(9 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(9 + 2 * self.num_actions):(9 + 3 * self.num_actions)] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 77)] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.left_foot_sample_points_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.left_foot_sample_points_indices, 0:3]
        self.right_foot_sample_points_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.right_foot_sample_points_indices, 0:3]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.xy_noise = torch.zeros(self.num_envs, self.num_height_points, 2, device=self.device, requires_grad=False)
            self.z_noise = torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
            self.noisy_base_quat = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
            self.measured_heights = self._get_heights()
        self.base_heights = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.feet_heights = self._get_feet_heights()
        self.left_foot_sample_heights, self.right_foot_sample_heights = self._get_feet_sample_heights()
        self.feet_pos_base = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device, requires_grad=False)
        self.feet_vel_base = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device, requires_grad=False)

        self.boot_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            print(f"Setting default joint angle for {name} to {angle}")
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
         
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        self.distance_avg = 0.0  # current average
        self.distance_count = 0  # number of records
        self.calculated_env_ids = set()  # env IDs whose average has already been computed
        self.logged_dis_x = []
        self.logged_dis_y = []

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        hip_roll_joint_names = [s for s in self.dof_names if self.cfg.asset.hip_roll_joint_name in s]
        hip_yaw_joint_names = [s for s in self.dof_names if self.cfg.asset.hip_yaw_joint_name in s]
        left_foot_sample_points_names = [s for s in body_names if self.cfg.asset.left_foot_sample_point_name in s]
        right_foot_sample_points_names = [s for s in body_names if self.cfg.asset.right_foot_sample_point_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        self.hip_roll_indices = torch.zeros(len(hip_roll_joint_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_roll_joint_names):
            self.hip_roll_indices[i] = self.dof_names.index(name)
        self.hip_yaw_indices = torch.zeros(len(hip_yaw_joint_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_yaw_joint_names):
            self.hip_yaw_indices[i] = self.dof_names.index(name)      

        self.left_foot_sample_points_indices = torch.zeros(len(left_foot_sample_points_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_foot_sample_points_names)):
            self.left_foot_sample_points_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_foot_sample_points_names[i])  
        self.right_foot_sample_points_indices = torch.zeros(len(right_foot_sample_points_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_foot_sample_points_names)):
            self.right_foot_sample_points_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_foot_sample_points_names[i])

    def _get_base_heights(self, env_ids=None):
        contact_sum = torch.sum(self.contact_filt, dim=1)
        measured_heights = torch.where(contact_sum > 0, torch.sum(self.feet_pos[..., 2] * self.contact_filt, dim=1) / torch.sum(self.contact_filt, dim=1), 
                                       torch.mean(self.feet_pos[..., 2], dim=1))
        base_height = self.root_states[:, 2] - (measured_heights - self.cfg.rewards.foot_height)
        return base_height
    
    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self.feet_pos[env_ids].clone()
        else:
            points = self.feet_pos.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        feet_height =  self.feet_pos[:, :, 2] - heights

        return feet_height

    def _get_feet_sample_heights(self, env_ids=None):
        if self.cfg.terrain.mesh_type == 'plane':
            return self.left_foot_sample_points_pos[:, :, 2].clone(), self.right_foot_sample_points_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            left_points = self.left_foot_sample_points_pos[env_ids].clone()
            right_points = self.right_foot_sample_points_pos[env_ids].clone()
        else:
            left_points = self.left_foot_sample_points_pos.clone()
            right_points = self.right_foot_sample_points_pos.clone()

        left_points += self.terrain.cfg.border_size
        left_points = (left_points/self.terrain.cfg.horizontal_scale).long()
        right_points += self.terrain.cfg.border_size
        right_points = (right_points/self.terrain.cfg.horizontal_scale).long()

        left_px = left_points[:, :, 0].view(-1)
        left_py = left_points[:, :, 1].view(-1)
        left_px = torch.clip(left_px, 0, self.height_samples.shape[0]-2)
        left_py = torch.clip(left_py, 0, self.height_samples.shape[1]-2)
        right_px = right_points[:, :, 0].view(-1)
        right_py = right_points[:, :, 1].view(-1)
        right_px = torch.clip(right_px, 0, self.height_samples.shape[0]-2)
        right_py = torch.clip(right_py, 0, self.height_samples.shape[1]-2)

        left_heights1 = self.height_samples[left_px, left_py]
        left_heights2 = self.height_samples[left_px+1, left_py]
        left_heights3 = self.height_samples[left_px, left_py+1]
        left_heights = (left_heights1 + left_heights2 + left_heights3) / 3

        right_heights1 = self.height_samples[right_px, right_py]
        right_heights2 = self.height_samples[right_px+1, right_py]
        right_heights3 = self.height_samples[right_px, right_py+1]
        right_heights = (right_heights1 + right_heights2 + right_heights3) / 3

        left_heights = left_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        right_heights = right_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        left_foot_sample_height =  self.left_foot_sample_points_pos[:, :, 2] - left_heights
        right_foot_sample_height = self.right_foot_sample_points_pos[:, :, 2] - right_heights

        return left_foot_sample_height, right_foot_sample_height

    def reset_buffer(self,env_ids):
        self.privileged_obs_buf[env_ids,:]=0
        self.obs_buf[env_ids,:] =0
        self.last_last_actions[env_ids,:] =0
        self.last_actions[env_ids,:] = 0
        self.last_torques[env_ids,:] = 0
        self.last_dof_vel[env_ids,:] = 0
        self.last_root_vel[env_ids,:] = 0
        self.last_contacts[env_ids,:] = 0

    #------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        rew = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        return rew

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew = torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
        return rew

    def _reward_tracking_global_lin_vel(self):
        lin_vel_norm = torch.norm(self.base_lin_vel[:, :2], dim=1)
        rew = torch.minimum(lin_vel_norm, self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        return rew
            
    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(wrap_to_pi(self.commands[:, 3] - self.heading)))
        return rew
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        rew = torch.square(self.base_lin_vel[:, 2])
        return rew
    
    def _reward_lin_vel_y(self):
        # Penalize y axis base linear velocity
        rew = torch.square(self.base_lin_vel[:, 1])
        return rew
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return rew
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_heights - self.cfg.rewards.base_height_target)

    def _reward_foot_clearance(self):
        footpos_in_body_frame = self.feet_pos_base
        footvel_in_body_frame = self.feet_vel_base
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_joint_tracking(self):
        # Penalize joint tracking error
        rew = torch.sum(torch.square(self.joint_pos_target - self.dof_pos), dim=1)
        return rew

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_time *= ~self.contact_filt
        return rew_airTime
        
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             3 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_feet_edge(self):
        # penalize feet on the edge of the terrain
        feet_pos_xy = ((self.feet_pos + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.edge_mask.shape[1]-1)
        feet_at_edge = self.edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
        self.feet_at_edge = self.contact_filt & feet_at_edge
        # don't use rew_feet_edge if the terrain is stair or discrete
        use_no_rew_feet_edge = (self.env_class == 2) | (self.env_class == 3) | (self.env_class == 4)
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        rew[use_no_rew_feet_edge] = 0.
        return rew
    
    def _reward_hip_dof_error(self):
        # QS: Should a penalty for hip_pitch be added?
        rew_hip_roll = torch.sum(torch.square(self.dof_pos[:, self.hip_roll_indices] - self.default_dof_pos[:, self.hip_roll_indices]), dim=1)
        rew_hip_yaw = torch.sum(torch.square(self.dof_pos[:, self.hip_yaw_indices] - self.default_dof_pos[:, self.hip_yaw_indices]), dim=1)
        return rew_hip_roll + rew_hip_yaw
    
    def _reward_no_fly(self):
        # Reward only one feet on ground
        contact_sum = torch.sum(self.contact_filt, dim=1)
        rew = 1.0 * (contact_sum == 1)
        return rew
    
    def _reward_feet_lateral_distance(self):
        lateral_diff = torch.abs(self.feet_pos_base[:, 0, 1] - self.feet_pos_base[:, 1, 1])
        d_min = self.cfg.rewards.min_feet_dist
        d_max = self.cfg.rewards.max_feet_dist
        rew = torch.clamp(lateral_diff - d_min, max=d_max-d_min)
        return rew

    def _reward_feet_slip(self):
        foot_speed_norm = torch.norm(self.feet_vel, dim=2)
        rew = torch.sum(foot_speed_norm * self.contact_filt, dim=1)
        return rew
    
    def _reward_feet_ground_parallel(self):
        # TODO: Consider removing the excessively large sampling points in foot_sample_heights (gap cases) to prevent them from skewing the variance calculation.
        return torch.var(self.left_foot_sample_heights, dim=-1) + torch.var(self.right_foot_sample_heights, dim=-1)

    def _reward_feet_parallel(self):
        sample_heights_diff = self.left_foot_sample_points_pos[:, :, 2] - self.right_foot_sample_points_pos[:, :, 2]
        return torch.var(sample_heights_diff, dim=-1)

    def _reward_feet_contact_momentums(self):
        return torch.sum(self.contact_forces[:, self.feet_indices, 2] * self.feet_vel[..., 2], dim=1)

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(2):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))