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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time


def play(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0, heading=0.0):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 150)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = True
    env_cfg.commands.resampling_time = 60
    env_cfg.commands.easy_terrain_ranges.lin_vel_x = [x_vel, x_vel]
    env_cfg.commands.easy_terrain_ranges.lin_vel_y = [y_vel, y_vel]
    env_cfg.commands.easy_terrain_ranges.ang_vel_yaw = [yaw_vel, yaw_vel]
    env_cfg.commands.easy_terrain_ranges.heading = [heading, heading]
    env_cfg.commands.hard_terrain_ranges.lin_vel_x = [x_vel, x_vel]
    env_cfg.commands.hard_terrain_ranges.lin_vel_y = [y_vel, y_vel]
    env_cfg.commands.hard_terrain_ranges.ang_vel_yaw = [yaw_vel, yaw_vel]
    env_cfg.commands.hard_terrain_ranges.heading = [heading, heading]
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    env.dt = round(env.dt, 2)

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    with torch.inference_mode():
        for i in range(10*int(env.max_episode_length)):
        
            start_time = time.time()

            result = policy(obs.detach())
        
            end_time = time.time()
            execution_time = end_time - start_time

            if isinstance(result, tuple):
                actions = result[0]
                pred_next_obs = result[1]
                pred_scan_dots = result[2]
                env.save_net_pred(pred_next_obs, pred_scan_dots)
            else:
                actions = result

            # env.commands[:, 0] = x_vel
            # env.commands[:, 1] = y_vel
            # # env.commands[:, 2] = yaw_vel
            # env.commands[:, 3] = heading
            obs, _, rews, dones, infos, _, _ ,win= env.step(actions.detach())

            ppo_runner.alg.actor_critic.reset(dones)
            env.reset_buffer(dones)

            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            if PLOT:
                if i < stop_state_log:
                    logger.log_states(
                        {
                            'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                            'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                            'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                            'dof_torque': env.torques[robot_index, joint_index].item(),
                            'command_x': env.commands[robot_index, 0].item(),
                            'command_y': env.commands[robot_index, 1].item(),
                            'command_yaw': env.commands[robot_index, 2].item(),
                            'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                            'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                            'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                            'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                            'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                        }
                    )
                elif i==stop_state_log:
                    logger.plot_states()
                if  0 < i < stop_rew_log:
                    if infos["episode"]:
                        num_episodes = torch.sum(env.reset_buf).item()
                        if num_episodes>0:
                            logger.log_rewards(infos["episode"], num_episodes)
                elif i==stop_rew_log:
                    logger.print_rewards()
                
            if env.episode_length_buf[env.lookat_id].item() * env.dt % 0.5 == 0.0:
                print("time:", env.episode_length_buf[env.lookat_id].item() * env.dt, 
                "cmd vx", env.commands[env.lookat_id, 0].item(),
                "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), 
                "base_height", env.root_states[env.lookat_id, 2].item(),)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    PLOT = False
    args = get_args()
    play(args, x_vel=0.8, y_vel=0.0, yaw_vel=0.0, heading=0.0)
