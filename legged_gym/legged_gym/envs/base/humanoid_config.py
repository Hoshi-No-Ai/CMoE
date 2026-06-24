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

from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HumanoidCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 10
        num_one_step_privileged_obs = 45 + 3 + 3 + 77 # additional: base_lin_vel, external_forces, scan_dots
        # WARNING: should not use base_height when there is a gap
        # TODO: Modify num_privileged_obs to include history of internal observations (obs * nsteps, additional)
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12
        episode_length_s = 20 # episode length in seconds

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # TODO: modify measured points to be more reasonable
        measured_points_x = [ 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 1mx1.6m rectangle (without center line)
        measured_points_y = [ -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]
        terrain_length = 10.
        terrain_width = 10.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 40 # number of terrain cols (types)
        # terrain types for blind
        terrain_dict = {"plane": 0.2, 
                        "rough slope": 0.,
                        "stairs up": 0.2, 
                        "stairs down": 0.2, 
                        "discrete": 0., 
                        "parkour_gap": 0.2,
                        "parkour_step_up": 0.,
                        "parkour_step_down": 0.,
                        "parkour_hurdle": 0.2,}
        terrain_proportions = list(terrain_dict.values())
        easy_terrain = terrain_proportions[0] + terrain_proportions[1]
        non_parkour_terrain = terrain_proportions[0] + terrain_proportions[1] + terrain_proportions[2] + terrain_proportions[3] + terrain_proportions[4]

    class domain_rand(LeggedRobotCfg.domain_rand):

        pass

    class control(LeggedRobotCfg.control):
        decimation = 4 # 100Hz

    class asset(LeggedRobotCfg.asset):
        name = "humanoid"  # actor name
        hip_roll_joint_name = "hip_roll"
        hip_yaw_joint_name = "hip_yaw"
        left_foot_sample_point_name = "left_foot_sample_point"
        right_foot_sample_point_name = "right_foot_sample_point"
        
    class sim(LeggedRobotCfg.sim):
        dt =  0.005
            
class HumanoidCfgPPO(LeggedRobotCfgPPO):
    pass