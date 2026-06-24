# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).
# Adapted from legged_gym (BSD-3-Clause, Copyright (c) 2021 ETH Zurich, Nikita Rudin
# and NVIDIA CORPORATION & AFFILIATES). See legged_gym/LICENSE.

from legged_gym.envs.base.humanoid_config import HumanoidCfg, HumanoidCfgPPO

class G1CMoECfg( HumanoidCfg ):
    class env( HumanoidCfg.env ):
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 10+77
        num_one_step_privileged_obs = 45 + 3 + 3 + 77 # additional: base_lin_vel, external_forces, scan_dots 139
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        episode_length_s = 20 # episode length in seconds
        
        use_detailed_privileged_obs = False

    class terrain(HumanoidCfg.terrain):
        mesh_type = 'trimesh' # none, plane, heightfield or trimesh
        # TODO: modify measured points to be more reasonable
        measured_points_x = [ -0.1,0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 1mx1.6m rectangle (without center line)
        measured_points_y = [ -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]
        terrain_length = 10.
        terrain_width = 10.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 40 # number of terrain cols (types)

        num_goals = 10

        terrain_dict = {"plane": 0.,   
                        "rough slope": 0.1,
                        "stairs up": 0.1, 
                        "stairs down": 0.1, 
                        "discrete": 0.1, 
                        "parkour_gap": 0.3,
                        "parkour_step_up": 0.,
                        "parkour_step_down": 0.,
                        "parkour_hurdle": 0.1,
                        "mix": 0.1,
                        "narrow_stairs": 0.1,
                        "composite": 0.,}
        terrain_proportions = list(terrain_dict.values())
        easy_terrain = terrain_proportions[0] + terrain_proportions[1]
        non_parkour_terrain = terrain_proportions[0] + terrain_proportions[1] + terrain_proportions[2] + terrain_proportions[3] + terrain_proportions[4]

    class init_state( HumanoidCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,    

           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
        }

    class control( HumanoidCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_reduction = 1.0

    class commands( HumanoidCfg.commands ):
            curriculum = False
            num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            resampling_time = 10. # time before command are changed[s]
            heading_command = True # if true: compute ang vel command from heading error
            class easy_terrain_ranges( HumanoidCfg.commands.easy_terrain_ranges):
                lin_vel_x = [-0.3, 1.0] # min max [m/s]
                lin_vel_y = [-0.3, 0.3]   # min max [m/s]
                ang_vel_yaw = [-1, 1]    # min max [rad/s]
                heading = [-1.6, 1.6]
            class hard_terrain_ranges( HumanoidCfg.commands.hard_terrain_ranges):
                lin_vel_x = [0.3, 1.0] # min max [m/s]
                lin_vel_y = [0.0, 0.0]   # min max [m/s]
                ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
                heading = [0.0, 0.0]

    class asset( HumanoidCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/29dof_urdf/g1_29dof_with_hand_fixed_modify_collision.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        hip_roll_joint_name = "hip_roll"
        hip_yaw_joint_name = "hip_yaw"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
  
    class rewards( HumanoidCfg.rewards ):
        class scales:
            tracking_lin_vel = 2.0
            tracking_yaw = 2.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -2.0
            base_height = -15.0

            feet_stumble = -1.0
            collision = -15.0
            feet_lateral_distance = 0.8
            feet_air_time = 1.0
            feet_ground_parallel = -0.02

            hip_dof_error = -0.5

            dof_acc = -2.5e-7
            dof_vel = -5.0e-4
            torques = -1.0e-5
            action_rate = -0.3
            dof_pos_limits = -2.0
            dof_vel_limits = -1.0
            torque_limits = -1.0
            feet_edge = -1.

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.75
        max_contact_force = 600. # forces above this value are penalized

        foot_height = 0.035
        min_feet_dist = 0.18
        max_feet_dist = 0.24
        max_foot_sample_heights = 0.4

class G1CMoECfgPPO( HumanoidCfgPPO ):
    class policy(HumanoidCfgPPO.policy):
        if_ada_boot = False
        with_explicit_input = True  # whether to use explicit input
        with_latent_input = True    # whether to use latent input
        use_estimation_loss = True
        use_latent_loss = True
        use_detailed_explicit = False # whether to use more detailed explicit input: terrain label
        use_map_estimator = True
    class algorithm( HumanoidCfgPPO.algorithm ):
        num_learning_epochs = 5
        num_mini_batches = 4
        entropy_coef = 0.01
    class runner( HumanoidCfgPPO.runner ):
        max_iterations = 50000
        num_steps_per_env = 24

        save_interval = 200

        run_name = 'cmoe'
        experiment_name = 'g1_cmoe'

  