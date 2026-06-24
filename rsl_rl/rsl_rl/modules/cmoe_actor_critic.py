# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University). All rights reserved.

import torch
import torch.nn as nn
from torch.distributions import Normal
from .actor_critic import get_activation
from rsl_rl.modules.expert_actor_critic import ExpertActorCritic
from rsl_rl.modules.state_estimator import StateEstimator
from rsl_rl.modules.terrain_estimator import TerrainEstimator

import torch.nn.functional as F

class CMoEActorCritic(nn.Module):
    def __init__(self, num_actor_obs, num_critic_obs, num_one_step_obs, num_actions,
                 expert_hidden_dims=[512, 256, 128], num_prototypes = 32, temperature=0.2, activation='elu',device='cpu', **kwargs):
        
        self.use_map_estimator = kwargs.pop("use_map_estimator", False)
        
        super(CMoEActorCritic, self).__init__()
        self.device = device
        self.num_experts = 5
        self.num_actions = num_actions
        
        estimator_class = eval("StateEstimator")
        self.state_estimator = estimator_class(temporal_steps=10, num_one_step_obs=45, 
                                      use_estimation_loss = True, use_latent_loss = True,
                                      use_detailed_explicit = False,use_map_estimator = self.use_map_estimator)
        
        estimator_class2 = eval("TerrainEstimator")
        self.terrain_estimator = estimator_class2(temporal_steps=10, num_one_step_obs=45, 
                                      use_estimation_loss = True, use_latent_loss = True,
                                      use_detailed_explicit = False,use_map_estimator = self.use_map_estimator)

        
        # Create the expert networks and register them as submodules of nn.Module.
        self.experts = nn.ModuleList([
            self.create_expert_network(num_actor_obs, num_critic_obs, num_one_step_obs, num_actions,
                                       expert_hidden_dims, activation, **kwargs).to(self.device),
            self.create_expert_network(num_actor_obs, num_critic_obs, num_one_step_obs, num_actions,
                                       expert_hidden_dims, activation, **kwargs).to(self.device),
            self.create_expert_network(num_actor_obs, num_critic_obs, num_one_step_obs, num_actions,
                                       expert_hidden_dims, activation, **kwargs).to(self.device),
            self.create_expert_network(num_actor_obs, num_critic_obs, num_one_step_obs, num_actions,
                                       expert_hidden_dims, activation, **kwargs).to(self.device),
            self.create_expert_network(num_actor_obs, num_critic_obs, num_one_step_obs, num_actions,
                                       expert_hidden_dims, activation, **kwargs).to(self.device)
        ])
        
        # Create the gating network.
        self.gating_network = nn.Sequential(
            nn.Linear(45+16+3+77+16, 128),  # input size equals the observation size
            get_activation(activation),  # activation function
            nn.Linear(128, self.num_experts),  # output the weight for each expert
            nn.Softmax(dim=1)  # use Softmax so the expert weights sum to 1
        ).to(self.device)
        
        self.gate_projector = nn.Sequential(
            nn.Linear(self.num_experts, 128),
            get_activation(activation),
            nn.Linear(128, 64),
            get_activation(activation),
            nn.Linear(64, 16),
        ).to(self.device)
        
        self.terrain_projector = nn.Sequential(
            nn.Linear(77+16, 128),
            get_activation(activation),
            nn.Linear(128, 64),
            get_activation(activation),
            nn.Linear(64, 16),
        ).to(self.device)
        
        # Prototype
        self.prototypes = nn.Embedding(num_prototypes, 16)
        
        self.temperature = temperature
        
        self.std = nn.Parameter(torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False
        

    def create_expert_network(self, num_actor_obs, num_critic_obs, num_one_step_obs, num_actions, 
                               expert_hidden_dims, activation, **kwargs):
        # Build each expert network structure using ExpertActorCritic.
        return ExpertActorCritic(num_actor_obs=num_actor_obs,
                              num_critic_obs=num_critic_obs,
                              num_one_step_obs=num_one_step_obs,
                              num_actions=num_actions,
                              **kwargs)
    
    
    def update_estimators(self, obs_batch, next_critic_obs_batch, lr):
    
        estimation_loss, latent_loss, recons_loss, kld_loss = self.state_estimator.update(obs_batch, next_critic_obs_batch, lr)
        estimation_loss2, latent_loss2, recons_loss2, kld_loss2 = self.terrain_estimator.update(obs_batch, next_critic_obs_batch, lr)
        return estimation_loss, latent_loss, recons_loss, kld_loss, estimation_loss2, latent_loss2, recons_loss2, kld_loss2

    def act(self, obs_history=None, **kwargs):
        # Use the gating network to compute the expert weights.
        # obs_history is the regular observation.
        with torch.no_grad():
            explicit, latent = self.state_estimator(obs_history)
        actor_input = obs_history[:,:45]
        actor_input = torch.cat((actor_input, explicit), dim=-1)
        actor_input = torch.cat((actor_input, latent), dim=-1)

        actor_input = torch.cat((actor_input, obs_history[:,-77:]), dim=-1)

        with torch.no_grad():
            latent2 = self.terrain_estimator(obs_history)
        actor_input = torch.cat((actor_input, latent2), dim=-1)

        self.gate_weights = self.gating_network(actor_input)  # (batch_size, num_experts)

        # Get the mean and std of each expert.
        expert_means = []
        expert_stds = []

        for expert in self.experts:
            mean, std = expert.act(actor_input, **kwargs)  # output of each expert
            expert_means.append(mean)
            expert_stds.append(std)

        # Stack the means and stds of all experts into a tensor.
        expert_meansm = torch.stack(expert_means, dim=1)  # (num_experts, batch_size, num_actions)

        weighted_means = (expert_meansm * self.gate_weights.unsqueeze(2)).sum(dim=1)  # weighted mean

        # Create a Gaussian distribution with the new mean and a fresh std.
        self.distribution = Normal(weighted_means, self.std)

        # Sample from the combined distribution and return.
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, obs_history, observations=None):
        explicit, latent = self.state_estimator(obs_history)
        actor_input = obs_history[:,:45]
        actor_input = torch.cat((actor_input, explicit), dim=-1)
        actor_input = torch.cat((actor_input, latent), dim=-1)
        actor_input = torch.cat((actor_input, obs_history[:,-77:]), dim=-1)

        with torch.no_grad():
            latent2 = self.terrain_estimator(obs_history)
        actor_input = torch.cat((actor_input, latent2), dim=-1)

        # Get the action output of each expert.
        expert_outputs = [expert.act_inference(actor_input.to(self.device)) for expert in self.experts]
        expert_means = torch.stack([output for output in expert_outputs], dim=1)

        # Weighted average of the means.
        self.gate_weights = self.gating_network(actor_input)
        weighted_mean = (expert_means * self.gate_weights.unsqueeze(2)).sum(dim=1)  # weighted mean

        return weighted_mean

    def evaluate(self, critic_observations, **kwargs):
        # get the evaluation output of each expert
        expert_outputs = [expert.evaluate(critic_observations.to(self.device), **kwargs) for expert in self.experts]
        value = torch.stack([output for output in expert_outputs], dim=1)

        # weighted average of the values
        value_gate_weights = self.gate_weights.detach()  # do not update the gating network when computing values
        #         [4096, 2, 1]      [4096, 2]
        weighted_value = (value * value_gate_weights.unsqueeze(2)).sum(dim=1)
        
        return weighted_value
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def compute_contrastive_loss(self, obs_history=None, **kwargs):
        with torch.no_grad():
            explicit, latent = self.state_estimator(obs_history)
        actor_input = obs_history[:,:45]
        actor_input = torch.cat((actor_input, explicit), dim=-1)
        actor_input = torch.cat((actor_input, latent), dim=-1)

        actor_input = torch.cat((actor_input, obs_history[:,-77:]), dim=-1)

        with torch.no_grad():
            latent2 = self.terrain_estimator(obs_history)
        actor_input = torch.cat((actor_input, latent2), dim=-1)
        
        gate_input = self.gating_network(actor_input.detach())
        height_input = torch.cat((obs_history[:,-77:], latent2), dim=-1).detach()
        
        gate_z = self.gate_projector(gate_input)
        height_z = self.terrain_projector(height_input) # latent from estimator 2
        
        gate_z = F.normalize(gate_z, dim=-1, p=2)
        height_z = F.normalize(height_z, dim=-1, p=2)
        
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()  # clone the prototype weights
            w = F.normalize(w, dim=-1, p=2)  # L2-normalize the prototype weights
            self.prototypes.weight.copy_(w)  # write the normalized weights back to the prototypes

        score_s = gate_z @ self.prototypes.weight.T
        score_t = height_z @ self.prototypes.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)
        
        contrastive_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        
        return contrastive_loss


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T