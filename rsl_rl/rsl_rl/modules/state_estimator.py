# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University). All rights reserved.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class StateEstimator(nn.Module):
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 prop_enc_hidden_dims=[128, 64, 32],
                 dec_hidden_dims=[32, 64, 128],
                 latent_dim = 16,
                 explicit_dim = 3, 
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 kld_weight=0.005,
                 **kwargs):
        
        self.use_estimation_loss = kwargs.pop("use_estimation_loss", True)
        self.use_latent_loss = kwargs.pop("use_latent_loss", True)
        use_detailed_explicit = kwargs.pop("use_detailed_explicit", False)
        self.use_map_estimator = kwargs.pop("use_map_estimator", False)
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
            
        super(StateEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps  #10
        self.num_one_step_obs = num_one_step_obs #45
        self.num_latent = prop_enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.kld_weight = kld_weight

        self.num_prop_obs = self.temporal_steps * self.num_one_step_obs #450
        if self.use_map_estimator:
            self.num_prop_obs = self.num_prop_obs+77
        use_detailed_explicit = True
        if use_detailed_explicit:
            explicit_dim = 3+11#3 + 1 + 12
        self.latent_dim = latent_dim #16

        # Proprioceptive MLP Encoder
        prop_enc_input_dim = 450
        latent_dim = 16 
        explicit_dim = 3
        prop_enc_layers = []
        for l in range(len(prop_enc_hidden_dims)):
            prop_enc_layers += [nn.Linear(prop_enc_input_dim, prop_enc_hidden_dims[l]), activation]
            prop_enc_input_dim = prop_enc_hidden_dims[l]
        self.encoder = nn.Sequential(*prop_enc_layers)

        self.fc_mu = nn.Linear(prop_enc_input_dim, latent_dim) # \mu  mean of the latent space, dimension latent_dim
        self.fc_var = nn.Linear(prop_enc_input_dim, latent_dim) # 2*log(\sigma) variance of the latent space, dimension latent_dim
        self.fc_explicit = nn.Linear(prop_enc_input_dim, explicit_dim) # explicit features, dimension explicit_dim

        # Decoder
        dec_input_dim = 16 + 3 #3+11+16
        dec_output_dim = self.num_one_step_obs  #45
        dec_layers = []
        for l in range(len(dec_hidden_dims)):
            dec_layers += [nn.Linear(dec_input_dim, dec_hidden_dims[l]), activation]
            dec_input_dim = dec_hidden_dims[l]
        dec_layers += [nn.Linear(dec_input_dim, dec_output_dim)]
        self.decoder = nn.Sequential(*dec_layers)

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        explicit, z, mu, log_var = self.encode(obs_history)
        return explicit.detach(), z.detach()

    def forward(self, obs_history):
        obs_history= obs_history[:, :450]
        result = self.encoder(obs_history.detach())
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        explicit = self.fc_explicit(result)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        return explicit.detach(), z.detach()

    def encode(self, obs_history):
        obs_history= obs_history[:, :450]
        result = self.encoder(obs_history.detach())
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        explicit = self.fc_explicit(result)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        return explicit, z, mu, log_var
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return eps * std + mu

    def update(self, obs_history, critic_obs, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        explicit = critic_obs[:, 45:45+3].detach()
        next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3]

        pred_explicit, z, mu, log_var = self.encode(obs_history)
        z = torch.cat((z, pred_explicit), dim = 1)
        pred_next_obs = self.decoder(z)

        recons_loss = F.mse_loss(pred_next_obs, next_obs)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        vae_loss = recons_loss + self.kld_weight * kld_loss

        estimation_loss = F.mse_loss(pred_explicit, explicit)
        losses = self.use_estimation_loss * estimation_loss + self.use_latent_loss * vae_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), vae_loss.item(), recons_loss.item(), kld_loss.item()

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
