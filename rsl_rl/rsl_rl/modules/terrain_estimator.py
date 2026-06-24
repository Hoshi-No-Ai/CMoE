# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University). All rights reserved.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TerrainEstimator(nn.Module):
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
            
        super(TerrainEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps  # 10
        self.num_one_step_obs = num_one_step_obs  # 45
        self.num_latent = prop_enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.kld_weight = kld_weight

        self.num_prop_obs = self.temporal_steps * self.num_one_step_obs  # 450



        # Proprioceptive MLP Encoder
        prop_enc_input_dim = 77
        latent_dim = 16 
        explicit_dim = 11
        prop_enc_layers = []
        for l in range(len(prop_enc_hidden_dims)):
            prop_enc_layers += [nn.Linear(prop_enc_input_dim, prop_enc_hidden_dims[l]), activation]
            prop_enc_input_dim = prop_enc_hidden_dims[l]
        self.encoder = nn.Sequential(*prop_enc_layers)

        self.fc_mu = nn.Linear(prop_enc_input_dim, latent_dim)  # \mu: outputs the latent-space mean (mu), dim = latent_dim
        self.fc_var = nn.Linear(prop_enc_input_dim, latent_dim)  # 2*log(\sigma): outputs the latent-space variance (2*log(sigma)), dim = latent_dim

        # Decoder
        dec_input_dim = 16
        dec_output_dim = 77
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
        z, mu, log_var = self.encode(obs_history)
        return z.detach()

    def forward(self, obs_history):

        obs_history= obs_history[:, 450:450+77]
        result = self.encoder(obs_history.detach())
        mu = self.fc_mu(result)
        return mu.detach()

    def encode(self, obs_history):
        obs_history = obs_history[:, 450:450+77]
        result = self.encoder(obs_history.detach())
        mu = self.fc_mu(result)
        return mu
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return eps * std + mu

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        next_obs = obs_history.detach()[:, 450:450+77]

        z = self.encode(obs_history)
        pred_next_obs = self.decoder(z)

        recons_loss = F.mse_loss(pred_next_obs, next_obs)
        vae_loss = recons_loss

        losses = self.use_latent_loss * vae_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return 0, vae_loss.item(), 0, 0 

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