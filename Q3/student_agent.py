import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from dm_control import suite
import math
from dmc import make_dmc_env

def make_env():
     # Create environment with state observations
     env_name = "humanoid-walk"
     env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
     return env

# Actor network definition (must match training)
class Pi_FC(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Pi_FC, self).__init__()
        self.trunk = nn.ModuleList([
            nn.Linear(obs_size, 256),
            nn.Linear(256, 256),
        ])
        self.mu_head  = nn.Linear(256, action_size)
        self.log_std_head = nn.Linear(256, action_size)
        self.log_std_min, self.log_std_max = -20.0, 2.0

    def forward(self, x, deterministic=False, with_logprob=False):
        h = x
        for layer in self.trunk:
            h = F.relu(layer(h))
        mu = self.mu_head(h)
        if deterministic:
            return torch.tanh(mu), None
        ls = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std  = torch.exp(ls)
        dist = Normal(mu, std)
        z  = dist.rsample()
        if with_logprob:
            lp = dist.log_prob(z).sum(dim=-1)
            corr = (2 * (np.log(2) - z - F.softplus(-2*z))).sum(dim=-1)
            lp = lp - corr
        else:
            lp = None
        return torch.tanh(z), lp

class Agent(object):
    """Loads only the SAC actor’s weights and returns deterministic actions."""
    def __init__(self):
        self.env = make_env()
        self.obs_size = math.prod(self.env.observation_space.shape)
        self.action_size = math.prod(self.env.action_space.shape)

        # build actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor  = Pi_FC(self.obs_size, self.action_size).to(self.device).double()

        # --- load the actor-only weights ---
        actor_ckpt = "../sac_actor_2_1500.pth"
        state_dict = torch.load(actor_ckpt, map_location=self.device)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    def act(self, observation):
        # print("observation shape: ", observation.shape, flush=True)
        with torch.no_grad():
            x = torch.tensor(observation, dtype=torch.float64, device=self.device).unsqueeze(0)
            a, _ = self.actor(x, deterministic=True)
        return a.cpu().numpy()[0]
