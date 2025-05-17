import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from dm_control import suite

# Actor network definition (must match training)
class Pi_FC(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Pi_FC, self).__init__()
        self.trunk = nn.ModuleList([
            nn.Linear(obs_size, 256),
            nn.Linear(256, 256),
        ])
        self.mu_head      = nn.Linear(256, action_size)
        self.log_std_head = nn.Linear(256, action_size)
        self.log_std_min, self.log_std_max = -20.0, 2.0

    def forward(self, x, deterministic=False, with_logprob=False):
        h = x
        for layer in self.trunk:
            h = F.relu(layer(h))
        mu = self.mu_head(h)
        if deterministic:
            return torch.tanh(mu), None
        ls   = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std  = torch.exp(ls)
        dist = Normal(mu, std)
        z    = dist.rsample()
        if with_logprob:
            lp   = dist.log_prob(z).sum(dim=-1)
            corr = (2 * (np.log(2) - z - F.softplus(-2*z))).sum(dim=-1)
            lp   = lp - corr
        else:
            lp = None
        return torch.tanh(z), lp

class Agent(object):
    """Loads only the SAC actor’s weights and returns deterministic actions."""
    def __init__(self):
        # signature unchanged
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)

        # rebuild env spec to infer dims
        domain, task, seed = "humanoid", "walk", 0
        env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        obs_spec = env.observation_spec()
        act_spec = env.action_spec()
        obs_size   = int(sum(np.prod(obs_spec[k].shape) for k in obs_spec))
        action_dim = int(np.prod(act_spec.shape))

        # build actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor  = Pi_FC(obs_size, action_dim).to(self.device).double()

        # --- load the actor-only weights ---
        actor_ckpt = "../sac_actor_2250.pth"
        state_dict = torch.load(actor_ckpt, map_location=self.device)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    def act(self, observation):
        # signature unchanged
        with torch.no_grad():
            x = torch.tensor(observation, dtype=torch.float64, device=self.device).unsqueeze(0)
            a, _ = self.actor(x)  
        return a.cpu().numpy()[0]
