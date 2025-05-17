import gymnasium
import numpy as np
import torch
import torch.nn as nn
from dmc import make_dmc_env

# —————————————————————————————————————————————
# ActorCritic must match your training architecture
# —————————————————————————————————————————————
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.mu_layer    = nn.Linear(hidden_dim, act_dim)
        self.log_std     = nn.Parameter(torch.zeros(act_dim))
        self.value_layer = nn.Linear(hidden_dim, 1)
        self.max_action  = max_action

    def forward(self, x):
        h = self.shared(x)
        mu    = torch.tanh(self.mu_layer(h)) * self.max_action
        std   = torch.exp(self.log_std)
        val   = self.value_layer(h)
        return mu, std, val

# —————————————————————————————————————————————
# Your Agent stub, now using the trained PPO policy
# —————————————————————————————————————————————
class Agent(object):
    def __init__(self):
        # keep the same Box spec
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)

        # build a dummy env to get dims & bounds
        env = make_dmc_env("cartpole-balance", np.random.randint(0, 1000000), flatten=True, use_pixels=False)
        obs_dim   = env.observation_space.shape[0]
        act_dim   = env.action_space.shape[0]
        max_action= float(env.action_space.high[0])

        # device & model load
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = ActorCritic(obs_dim, act_dim, max_action).to(self.device)
        ckpt = torch.load("ppo_cartpole_1.pth", map_location=self.device)
        self.ac.load_state_dict(ckpt)
        self.ac.eval()

    def act(self, observation):
        # inference: use the policy mean (mu) deterministically
        state = torch.tensor(observation, dtype=torch.float32, device=self.device)\
                       .unsqueeze(0)
        with torch.no_grad():
            mu, std, val = self.ac(state)
        action = mu.cpu().numpy().flatten()
        # clip to Box just in case
        return np.clip(action, self.action_space.low, self.action_space.high)