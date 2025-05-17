import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# ——— Actor definition must match your trained DDPG actor ———
class Actor(nn.Module):
    def __init__(self, obs_dim=3, act_dim=1, act_limit=2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )
        self.act_limit = act_limit

    def forward(self, x):
        # scale tanh output to action range
        return self.act_limit * self.net(x)

# ——— Your Agent stub, now loading & using the trained actor ———
class Agent(object):
    """Agent that uses a pre-trained DDPG actor to choose actions."""
    def __init__(self):
        # Pendulum-v1: Box(-2.0, 2.0, (1,), float32)
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize actor and load weights
        self.actor = Actor(
            obs_dim=3,
            act_dim=1,
            act_limit=self.action_space.high[0]
        ).to(self.device)
        self.actor.load_state_dict(
            torch.load("ddpg_pendulum_1.pth", map_location=self.device)
        )
        self.actor.eval()

    def act(self, observation):
        # observation: np.ndarray of shape (3,)
        obs_t = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        # action is np.ndarray of shape (1,)
        return action