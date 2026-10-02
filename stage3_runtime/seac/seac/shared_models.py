"""Separate local actors and local/team critics for the shared PPO baselines."""
import torch
from torch import nn
from torch.distributions import Categorical


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, recurrent=False):
        super().__init__()
        self.recurrent = recurrent
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.gru = nn.GRUCell(hidden_size, hidden_size) if recurrent else None
        self.head = nn.Linear(hidden_size, output_size)

    def initial_state(self, shape, device):
        return torch.zeros(*shape, self.hidden_size, device=device)

    def forward(self, obs, state=None, reset=None):
        features = self.encoder(obs)
        if self.recurrent:
            if state is None:
                state = torch.zeros_like(features)
            if reset is not None:
                state = state * (~reset).unsqueeze(-1)
            shape = features.shape
            state = self.gru(features.reshape(-1, self.hidden_size),
                             state.reshape(-1, self.hidden_size)).reshape(shape)
            features = state
        return self.head(features), state


class Actor(Network):
    def act(self, obs, state=None, reset=None, deterministic=False):
        logits, state = self(obs, state, reset)
        distribution = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), state

    def evaluate_actions(self, obs, actions, state=None, reset=None):
        logits, state = self(obs, state, reset)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), state
