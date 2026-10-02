import os

import torch
import torch.nn as nn
import torch.optim as optim
from sacred import Ingredient

from model import Policy
from storage import RolloutStorage


class SharedA2C:
    """One A2C policy and optimizer trained from every robot's rollout."""

    def __init__(
        self, obs_spaces, action_spaces, lr, adam_eps, recurrent_policy,
        num_steps, num_processes, device,
    ):
        if any(space != obs_spaces[0] for space in obs_spaces[1:]):
            raise ValueError("SharedA2C requires identical observation spaces")
        if any(space != action_spaces[0] for space in action_spaces[1:]):
            raise ValueError("SharedA2C requires identical action spaces")
        self.model = Policy(
            obs_spaces[0], action_spaces[0],
            base_kwargs={"recurrent": recurrent_policy},
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr, eps=adam_eps)
        self.storages = [
            RolloutStorage(
                obs_space, action_space, self.model.recurrent_hidden_state_size,
                num_steps, num_processes,
            )
            for obs_space, action_space in zip(obs_spaces, action_spaces)
        ]
        for storage in self.storages:
            storage.to(device)

    @property
    def models(self):
        return [self.model]

    def initialize(self, observations):
        for storage, observation in zip(self.storages, observations):
            storage.obs[0].copy_(observation)

    def act(self, step):
        outputs = [
            self.model.act(
                storage.obs[step], storage.recurrent_hidden_states[step],
                storage.masks[step],
            )
            for storage in self.storages
        ]
        return tuple(zip(*outputs))

    def insert(self, observations, recurrent_states, actions, log_probs, values,
               rewards, masks, bad_masks):
        for i, storage in enumerate(self.storages):
            storage.insert(
                observations[i], recurrent_states[i], actions[i], log_probs[i],
                values[i], rewards[:, i].unsqueeze(1), masks, bad_masks,
            )

    def compute_returns(self, use_gae, gamma, gae_lambda, use_proper_time_limits):
        for storage in self.storages:
            with torch.no_grad():
                next_value = self.model.get_value(
                    storage.obs[-1], storage.recurrent_hidden_states[-1],
                    storage.masks[-1],
                )
            storage.compute_returns(
                next_value, use_gae, gamma, gae_lambda, use_proper_time_limits,
            )

    def update(self, value_loss_coef, entropy_coef, max_grad_norm, **_):
        policy_losses, value_losses, entropies = [], [], []
        for storage in self.storages:
            obs_shape = storage.obs.size()[2:]
            action_shape = storage.actions.size(-1)
            steps, processes, _ = storage.rewards.size()
            values, log_probs, entropy, _ = self.model.evaluate_actions(
                storage.obs[:-1].view(-1, *obs_shape),
                storage.recurrent_hidden_states[0].view(
                    -1, self.model.recurrent_hidden_state_size
                ),
                storage.masks[:-1].view(-1, 1),
                storage.actions.view(-1, action_shape),
            )
            values = values.view(steps, processes, 1)
            log_probs = log_probs.view(steps, processes, 1)
            advantages = storage.returns[:-1] - values
            policy_losses.append(-(advantages.detach() * log_probs).mean())
            value_losses.append(advantages.pow(2).mean())
            entropies.append(entropy)

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()
        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), max_grad_norm
        )
        self.optimizer.step()
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": (value_loss_coef * value_loss).item(),
            "dist_entropy": (entropy_coef * entropy).item(),
            "gradient_norm": float(gradient_norm),
        }

    def after_update(self):
        for storage in self.storages:
            storage.after_update()

    def save(self, path, run_state):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "run_state": run_state,
            "recurrent_hidden_states": [
                storage.recurrent_hidden_states[0].cpu() for storage in self.storages
            ],
        }, os.path.join(path, "shared_a2c.pt"))

    def restore(self, path, device):
        checkpoint = torch.load(
            os.path.join(path, "shared_a2c.pt"), map_location=device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for storage, state in zip(
            self.storages, checkpoint.get("recurrent_hidden_states", [])
        ):
            storage.recurrent_hidden_states[0].copy_(state.to(device))
        return checkpoint.get("run_state", {})

    def eval_states(self, processes, device):
        return [torch.zeros(processes, self.model.recurrent_hidden_state_size,
                            device=device) for _ in self.storages]

    def eval_act(self, observations, states, masks):
        outputs = [self.model.act(obs, state, masks, deterministic=True)
                   for obs, state in zip(observations, states)]
        _, actions, _, next_states = zip(*outputs)
        return actions, list(next_states)
