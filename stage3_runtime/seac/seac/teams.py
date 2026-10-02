import os

import torch

from a2c import A2C


class SEACTeam:
    def __init__(self, obs_spaces, action_spaces, lr, adam_eps,
                 recurrent_policy, num_steps, num_processes, device):
        self.agents = [
            A2C(i, obs, action, lr, adam_eps, recurrent_policy,
                num_steps, num_processes, device)
            for i, (obs, action) in enumerate(zip(obs_spaces, action_spaces))
        ]
        self.storages = [agent.storage for agent in self.agents]
        for storage in self.storages:
            storage.to(device)
        size = self.agents[0].model.recurrent_hidden_state_size
        self.cross_states = [
            [torch.zeros(num_processes, size, device=device)
             for _ in self.agents] for _ in self.agents
        ]
        self.cross_rollout_starts = None

    @property
    def models(self):
        return [agent.model for agent in self.agents]

    def initialize(self, observations):
        for storage, observation in zip(self.storages, observations):
            storage.obs[0].copy_(observation)

    def act(self, step):
        self.cross_rollout_starts = self.cross_rollout_starts or [
            [state.clone() for state in row] for row in self.cross_states
        ]
        outputs = []
        for i, agent in enumerate(self.agents):
            own = None
            for j, storage in enumerate(self.storages):
                if i == j:
                    own = agent.model.act(
                        storage.obs[step], self.cross_states[i][j],
                        storage.masks[step],
                    )
                    self.cross_states[i][j] = own[3]
                elif agent.model.is_recurrent:
                    self.cross_states[i][j] = agent.model.advance_recurrent_state(
                        storage.obs[step], self.cross_states[i][j],
                        storage.masks[step],
                    )
            outputs.append(own)
        return tuple(zip(*outputs))

    def insert(self, observations, recurrent_states, actions, log_probs, values,
               rewards, masks, bad_masks):
        for i, storage in enumerate(self.storages):
            storage.insert(
                observations[i], recurrent_states[i], actions[i], log_probs[i],
                values[i], rewards[:, i].unsqueeze(1), masks, bad_masks,
            )

    def compute_returns(self, **kwargs):
        for agent in self.agents:
            agent.compute_returns(**kwargs)

    def update(self, device, **kwargs):
        losses = []
        for i, agent in enumerate(self.agents):
            starts = self.cross_rollout_starts[i]
            losses.append(agent.update(
                self.storages, starts, device=device, **kwargs
            ))
        return losses

    def after_update(self):
        for storage in self.storages:
            storage.after_update()
        self.cross_rollout_starts = None

    def save(self, path, run_state):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "models": [agent.model.state_dict() for agent in self.agents],
            "optimizers": [agent.optimizer.state_dict() for agent in self.agents],
            "run_state": run_state,
            "cross_states": [[state.cpu() for state in row]
                             for row in self.cross_states],
        }, os.path.join(path, "seac.pt"))

    def restore(self, path, device):
        legacy = os.path.join(path, "agent0", "models.pt")
        if os.path.exists(legacy):
            for agent in self.agents:
                agent.restore(os.path.join(path, f"agent{agent.agent_id}"))
            return {"legacy_checkpoint": True}
        checkpoint = torch.load(
            os.path.join(path, "seac.pt"), map_location=device,
            weights_only=False,
        )
        for agent, state in zip(self.agents, checkpoint["models"]):
            agent.model.load_state_dict(state)
        for agent, state in zip(self.agents, checkpoint["optimizers"]):
            agent.optimizer.load_state_dict(state)
        if "cross_states" in checkpoint:
            self.cross_states = [
                [state.to(device) for state in row]
                for row in checkpoint["cross_states"]
            ]
        return checkpoint.get("run_state", {})

    def eval_states(self, processes, device):
        return [torch.zeros(processes, agent.model.recurrent_hidden_state_size,
                            device=device) for agent in self.agents]

    def eval_act(self, observations, states, masks):
        outputs = [agent.model.act(observations[i], states[i], masks,
                                   deterministic=True)
                   for i, agent in enumerate(self.agents)]
        _, actions, _, next_states = zip(*outputs)
        return actions, list(next_states)
