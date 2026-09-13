"""One clipped PPO learner for feedforward/recurrent local and team critics."""
import time

import numpy as np
import torch
from torch.distributions import Categorical

from shared_models import Actor, Network
from shared_storage import batches, gae, replay, retain_history


class SharedPPO:
    def __init__(self, obs_size, actions, state_size, config, device):
        self.config, self.device = config, device
        self.central = config['method'] == 'mappo'
        self.actor = Actor(obs_size, actions, 128, config['recurrent']).to(device)
        self.critic = Network(state_size if self.central else obs_size, 1,
                              256 if self.central else 128,
                              config['recurrent'] and not self.central).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        self.actor_state = self.critic_state = self.reset = None
        self.actor_history = self.critic_history = None

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device)

    @torch.no_grad()
    def collect(self, envs):
        start = time.perf_counter()
        E, N = envs.obs.shape[:2]
        if self.reset is None:
            self.reset = torch.ones(E, N, dtype=torch.bool, device=self.device)
            self.actor_state = self.actor.initial_state((E, N), self.device)
            self.critic_state = self.critic.initial_state((E,) if self.central else (E, N), self.device)
        rows, infos = [], []
        for _ in range(self.config['rollout_steps']):
            obs, state = self.tensor(envs.obs), self.tensor(envs.state)
            critic_input = state if self.central else obs
            critic_reset = self.reset[:, 0] if self.central else self.reset
            actor_before, critic_before = self.actor_state, self.critic_state
            actions, log_probs, actor_after = self.actor.act(obs, actor_before, self.reset)
            values, critic_after = self.critic(critic_input, critic_before, critic_reset)
            transition = envs.step(actions.cpu().numpy())
            final_input = self.tensor(transition.final_state if self.central else transition.final_obs)
            bootstrap, _ = self.critic(final_input, critic_after)
            rows.append(dict(obs=obs, critic_input=critic_input, actions=actions,
                             log_probs=log_probs, values=values.squeeze(-1),
                             bootstrap=bootstrap.squeeze(-1), rewards=self.tensor(transition.rewards),
                             terminated=self.tensor(transition.terminated), truncated=self.tensor(transition.truncated),
                             eligible=self.tensor(transition.decision_mask), resets=self.reset,
                             actor_states=actor_before, critic_states=critic_before))
            self.actor_state = actor_after if self.actor.recurrent else actor_before
            self.critic_state = critic_after if self.critic.recurrent else critic_before
            done = self.tensor(transition.terminated | transition.truncated)
            self.reset = done[:, None].expand(E, N)
            infos.extend(transition.infos)
        data = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
        advantages, returns = gae(data['rewards'].mean(-1), data['values'], data['bootstrap'],
                                  data['terminated'], data['truncated'], self.config['gamma'],
                                  self.config['gae_lambda'])
        data.update(advantages=advantages, returns=returns)
        return data, infos, time.perf_counter() - start

    def streams(self, data, actor):
        if actor or not self.central:
            inputs = data['obs'].flatten(1, 2)
            states = data['actor_states' if actor else 'critic_states'].flatten(1, 2)
            resets = data['resets'].flatten(1, 2)
        else:
            inputs, states, resets = data['critic_input'], data['critic_states'], data['resets'][:, :, 0]
        return inputs, states, resets

    def minibatches(self, data, actor, shuffle=True):
        model = self.actor if actor else self.critic
        return batches(*self.streams(data, actor), model.recurrent,
                       self.config['num_minibatches'], self.config['sequence_length'],
                       self.config['burn_in'], self.actor_history if actor else self.critic_history,
                       shuffle=shuffle)

    @torch.no_grad()
    def policy_diagnostics(self, data):
        old = data['log_probs'].flatten()
        actions, eligible = data['actions'].flatten(), data['eligible'].flatten()
        logs, entropies = torch.empty_like(old), torch.empty_like(old)
        for indices, batch in self.minibatches(data, True, shuffle=False):
            valid = indices >= 0
            idx = indices[valid]
            dist = Categorical(logits=replay(self.actor, batch)[valid])
            logs[idx], entropies[idx] = dist.log_prob(actions[idx]), dist.entropy()
        difference = logs[eligible] - old[eligible]
        ratio = difference.exp()
        return dict(approx_kl=float(((ratio - 1) - difference).mean()) if difference.numel() else 0.,
                    clip_fraction=float(((ratio - 1).abs() > self.config['clip_epsilon']).float().mean()) if difference.numel() else 0.,
                    entropy=float(entropies[eligible].mean()) if difference.numel() else 0.,
                    max_log_prob_error=float((logs - old).abs().max()))

    def update(self, data):
        start = time.perf_counter()
        advantages = data['advantages']
        if self.central:
            advantages = advantages.unsqueeze(-1).expand_as(data['log_probs'])
        advantages = advantages.flatten().clone()
        eligible = data['eligible'].flatten()
        if eligible.any():
            valid_adv = advantages[eligible]
            advantages = (advantages - valid_adv.mean()) / (valid_adv.std(unbiased=False) + 1e-8)
        actions, old_logs = data['actions'].flatten(), data['log_probs'].flatten()
        actor_losses, actor_norms, critic_losses, critic_norms = [], [], [], []
        epochs = 0
        for _ in range(self.config['ppo_epochs']):
            if not eligible.any():
                break
            for indices, batch in self.minibatches(data, True):
                valid = indices >= 0
                keep = valid & eligible[indices.clamp_min(0)]
                if not keep.any():
                    continue
                idx = indices[keep]
                dist = Categorical(logits=replay(self.actor, batch)[keep])
                ratio = (dist.log_prob(actions[idx]) - old_logs[idx]).exp()
                unclipped = ratio * advantages[idx]
                clipped = ratio.clamp(1 - self.config['clip_epsilon'], 1 + self.config['clip_epsilon']) * advantages[idx]
                loss = -torch.minimum(unclipped, clipped).mean() - self.config['entropy_coef'] * dist.entropy().mean()
                self.actor_optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['max_grad_norm'])
                self.actor_optimizer.step()
                actor_losses.append(float(loss.detach()))
                actor_norms.append(float(norm))
            epochs += 1
            if self.policy_diagnostics(data)['approx_kl'] > self.config['target_kl']:
                break
        targets = data['returns'].flatten()
        for _ in range(self.config['ppo_epochs']):
            for indices, batch in self.minibatches(data, False):
                valid = indices >= 0
                predicted = replay(self.critic, batch)[valid, 0]
                loss = .5 * (predicted - targets[indices[valid]]).square().mean()
                self.critic_optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['max_grad_norm'])
                self.critic_optimizer.step()
                critic_losses.append(float(loss.detach()))
                critic_norms.append(float(norm))
        diagnostics = self.policy_diagnostics(data)
        for actor in (True, False):
            model = self.actor if actor else self.critic
            if model.recurrent:
                old = self.actor_history if actor else self.critic_history
                history = retain_history(*self.streams(data, actor), old, self.config['burn_in'])
                if actor:
                    self.actor_history = history
                else:
                    self.critic_history = history
        variance = targets.var(unbiased=False)
        diagnostics.update(actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.,
                           critic_loss=float(np.mean(critic_losses)), actor_grad_norm=max(actor_norms, default=0.),
                           critic_grad_norm=max(critic_norms, default=0.), actor_epochs=epochs,
                           critic_epochs=self.config['ppo_epochs'], valid_decision_fraction=float(eligible.float().mean()),
                           explained_variance=float(1 - (targets - data['values'].flatten()).var(unbiased=False) / variance) if variance > 1e-8 else 0.,
                           update_seconds=time.perf_counter() - start)
        return diagnostics
