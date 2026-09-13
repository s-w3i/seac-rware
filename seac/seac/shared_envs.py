"""Explicit before-reset transitions and read-only privileged routing state."""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import rware  # noqa: F401
from rware.warehouse import TaskPhase

from wrappers import RecordEpisodeStatistics

STATE_SCHEMA = 'routing-team-v1'
AUTOMATIC = {TaskPhase.AUTO_PICKUP, TaskPhase.AUTO_DELIVERY, TaskPhase.AUTO_DROP}


def decision_mask(warehouse):
    return np.array([task is not None and task.phase not in AUTOMATIC
                     for task in warehouse.task_manager.tasks], dtype=bool)


def centralized_state(w):
    """3 row-major maps, then 28 fields/robot in ID order; no mutation/RNG.

    Robot: xy(2), heading(4), load(1), presence(1), phase(6), task xy(6),
    previous action(4), movement(1), blocked(1), distance(1), previous distance(1).
    Coordinates divide by width/height minus one; distances divide by map area.
    """
    height, width = w.grid_size
    maps = np.zeros((3, height, width), np.float32)
    for channel, positions in enumerate((w.goals, w._rack_positions,
                                         [(s.x, s.y) for s in w.shelfs])):
        for x, y in positions:
            maps[channel, y, x] = 1
    scale = np.array([max(width - 1, 1), max(height - 1, 1)], np.float32)
    records = []
    for i, agent in enumerate(w.agents):
        task = w.task_manager.tasks[i]
        records.extend(np.array([agent.x, agent.y]) / scale)
        records.extend(np.eye(4)[agent.dir.value])
        records.extend([float(agent.carrying_shelf is not None), float(task is not None)])
        records.extend(np.eye(6)[task.phase.value] if task else np.zeros(6))
        for coord in ('rack_position', 'workstation_position', 'return_position'):
            records.extend(np.array(getattr(task, coord)) / scale if task else [0, 0])
        records.extend(w._previous_actions[i])
        records.extend([w._movement_success[i], min(w._blocked_streak[i], 10) / 10,
                        w._distances[i] / (height * width),
                        w._previous_distances[i] / (height * width)])
    return np.concatenate([maps.ravel(), np.asarray(records, np.float32)])


def make_shared_env(env_name, time_limit=500):
    env = gym.make(env_name, max_steps=None, max_inactivity_steps=None)
    if time_limit:
        env = gym.wrappers.TimeLimit(env, time_limit)
    return RecordEpisodeStatistics(env)


@dataclass
class Transition:
    next_obs: np.ndarray
    next_state: np.ndarray
    final_obs: np.ndarray  # next state BEFORE any reset, valid for every transition
    final_state: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    decision_mask: np.ndarray
    infos: list


class SharedEnvs:
    def __init__(self, env_name, num_envs, seed, time_limit=500):
        self.envs = [make_shared_env(env_name, time_limit) for _ in range(num_envs)]
        self.obs = np.asarray([env.reset(seed=seed + i)[0]
                               for i, env in enumerate(self.envs)], np.float32)
        spaces = self.envs[0].observation_space
        actions = self.envs[0].action_space
        if len(spaces) != 5 or any(s.shape != (199,) for s in spaces) or any(s.n != 4 for s in actions):
            self.close()
            raise ValueError('Shared baselines require five robots, 199 local features and four actions')
        self.state = self.states()

    def states(self):
        return np.stack([centralized_state(e.unwrapped) for e in self.envs])

    def decisions(self):
        return np.stack([decision_mask(e.unwrapped) for e in self.envs])

    def step(self, actions):
        eligible = self.decisions()
        executed = np.where(eligible, actions, 0)
        obs, final_obs, states, final_states, rewards, terms, truncs, infos = ([] for _ in range(8))
        for env, action in zip(self.envs, executed):
            observation, reward, term, trunc, info = env.step(action)
            final_obs.append(np.asarray(observation, np.float32).copy())
            final_states.append(centralized_state(env.unwrapped))
            if term or trunc:
                observation, _ = env.reset()
            obs.append(observation)
            states.append(centralized_state(env.unwrapped))
            rewards.append(reward)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)
        self.obs, self.state = np.asarray(obs, np.float32), np.asarray(states, np.float32)
        return Transition(self.obs, self.state, np.asarray(final_obs), np.asarray(final_states),
                          np.asarray(rewards, np.float32), np.asarray(terms, bool),
                          np.asarray(truncs, bool), eligible, infos)

    def close(self):
        for env in self.envs:
            env.close()
