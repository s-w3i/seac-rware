
from collections import deque
from time import perf_counter

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, spaces
from gymnasium.wrappers import TimeLimit


class RecordEpisodeStatistics(gym.Wrapper):
    """ Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.t0 = perf_counter()
        self.episode_reward = np.zeros(self.env.unwrapped.n_agents)
        self.episode_length = 0
        self.metric_totals = {}
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.episode_reward = np.zeros(self.env.unwrapped.n_agents)
        self.episode_length = 0
        self.metric_totals = {}
        self.t0 = perf_counter()

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.episode_reward += np.array(reward, dtype=np.float64)
        self.episode_length += 1
        metric_names = ("completed_cycles", "deliveries", "pickups", "pickup_time",
                        "delivery_time", "return_time", "cycle_time", "path_length",
                        "wait_steps", "conflict_attempts", "movement_denied",
                        "robot_blocked", "deadlock_events", "reward_progress", "reward_step",
                        "reward_conflict", "reward_stall", "reward_event")
        for key in metric_names:
            if key in info:
                self.metric_totals[key] = self.metric_totals.get(key, 0.0) + float(np.sum(info[key]))
        if terminated or truncated:
            info["episode_reward"] = self.episode_reward
            for i, agent_reward in enumerate(self.episode_reward):
                info[f"agent{i}/episode_reward"] = agent_reward
            info["episode_length"] = self.episode_length
            info["episode_time"] = perf_counter() - self.t0

            totals = self.metric_totals
            durations = {"pickup_time": "pickups", "delivery_time": "deliveries",
                         "return_time": "completed_cycles", "cycle_time": "completed_cycles"}
            summary = {key: value for key, value in totals.items() if key not in durations}
            for duration, count in durations.items():
                if duration in totals and totals.get(count, 0) > 0:
                    summary[f"mean_{duration}"] = totals[duration] / totals[count]
            if "completed_cycles" in totals:
                summary["cycles_per_1000_steps"] = 1000 * totals["completed_cycles"] / self.episode_length
            if getattr(self.env.unwrapped, "task_manager_enabled", False):
                warehouse = self.env.unwrapped
                summary["unfinished_cycles"] = sum(
                    task is not None and (
                        warehouse._cycle_steps[i] > 0 if warehouse.routing_features_enabled else True
                    ) for i, task in enumerate(warehouse.task_manager.tasks)
                )
            info["episode_metrics"] = summary

            self.reward_queue.append(self.episode_reward)
            self.length_queue.append(self.episode_length)
        return observation, reward, terminated, truncated, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple([
            spaces.flatten(obs_space, obs)
            for obs_space, obs in zip(self.env.observation_space, observation)
        ])


class SquashDones(gym.Wrapper):
    r"""Wrapper that squashes multiple dones to a single one using all(dones)"""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, bool(terminated), bool(truncated), info


class GlobalizeReward(gym.RewardWrapper):
    def reward(self, reward):
        return self.env.unwrapped.n_agents * [sum(reward)]


class ClearInfo(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(action)
        return observation, reward, terminated, truncated, {}
