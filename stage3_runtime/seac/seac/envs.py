import gymnasium as gym
import numpy as np
import torch

from wrappers import TimeLimit


class MultiAgentVecEnv:
    """Small synchronous vector runner for RWARE's multi-agent API."""

    def __init__(self, env_fns, device):
        self.envs = [fn() for fn in env_fns]
        self.device = device
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        observations = [env.reset()[0] for env in self.envs]
        return self._observations_to_tensors(observations)

    def step(self, actions):
        actions = np.stack(
            [action.squeeze(-1).detach().cpu().numpy() for action in actions], axis=1
        )
        observations, rewards, dones, infos = [], [], [], []
        for env, joint_action in zip(self.envs, actions):
            observation, reward, terminated, truncated, info = env.step(joint_action)
            done = bool(terminated or truncated)
            if done:
                info["TimeLimit.truncated"] = bool(truncated and not terminated)
                observation, _ = env.reset()
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (
            self._observations_to_tensors(observations),
            torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def _observations_to_tensors(self, observations):
        return [
            torch.as_tensor(np.stack(agent_obs), dtype=torch.float32, device=self.device)
            for agent_obs in zip(*observations)
        ]

    def close(self):
        for env in self.envs:
            env.close()


def make_env(env_id, seed, rank, time_limit, wrappers, monitor_dir):
    def _thunk():
        env = gym.make(env_id)
        if time_limit:
            env = TimeLimit(env, time_limit)
        for wrapper in wrappers:
            env = wrapper(env)
        env.reset(seed=seed + rank)
        return env

    return _thunk


def make_vec_envs(
    env_name, seed, dummy_vecenv, parallel, time_limit, wrappers, device,
    monitor_dir=None,
):
    env_fns = [
        make_env(env_name, seed, rank, time_limit, wrappers, monitor_dir)
        for rank in range(parallel)
    ]
    return MultiAgentVecEnv(env_fns, device)
