import argparse
import os
import sys

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "robotic-warehouse"))
sys.path.insert(0, os.path.join(ROOT, "seac", "seac"))
import rware  # noqa: F401
from a2c import A2C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=1)
    args = p.parse_args()

    env = gym.make(args.env, render_mode="rgb_array")
    agents = [
        A2C(i, obs, act, 3e-4, 0.001, False, 5, 4, "cpu")
        for i, (obs, act) in enumerate(zip(env.observation_space, env.action_space))
    ]
    for agent in agents:
        agent.restore(os.path.join(args.checkpoint, f"agent{agent.agent_id}"))
        agent.model.eval()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    frames = []
    rewards = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        frames.append(env.render())
        while not done:
            with torch.no_grad():
                actions = [
                    agent.model.act(torch.from_numpy(np.asarray(obs[i])), None, None)[1].item()
                    for i, agent in enumerate(agents)
                ]
            obs, reward, terminated, truncated, _ = env.step(actions)
            episode_reward += float(np.sum(reward))
            frames.append(env.render())
            done = bool(terminated or truncated)
        rewards.append(episode_reward)
    env.close()
    # H.264/yuv420p requires even frame dimensions.
    h, w = frames[0].shape[:2]
    if h % 2 or w % 2:
        padded = np.zeros((h + h % 2, w + w % 2, 3), dtype=frames[0].dtype)
        padded[:h, :w] = frames[0]
        frames[0] = padded
        frames = [
            np.pad(frame, ((0, padded.shape[0] - frame.shape[0]),
                           (0, padded.shape[1] - frame.shape[1]), (0, 0)))
            for frame in frames
        ]
    imageio.mimsave(args.output, frames, fps=10, macro_block_size=1)
    print(f"saved={args.output} frames={len(frames)} rewards={rewards}")


if __name__ == "__main__":
    main()
