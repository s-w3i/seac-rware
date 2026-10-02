import glob
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from os import path
from pathlib import Path

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import (  # noqa
    FileStorageObserver,
    MongoObserver,
    QueuedMongoObserver,
    QueueObserver,
)
from torch.utils.tensorboard import SummaryWriter

import utils
from a2c import algorithm
from envs import make_vec_envs
from shared_a2c import SharedA2C
from teams import SEACTeam
from wrappers import RecordEpisodeStatistics, SquashDones
from model import Policy

import rware  # noqa: F401 - importing registers the warehouse environments

try:
    import lbforaging  # noqa: F401
except ImportError:
    # Level-Based Foraging is not needed for RWARE experiments.
    lbforaging = None

ex = Experiment(ingredients=[algorithm])
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
ex.observers.append(FileStorageObserver("./results/sacred"))

logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)


@ex.config
def config():
    env_name = None
    time_limit = None
    wrappers = (
        RecordEpisodeStatistics,
        SquashDones,
    )
    dummy_vecenv = False

    num_env_steps = 100e6

    eval_dir = "./results/video/{id}"
    loss_dir = "./results/loss/{id}"
    save_dir = "./results/trained_models/{id}"

    log_interval = 2000
    save_interval = int(1e6)
    eval_interval = int(1e6)
    episodes_per_eval = 8
    monitor_eval_seed = 10000
    final_eval_seed = 20000
    trainer = "seac"
    deterministic = True


for conf in glob.glob("configs/*.yaml"):
    name = f"{Path(conf).stem}"
    ex.add_named_config(name, conf)


def _resolve_device(configured_device):
    configured_device = str(configured_device)
    if configured_device == "auto":
        configured_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if configured_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{configured_device}' was requested, but CUDA is unavailable"
        )
    return torch.device(configured_device)


def _seed_everything(seed, deterministic):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _provenance(seed, trainer, algorithm, resolved_config):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        packages = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        packages = []
    return {
        "seed": seed,
        "trainer": trainer,
        "algorithm": dict(algorithm),
        "resolved_config": resolved_config,
        "git_commit": commit,
        "command": sys.argv,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available() and str(algorithm["device"]).startswith("cuda")
        else None,
        "physical_gpu_id": os.environ.get("PHYSICAL_GPU_ID"),
        "packages": packages,
    }


def _append_metrics(save_dir, record):
    with open(path.join(save_dir, "metrics.jsonl"), "a") as handle:
        handle.write(json.dumps(record, default=float) + "\n")

def _squash_info(info):
    episodes = [i for i in info if "episode_reward" in i]
    rows = [{**{key: value for key, value in item.items()
                if key in ("episode_reward", "episode_length", "episode_time")
                or key.endswith("/episode_reward")},
             **item.get("episode_metrics", {})} for item in episodes]
    result = {key: np.mean([np.asarray(row[key]).sum() for row in rows if key in row])
              for key in set().union(*(row.keys() for row in rows))}
    for mean, count in (("mean_pickup_time", "pickups"), ("mean_delivery_time", "deliveries"),
                        ("mean_return_time", "completed_cycles"), ("mean_cycle_time", "completed_cycles")):
        completed = [row for row in rows if mean in row and row.get(count, 0) > 0]
        if completed:
            result[mean] = sum(row[mean] * row[count] for row in completed) / sum(row[count] for row in completed)
    if rows and "cycles_per_1000_steps" in result:
        result["cycles_per_1000_steps"] = 1000 * sum(row.get("completed_cycles", 0) for row in rows) / sum(row["episode_length"] for row in rows)
    if rows:
        episode_steps = sum(row["episode_length"] for row in rows)
        agent_count = len([key for key in rows[0] if key.endswith("/episode_reward")])
        agent_steps = episode_steps * agent_count
        if episode_steps:
            result["deliveries_per_1000_steps"] = 1000 * sum(
                row.get("deliveries", 0) for row in rows
            ) / episode_steps
            result["deadlock_event_rate"] = 1000 * sum(
                row.get("deadlock_events", 0) for row in rows
            ) / episode_steps
        movement_attempts = sum(row.get("movement_attempts", 0) for row in rows)
        if movement_attempts:
            result["movement_denied_rate"] = sum(
                row.get("movement_denied", 0) for row in rows
            ) / movement_attempts
            result["conflict_attempt_rate"] = sum(
                row.get("conflict_attempts", 0) for row in rows
            ) / movement_attempts
        if agent_steps:
            result["wait_step_ratio"] = sum(
                row.get("wait_steps", 0) for row in rows
            ) / agent_steps
    return result



@ex.capture
def evaluate(
    team,
    monitor_dir,
    episodes_per_eval,
    env_name,
    eval_seed,
    wrappers,
    dummy_vecenv,
    time_limit,
    algorithm,
    _log,
    _run,
):
    device = _resolve_device(algorithm["device"])

    eval_envs = make_vec_envs(
        env_name,
        eval_seed,
        dummy_vecenv,
        episodes_per_eval,
        time_limit,
        wrappers,
        device,
        monitor_dir=monitor_dir,
    )

    n_obs = eval_envs.reset()
    n_recurrent_hidden_states = team.eval_states(episodes_per_eval, device)
    masks = torch.zeros(episodes_per_eval, 1, device=device)

    all_infos = []
    inference_seconds = 0.0
    inference_calls = 0

    while len(all_infos) < episodes_per_eval:
        with torch.no_grad():
            inference_start = time.perf_counter()
            n_action, n_recurrent_hidden_states = team.eval_act(
                n_obs, n_recurrent_hidden_states, masks
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_start
            inference_calls += 1

        # Obser reward and next obs
        n_obs, _, done, infos = eval_envs.step(n_action)

        n_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )
        masks = n_masks
        all_infos.extend([i for i in infos if i and "episode_reward" in i])

    eval_envs.close()
    info = _squash_info(all_infos)
    info["inference_latency_ms"] = 1000 * inference_seconds / inference_calls
    for key, value in info.items():
        _run.log_scalar(f"evaluation/{key}", value)
    _log.info(
        f"Evaluation using {len(all_infos)} episodes: mean reward {info['episode_reward']:.5f}\n"
    )
    return info


@ex.automain
def main(
    _run,
    _log,
    num_env_steps,
    env_name,
    seed,
    algorithm,
    dummy_vecenv,
    time_limit,
    wrappers,
    save_dir,
    eval_dir,
    loss_dir,
    log_interval,
    save_interval,
    eval_interval,
    trainer,
    deterministic,
    monitor_eval_seed,
    final_eval_seed,
    _config,
):

    if loss_dir:
        loss_dir = path.expanduser(loss_dir.format(id=str(_run._id)))
        utils.cleanup_log_dir(loss_dir)
        writer = SummaryWriter(loss_dir)
    else:
        writer = None

    eval_dir = path.expanduser(eval_dir.format(id=str(_run._id)))
    save_dir = path.expanduser(save_dir.format(id=str(_run._id)))

    if path.exists(path.join(save_dir, "metrics.jsonl")) or path.exists(
        path.join(save_dir, "provenance.json")
    ):
        raise FileExistsError(
            f"Refusing to mix attempts in existing result directory: {save_dir}"
        )

    utils.cleanup_log_dir(eval_dir)
    utils.cleanup_log_dir(save_dir)

    torch.set_num_threads(1)
    _seed_everything(seed, deterministic)
    device = _resolve_device(algorithm["device"])
    envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        algorithm["num_processes"],
        time_limit,
        wrappers,
        device,
    )

    if trainer == "seac":
        team = SEACTeam(
            envs.observation_space, envs.action_space,
            algorithm["lr"], algorithm["adam_eps"],
            algorithm["recurrent_policy"], algorithm["num_steps"],
            algorithm["num_processes"], device,
        )
    elif trainer == "shared_a2c":
        team = SharedA2C(
            envs.observation_space, envs.action_space,
            algorithm["lr"], algorithm["adam_eps"],
            algorithm["recurrent_policy"], algorithm["num_steps"],
            algorithm["num_processes"], device,
        )
    else:
        raise ValueError(f"Unknown trainer: {trainer}")
    obs = envs.reset()
    team.initialize(obs)
    provenance = _provenance(seed, trainer, algorithm, _config)
    os.makedirs(save_dir, exist_ok=True)
    with open(path.join(save_dir, "provenance.json"), "w") as handle:
        json.dump(provenance, handle, indent=2, default=str)
    parameter_counts = [sum(parameter.numel() for parameter in model.parameters())
                        for model in team.models]
    _append_metrics(save_dir, {
        "kind": "compute", "update": 0,
        "parameters_per_policy": parameter_counts,
        "total_deployed_parameters": sum(parameter_counts),
        "policy_networks": len(team.models),
    })

    start = time.time()
    num_updates = (
        int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
    )

    all_infos = deque(maxlen=10)

    for j in range(1, num_updates + 1):
        update_start = time.perf_counter()

        for step in range(algorithm["num_steps"]):
            # Sample actions
            with torch.no_grad():
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = team.act(step)
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(n_action)
            # envs.envs[0].render()

            # If done then clean the history of observations.
            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device,
            )

            bad_masks = torch.tensor(
                [
                    [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                    for info in infos
                ],
                dtype=torch.float32,
                device=device,
            )
            team.insert(
                obs, n_recurrent_hidden_states, n_action, n_action_log_prob,
                n_value, reward, masks, bad_masks,
            )

            for info in infos:
                if "episode_reward" in info:
                    all_infos.append(info)

        # value_loss, action_loss, dist_entropy = agent.update(rollouts)
        team.compute_returns(
            use_gae=algorithm["use_gae"], gamma=algorithm["gamma"],
            gae_lambda=algorithm["gae_lambda"],
            use_proper_time_limits=algorithm["use_proper_time_limits"],
        )

        losses = team.update(
            device=device, value_loss_coef=algorithm["value_loss_coef"],
            entropy_coef=algorithm["entropy_coef"],
            seac_coef=algorithm["seac_coef"],
            max_grad_norm=algorithm["max_grad_norm"],
        )
        if isinstance(losses, dict):
            losses = [losses]
        for agent_id, loss in enumerate(losses):
            for k, v in loss.items():
                if writer:
                    writer.add_scalar(f"agent{agent_id}/{k}", v, j)

        mean_losses = {
            key: float(np.mean([loss[key] for loss in losses if key in loss]))
            for key in set().union(*(loss.keys() for loss in losses))
        }
        mean_losses.update({
            "kind": "learning", "update": j,
            "update_time_seconds": time.perf_counter() - update_start,
            "gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else 0,
        })
        mean_losses["training_fps"] = (
            algorithm["num_processes"] * algorithm["num_steps"]
            / mean_losses["update_time_seconds"]
        )
        _append_metrics(save_dir, mean_losses)

        team.after_update()

        if j % log_interval == 0 and len(all_infos) > 0:
            squashed = _squash_info(all_infos)

            total_num_steps = (
                (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
            )
            end = time.time()
            _log.info(
                f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
            )
            if "episode_reward" in squashed:
                _log.info(
                    f"Last {len(all_infos)} training episodes mean reward "
                    f"{squashed['episode_reward'].sum():.3f}"
                )

            for k, v in squashed.items():
                _run.log_scalar(k, v, j)
            all_infos.clear()

        if save_interval is not None and (
            j > 0 and j % save_interval == 0 or j == num_updates
        ):
            cur_save_dir = path.join(save_dir, f"u{j}")
            team.save(cur_save_dir, {
                "update": j,
                "provenance": provenance,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_states": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else [],
            })
            archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
            shutil.rmtree(cur_save_dir)
            _run.add_artifact(archive_name)

        if eval_interval is not None and j < num_updates and j % eval_interval == 0:
            evaluation_metrics = evaluate(
                team, os.path.join(eval_dir, f"monitor-u{j}"),
                eval_seed=monitor_eval_seed,
            )
            _append_metrics(save_dir, {
                "kind": "evaluation", "suite": "monitor", "update": j,
                **evaluation_metrics,
            })
            videos = glob.glob(os.path.join(eval_dir, f"monitor-u{j}") + "/*.mp4")
            for i, v in enumerate(videos):
                _run.add_artifact(v, f"u{j}.{i}.mp4")
        if j == num_updates:
            evaluation_metrics = evaluate(
                team, os.path.join(eval_dir, "final"), eval_seed=final_eval_seed,
            )
            _append_metrics(save_dir, {
                "kind": "evaluation", "suite": "final", "update": j,
                **evaluation_metrics,
            })
    envs.close()
