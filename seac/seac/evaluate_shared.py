"""Actor-only evaluation; no critic or training runner is constructed."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from shared_envs import decision_mask, make_shared_env
from shared_models import Actor


def load_actor(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture = saved['architecture']
    actor = Actor(architecture['obs_size'], architecture['actions'], 128,
                  architecture['recurrent']).to(device)
    actor.load_state_dict(saved['actor'])
    actor.eval()
    return actor, saved


@torch.no_grad()
def evaluate_actor(actor, env_name, seeds, steps=500, continuous=False, deterministic=True,
                   layout=None, metadata=None, trace_callback=None):
    if steps < 1:
        raise ValueError('steps must be positive')
    device = next(actor.parameters()).device
    rows = []
    # Stochastic evaluation must not consume the training policy's RNG stream.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        for seed in seeds:
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            env = make_shared_env(env_name, None if continuous else steps,
                                  coordination_trace=trace_callback is not None, layout=layout)
            try:
                obs, _ = env.reset(seed=seed)
                N = len(obs)
                if (N != 5 or any(space.shape != (actor.encoder[0].in_features,)
                                  for space in env.observation_space)
                        or any(space.n != actor.head.out_features for space in env.action_space)):
                    raise ValueError('Checkpoint and environment observation/action dimensions are incompatible')
                state = actor.initial_state((N,), device)
                totals, durations = {}, []
                navigation = 0
                inference = 0.
                individual_rewards = np.zeros(N)
                cycles = np.zeros(N, dtype=np.int64)
                for t in range(steps):
                    mask = decision_mask(env.unwrapped)
                    navigation += int(mask.sum())
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    action, _, state = actor.act(torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device),
                                                  state, deterministic=deterministic)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    inference += time.perf_counter() - start
                    obs, rewards, term, trunc, info = env.step(np.where(mask, action.cpu().numpy(), 0))
                    if trace_callback is not None:
                        trace_callback(t, env.unwrapped, action.cpu().numpy(), mask, info)
                    individual_rewards += rewards
                    completed = np.asarray(info.get('completed_cycles', np.zeros(N)))
                    cycles += completed.astype(np.int64)
                    if 'cycle_time' in info:
                        durations.extend(np.asarray(info['cycle_time'])[completed > 0].tolist())
                    for key, value in info.items():
                        if key.startswith('reward_') or key in ('completed_cycles', 'deliveries', 'pickups',
                                'robot_blocked', 'movement_denied', 'movement_attempts', 'conflict_attempts',
                                'deadlock_events', 'wait_steps', 'pickup_time', 'delivery_time', 'return_time'):
                            totals[key] = totals.get(key, 0.) + float(np.asarray(value).sum())
                    if term or trunc:
                        break
                elapsed = t + 1
                ages = env.unwrapped._cycle_steps.copy()
                rows.append(dict(**(metadata or {}), seed=seed, steps=elapsed,
                                 requested_steps=steps, continuous=continuous, deterministic=deterministic,
                                 **totals, cycles_per_1000_steps=1000 * int(cycles.sum()) / elapsed,
                                 cycles_per_robot=cycles.tolist(), reward_per_robot=individual_rewards.tolist(),
                                 cycle_durations=durations, mean_cycle_time=float(np.mean(durations)) if durations else None,
                                 p95_cycle_time=float(np.percentile(durations, 95)) if durations else None,
                                 unfinished_task_ages=ages.tolist(), max_unfinished_task_age=int(ages.max()),
                                 unfinished_tasks=sum(task is not None for task in env.unwrapped.task_manager.tasks),
                                 zero_cycle_fraction=float(np.mean(cycles == 0)), navigation_decisions=navigation,
                                 denied_per_navigation=totals.get('movement_denied', 0) / max(navigation, 1),
                                 conflicts_per_navigation=totals.get('conflict_attempts', 0) / max(navigation, 1),
                                 team_stall_events=totals.get('deadlock_events', 0),
                                 inference_ms_per_fleet_step=1000 * inference / elapsed))
            finally:
                env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(2000, 2050)))
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--layout-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    steps = args.steps if args.steps is not None else (10000 if args.continuous else 500)
    if steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(1)
    actor, saved = load_actor(args.checkpoint, args.device)
    layout = args.layout_file.read_text() if args.layout_file else None
    metadata = checkpoint_metadata(args.checkpoint, saved, args.device, layout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve before evaluation so an existing report is never overwritten.
    with args.output.open('x') as handle:
        rows = evaluate_actor(actor, saved['config']['env_name'], args.seeds, steps,
                              args.continuous, not args.stochastic, layout, metadata)
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    print(json.dumps({'episodes': len(rows), 'cycles_per_1000_steps': np.mean([r['cycles_per_1000_steps'] for r in rows])}))


def checkpoint_metadata(checkpoint, saved, device, layout=None):
    import inspect
    import gymnasium as gym
    import rware
    from rware.warehouse import Warehouse
    environment = {key: parameter.default for key, parameter in inspect.signature(Warehouse).parameters.items()
                   if parameter.default is not inspect.Parameter.empty}
    environment.update(gym.spec(saved['config']['env_name']).kwargs)
    if layout is not None:
        environment['layout'] = layout
    environment.update(max_steps=None, max_inactivity_steps=None, coordination_trace_enabled=False)
    return dict(checkpoint=str(Path(checkpoint).resolve()),
                checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                training_seed=saved['config']['seed'], checkpoint_env_steps=saved.get('env_steps'),
                env_name=saved['config']['env_name'],
                architecture=saved['architecture'], device=str(device),
                environment=json.loads(json.dumps(environment, default=lambda x: x.name)),
                simulator_sha256=hashlib.sha256(Path(rware.__file__).with_name('warehouse.py').read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
