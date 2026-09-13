"""Independent Sacred entry point for the four shared PPO baselines."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time
import uuid

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.utils.tensorboard import SummaryWriter

from evaluate_shared import evaluate_actor
from shared_envs import SharedEnvs, STATE_SCHEMA
from shared_ppo import SharedPPO

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = dict(env_name='rware-custom-5ag-routing-v2', method='shared_ppo', recurrent=False,
                seed=0, num_env_steps=2000000, time_limit=500, num_envs=8, rollout_steps=256,
                device='auto', gamma=.99, gae_lambda=.95, actor_lr=.0003, critic_lr=.0003,
                ppo_epochs=4, num_minibatches=4, clip_epsilon=.2, entropy_coef=.01,
                max_grad_norm=.5, target_kl=.02, sequence_length=32, burn_in=16,
                eval_interval_env_steps=250000, save_interval_env_steps=250000,
                eval_seeds=list(range(1000, 1004)), eval_steps=500, run_dir=None, resume=None,
                deterministic=True)
RESUME_OVERRIDES = {'run_dir', 'resume', 'device', 'num_env_steps',
                    'save_interval_env_steps', 'eval_interval_env_steps'}
ex = Experiment('shared_baselines')
ex.add_config(DEFAULTS)
for name in ('shared_ppo_routing', 'shared_ppo_gru_routing', 'mappo_routing', 'mappo_gru_routing'):
    ex.add_named_config(name, str(Path(__file__).parent / 'configs' / (name + '.yaml')))


def validate(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    if config['method'] not in ('shared_ppo', 'mappo'):
        raise ValueError('method must be shared_ppo or mappo')
    for key in ('num_env_steps', 'time_limit', 'num_envs', 'rollout_steps', 'ppo_epochs',
                'num_minibatches', 'sequence_length', 'eval_steps', 'eval_interval_env_steps',
                'save_interval_env_steps'):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not isinstance(config['burn_in'], int) or config['burn_in'] < 0:
        raise ValueError('burn_in must be a nonnegative integer')
    for key in ('actor_lr', 'critic_lr', 'max_grad_norm', 'target_kl'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('gamma', 'gae_lambda', 'clip_epsilon'):
        if not 0 <= config[key] <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    if not np.isfinite(config['entropy_coef']) or config['entropy_coef'] < 0:
        raise ValueError('entropy_coef must be finite and nonnegative')
    if not config['eval_seeds'] or any(not isinstance(s, int) or s < 0 for s in config['eval_seeds']):
        raise ValueError('eval_seeds must contain nonnegative integers')
    if not isinstance(config['seed'], int) or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not isinstance(config['recurrent'], bool) or not isinstance(config['deterministic'], bool):
        raise ValueError('recurrent and deterministic must be booleans')


@ex.config_hook
def prepare(config, command_name, logger):
    validate(config)
    output = Path(config['run_dir'] or ROOT / 'results' / 'shared_baselines' /
                  f"{config['method']}{'_gru' if config['recurrent'] else ''}-{uuid.uuid4().hex[:12]}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    ex.observers[:] = [FileStorageObserver(str(output / 'sacred'))]
    return dict(run_dir=str(output))


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def checkpoint(learner, config, architecture, steps, updates, best):
    return dict(format_version=1, architecture=architecture, state_schema=STATE_SCHEMA,
                config=dict(config), actor=learner.actor.state_dict(), critic=learner.critic.state_dict(),
                actor_optimizer=learner.actor_optimizer.state_dict(), critic_optimizer=learner.critic_optimizer.state_dict(),
                env_steps=steps, updates=updates, best_score=best,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if learner.device.type == 'cuda' else None)


def restore(learner, path, config, architecture):
    saved = torch.load(path, map_location=learner.device, weights_only=False)
    # Only operational settings may change when continuing an existing optimizer.
    mismatch = [k for k in DEFAULTS if k not in RESUME_OVERRIDES and saved['config'][k] != config[k]]
    if saved['architecture'] != architecture or saved['state_schema'] != STATE_SCHEMA or mismatch:
        raise ValueError(f'Incompatible checkpoint/configuration: {mismatch}')
    learner.actor.load_state_dict(saved['actor'])
    learner.critic.load_state_dict(saved['critic'])
    learner.actor_optimizer.load_state_dict(saved['actor_optimizer'])
    learner.critic_optimizer.load_state_dict(saved['critic_optimizer'])
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'].cpu())
    if learner.device.type == 'cuda' and saved['cuda_rng']:
        torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    return saved['env_steps'], saved['updates'], saved['best_score']


def append_json(path, value):
    with path.open('a') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')


def git_output(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


@ex.main
def train(_config, _run):
    config = dict(_config)
    output = Path(config['run_dir'])
    torch.set_num_threads(1)
    if config['deterministic']:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if config['device'] == 'auto' else torch.device(config['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
    envs = SharedEnvs(config['env_name'], config['num_envs'], config['seed'], config['time_limit'])
    writer = None
    try:
        architecture = dict(obs_size=envs.obs.shape[-1], actions=int(envs.envs[0].action_space[0].n),
                            state_size=envs.state.shape[-1], recurrent=config['recurrent'])
        learner = SharedPPO(architecture['obs_size'], architecture['actions'], architecture['state_size'], config, device)
        steps, updates, best = 0, 0, -1.
        inherited_best = False
        if config['resume']:
            steps, updates, best = restore(learner, config['resume'], config, architecture)
            # Fresh simulator/memory stream, reproducible from the saved progress.
            envs.close()
            envs = SharedEnvs(config['env_name'], config['num_envs'], config['seed'] + steps, config['time_limit'])
            previous_best = Path(config['resume']).resolve().parent / 'best.pt'
            if previous_best.is_file():
                candidate = torch.load(previous_best, map_location='cpu', weights_only=False)
                inherited_best = (candidate['architecture'] == architecture and
                                  candidate['state_schema'] == STATE_SCHEMA and
                                  all(candidate['config'][k] == config[k] for k in DEFAULTS if k not in RESUME_OVERRIDES) and
                                  candidate['best_score'] == best and candidate['env_steps'] <= steps)
                if inherited_best:
                    atomic_save(candidate, output / 'best.pt')
            if not inherited_best:
                best = -1.  # A standalone last checkpoint cannot reconstruct historical best weights.

        if steps >= config['num_env_steps']:
            raise ValueError('num_env_steps must exceed checkpoint progress')
        warehouse = envs.envs[0].unwrapped
        layout = dict(grid_size=list(map(int, warehouse.grid_size)),
                      goals=sorted([int(x), int(y)] for x, y in warehouse.goals),
                      rack_homes=sorted([int(x), int(y)] for x, y in warehouse._rack_positions))
        provenance = dict(config=config, architecture=architecture, state_schema=STATE_SCHEMA,
                          git_commit=git_output('rev-parse', 'HEAD').strip(), git_status=git_output('status', '--short'),
                          map_sha256=hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest(),
                          actual_layout=layout,
                          asset_map_sha256=hashlib.sha256((ROOT / 'assets/warehouse-10-6.map').read_bytes()).hexdigest(),
                          packages={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'gymnasium', 'sacred', 'PyYAML', 'tensorboard')},
                          cuda=torch.version.cuda, device=str(device),
                          gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                          physical_gpu=os.environ.get('PHYSICAL_GPU_ID'), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                          actor_parameters=sum(p.numel() for p in learner.actor.parameters()),
                          critic_parameters=sum(p.numel() for p in learner.critic.parameters()),
                          resume_restarts_environments=bool(config['resume']), inherited_best=inherited_best)
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        (output / 'source.diff').write_text(git_output('diff', 'HEAD'))
        # Include new source files in dirty-run provenance, not just tracked diffs.
        source_dir = output / 'source'
        source_dir.mkdir()
        for source in Path(__file__).parent.glob('*shared*.py'):
            (source_dir / source.name).write_bytes(source.read_bytes())
        launcher_source = ROOT / 'scripts/run_shared_baselines.py'
        (source_dir / launcher_source.name).write_bytes(launcher_source.read_bytes())
        writer = SummaryWriter(str(output / 'tensorboard'))
        next_eval = (steps // config['eval_interval_env_steps'] + 1) * config['eval_interval_env_steps']
        next_save = (steps // config['save_interval_env_steps'] + 1) * config['save_interval_env_steps']
        initial_actor = [p.detach().clone() for p in learner.actor.parameters()]
        initial_critic = [p.detach().clone() for p in learner.critic.parameters()]
        started = time.perf_counter()
        while steps < config['num_env_steps']:
            data, infos, collection_seconds = learner.collect(envs)
            metrics = learner.update(data)
            steps += config['num_envs'] * config['rollout_steps']
            updates += 1
            metrics.update(kind='learning', env_steps=steps, agent_steps=steps * envs.obs.shape[1],
                           updates=updates, collection_seconds=collection_seconds,
                           peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                           individual_reward_sum=data['rewards'].sum((0, 1)).cpu().tolist(),
                           wall_seconds=time.perf_counter() - started)
            metrics['environment_steps_per_second'] = config['num_envs'] * config['rollout_steps'] / (collection_seconds + metrics['update_seconds'])
            for key in infos[0]:
                if key.startswith('reward_') or key in ('completed_cycles', 'deliveries', 'pickups', 'robot_blocked', 'movement_denied', 'movement_attempts', 'conflict_attempts', 'deadlock_events', 'wait_steps'):
                    metrics[key] = sum(float(np.asarray(i.get(key, 0)).sum()) for i in infos)
            append_json(output / 'metrics.jsonl', metrics)
            for key in ('actor_loss', 'critic_loss', 'approx_kl', 'entropy'):
                _run.log_scalar(key, metrics[key], steps)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, steps)
            print(f"{config['method']}{'_gru' if config['recurrent'] else ''} steps={steps} actor={metrics['actor_loss']:.4f} critic={metrics['critic_loss']:.4f} KL={metrics['approx_kl']:.5f}", flush=True)
            final = steps >= config['num_env_steps']
            if steps >= next_eval or final:
                rows = evaluate_actor(learner.actor, config['env_name'], config['eval_seeds'], config['eval_steps'])
                for row in rows:
                    append_json(output / 'evaluation.jsonl', dict(env_steps=steps, **row))
                score = float(np.mean([r['cycles_per_1000_steps'] for r in rows]))
                if score > best:
                    best = score
                    atomic_save(checkpoint(learner, config, architecture, steps, updates, best), output / 'best.pt')
                next_eval = (steps // config['eval_interval_env_steps'] + 1) * config['eval_interval_env_steps']
            if steps >= next_save or final:
                atomic_save(checkpoint(learner, config, architecture, steps, updates, best), output / 'last.pt')
                atomic_save(dict(actor=learner.actor.state_dict(), architecture=architecture, config=config,
                                 format_version=1, observation_schema='routing-local-199-v1'), output / 'actor.pt')
                next_save = (steps // config['save_interval_env_steps'] + 1) * config['save_interval_env_steps']
        summary = dict(env_steps=steps, agent_steps=steps * envs.obs.shape[1], updates=updates,
                       budget_overshoot=steps - config['num_env_steps'], best_score=best,
                       actor_changed=any(not torch.equal(a, b) for a, b in zip(initial_actor, learner.actor.parameters())),
                       critic_changed=any(not torch.equal(a, b) for a, b in zip(initial_critic, learner.critic.parameters())),
                       wall_seconds=time.perf_counter() - started,
                       peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        if writer is not None:
            writer.close()
        envs.close()


if __name__ == '__main__':
    ex.run_commandline()
