#!/usr/bin/env python3
"""Matched, fixed-budget Stage 4 adaptation; --campaign profiles before training."""
import argparse
import contextlib
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'seac/seac'), str(ROOT / 'robotic-warehouse'), str(ROOT / 'scripts')]

import numpy as np
import torch

from navigation_envs import NavigationEnvs, stream_seed
from navigation_policy import NavigationActor, NavigationCritic, SPEC, STATE_SCHEMA, export_actor
from navigation_train import (CONDITIONS as NAVIGATION_CONDITIONS, NavigationPPO, configuration, evaluate,
                              load_checkpoint, sha, source_hashes, write_json)
from run_stage3 import MANIFEST, TRANSFER_RESET, converted_actor, fingerprint, load_layout
from train_shared import atomic_save

EXECUTION_SOURCES_SHA256 = fingerprint(source_hashes())
DEFAULT_OUTPUT = PROJECT / 'results/stage4_40x20_7x7'
SCHEDULE = ((1, 98), (10, 98), (20, 196), (30, 196), (40, 683), (50, 683))
CONDITIONS = {method: NAVIGATION_CONDITIONS[method] for method in ('local', 'communicating_ccpd')}
SEEDS = (0, 1)
ANALYSIS = dict(version=2, progress_threshold=500, throughput_gain=.03,
                noninferiority_margin=.02, bootstrap_draws=10000, bootstrap_seed=1729,
                holdout_seeds=list(range(300000, 300050)), final_fleets=[40, 50],
                checkpoint_selection='passing: throughput, -p95 age, -update; otherwise: '
                                     '-failure fraction, throughput, -p95 age, -update',
                uncertainty='paired hierarchical bootstrap: training seed, then scenario; '
                            'action replicates stay inside scenario blocks',
                comparisons=[['communicating_ccpd', 'local']],
                scope='Fixed-map comparison of MAPPO and communication + CCPD, with two training seeds. '
                      'Adapted pretrained pipelines; component effects cannot be separated. '
                      'Two seeds limit uncertainty estimates; the 500-step screen is not proof of deadlock.')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def append(path, row):
    with Path(path).open('a') as file:
        file.write(json.dumps(row, allow_nan=False) + '\n')
        file.flush()


@contextlib.contextmanager
def locked(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def immutable_json(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f'Frozen protocol changed: {path}')
    else:
        write_json(path, value)


def schedule_arg(text):
    try:
        values = tuple(tuple(map(int, item.split(':'))) for item in text.split(','))
        if (any(len(x) != 2 or x[1] < 1 for x in values)
                or [x[0] for x in values] != sorted(set(x[0] for x in values))
                or values[0][0] != 1 or any(x[0] not in dict(SCHEDULE) for x in values)):
            raise ValueError()
        return values
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError('Use increasing fleet:updates pairs starting at fleet 1')


def config_for(condition, seed, agents, updates, device, smoke=False):
    config = configuration(MANIFEST, condition, seed, budget=updates * (64 if smoke else 2048), device=device)
    config.update(n_agents=agents, num_envs=2 if smoke else 8, rollout_steps=32 if smoke else 256,
                  horizons=[16, 32] if smoke else [500, 500, 2000, 2000, 10000, 10000, 10000, 10000],
                  stage4_version=2, max_stage_updates=updates)
    return config


def source_path(condition, seed):
    return PROJECT / 'results/decentralized_navigation/training' / condition / f'seed_{seed}/train/last.pt'


def run_path(output, condition, seed):
    return Path(output) / condition / f'seed_{seed}'


def stage_path(output, condition, seed, agents):
    return run_path(output, condition, seed) / f'n{agents}'


def verify_study(output):
    study = read_json(Path(output) / 'protocol.json')
    current_sources = source_hashes()
    if study['sources'] != current_sources:
        repair_path = Path(output) / 'runtime_repair.json'
        repair = read_json(repair_path) if repair_path.exists() else {}
        if (repair.get('study_sha256') != fingerprint(study)
                or repair.get('sources_before') != study['sources']
                or repair.get('sources_after') != current_sources):
            raise ValueError('Frozen Stage 4 sources changed without a matching runtime repair receipt')
    if (study['manifest_sha256'] != sha(MANIFEST) or study['analysis'] != ANALYSIS):
        raise ValueError('Frozen Stage 4 source, map manifest, or analysis changed')
    for record in study['initial_checkpoints'].values():
        if sha(record['path']) != record['sha256']:
            raise ValueError(f"Source checkpoint changed: {record['path']}")
    return study


def ensure_study(args, layout):
    path = args.output / 'protocol.json'
    previous = verify_study(args.output) if path.exists() else None
    checkpoints = previous['initial_checkpoints'] if previous else {
        f'{method}/seed_{seed}': dict(path=str(source_path(method, seed)),
                                   sha256=sha(source_path(method, seed)))
        for method in CONDITIONS for seed in SEEDS}
    if args.source_checkpoint:
        key = f'{args.condition}/seed_{args.seed}'
        checkpoints = dict(checkpoints)
        checkpoints[key] = dict(path=str(args.source_checkpoint.resolve()), sha256=sha(args.source_checkpoint))
    study = dict(version=2, smoke=args.smoke, schedule=[list(x) for x in args.schedule],
                 methods=list(CONDITIONS), training_seeds=list(SEEDS), map_sha256=layout['sha256'],
                 manifest_sha256=sha(MANIFEST), sources=previous['sources'] if previous else source_hashes(), observation_spec=SPEC,
                 initial_checkpoints=checkpoints, analysis=ANALYSIS,
                 validation=dict(seeds=list(range(100000, 100002 if args.smoke else 100010)),
                                 replicates=1 if args.smoke else 3, steps=64 if args.smoke else 5000,
                                 interval_updates=1 if args.smoke else 49),
                 isolated=dict(seeds=list(range(100100, 100102 if args.smoke else 100200)),
                               replicates=1 if args.smoke else 3, steps=32 if args.smoke else 500,
                               target=.99),
                 joint_steps_per_update=64 if args.smoke else 2048,
                 numerics=dict(dtype='float32', tf32=False, deterministic_algorithms=True, torch_threads=1),
                 early_stopping='technical faults only; learning does not change budgets')
    args.output.mkdir(parents=True, exist_ok=True)
    immutable_json(path, study)
    return study


def conversion(output, study, condition, seed, layout):
    source = study['initial_checkpoints'][f'{condition}/seed_{seed}']
    path = Path(output) / 'converted' / condition / f'seed_{seed}.pt'
    provenance = dict(source_path=source['path'], source_sha256=source['sha256'],
                      study_sha256=fingerprint(study), seed=seed, condition=condition)
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (saved.get('initialization') != provenance
                or read_json(path.with_suffix('.json'))['sha256'] != sha(path)):
            raise ValueError(f'Incompatible/corrupt conversion: {path}')
        return path
    old = torch.load(source['path'], map_location='cpu', weights_only=False)
    if any(old['config'].get(k) != v for k, v in dict(
            seed=seed, condition=condition, ccpd_mode=CONDITIONS[condition][1]).items()):
        raise ValueError('Stage 1 checkpoint has the wrong condition/training seed')
    actor = converted_actor(old, CONDITIONS[condition][0], seed)
    critic = NavigationCritic(layout['shape'])
    saved = dict(format='navigation-training-v2', observation_spec=SPEC, state_schema=STATE_SCHEMA,
                 map_shape=layout['shape'], map_sha256=layout['sha256'],
                 config=config_for(condition, seed, 1, study['schedule'][0][1], 'cpu', study['smoke']),
                 actor=actor.state_dict(), critic=critic.state_dict(), initialization=provenance,
                 pretraining_joint_steps=old['env_steps'], updates=0, env_steps=0,
                 cumulative_env_steps=0, transferred=sorted(set(actor.state_dict()) - TRANSFER_RESET),
                 reinitialized=sorted(TRANSFER_RESET), optimizers='fresh')
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(saved, path)
    write_json(path.with_suffix('.json'), dict(**provenance, sha256=sha(path),
               pretraining_joint_steps=old['env_steps'], reinitialized=saved['reinitialized']))
    if sha(source['path']) != source['sha256']:
        raise ValueError('Source checkpoint changed during conversion')
    return path


def setup_device(device, seed):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and (
            device.index is None or not torch.cuda.is_available() or device.index >= torch.cuda.device_count())):
        raise ValueError(f'Requested device unavailable: {device}; no fallback')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    # Collection and PPO replay use different convolution batch shapes. TF32
    # produced ~8e-4 log-probability drift at 50 robots on the RTX A2000.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    return device


def make_learner(layout, config, initial, restart=0):
    device = setup_device(config['device'], config['seed'])
    env_seed = stream_seed(config['seed'], config['n_agents'], restart, 0xE4)
    envs = NavigationEnvs([layout], env_seed, config['horizons'], config['communication'],
                          config['packet_loss'], config['n_agents'])
    try:
        learner = NavigationPPO(envs.obs.shape[-1], 4, envs.state.shape[-1], config, device,
                                actor=NavigationActor(config['communication']),
                                critic=NavigationCritic(layout['shape']))
        learner.actor.load_state_dict(initial['actor'])
        learner.critic.load_state_dict(initial['critic'])
        learner.actor.seed_streams(stream_seed(config['seed'], config['n_agents'], restart, 0xAA),
                                  config['num_envs'] * config['n_agents'])
        return envs, learner
    except BaseException:
        envs.close()
        raise


def progress_summary(rows):
    return dict(throughput=float(np.mean([r['cycles_per_1000_steps'] for r in rows])),
                failure_episode_fraction=float(np.mean([r['progress_failure'] for r in rows])),
                progress_passed=not any(r['progress_failure'] for r in rows),
                p95_unfinished_age=float(np.percentile([a for r in rows for a in r['unfinished_task_ages']], 95)),
                max_task_age=max(max(r['max_task_ages']) for r in rows), rows=rows)


def validation(actor, layout, agents, study):
    spec = study['validation']
    actor = copy.deepcopy(actor).cpu().eval()
    return progress_summary([row for rep in range(spec['replicates'])
                             for row in evaluate(actor, layout, spec['seeds'], spec['steps'],
                                                 replicate=rep, n_agents=agents)])


def checkpoint_rank(result, update):
    return [int(result['progress_passed']),
            0. if result['progress_passed'] else -result['failure_episode_fraction'],
            result['throughput'], -result['p95_unfinished_age'], -update]


def isolated(actor, layout, study):
    spec = study['isolated']
    actor = copy.deepcopy(actor).cpu().eval()
    rows = [row for rep in range(spec['replicates']) for row in evaluate(
        actor, layout, spec['seeds'], spec['steps'], replicate=rep, n_agents=1, stop_after_first_cycle=True)]
    success = float(np.mean([r.get('completed_cycles', 0) >= 1 for r in rows]))
    return dict(completion_fraction=success, target=spec['target'], passed=success >= spec['target'],
                budget_steps=spec['steps'], independent_scenarios=len(spec['seeds']),
                action_replicates=spec['replicates'], rows=rows, controls_training=False)


def checked_update(learner, envs):
    start = time.perf_counter()
    data, infos, collection = learner.collect(envs)
    del infos
    diagnostics = learner.policy_diagnostics(data)
    error = diagnostics['max_log_prob_error']
    if not math.isfinite(error) or error > 2e-5:
        raise RuntimeError(f'PPO replay mismatch: {error}')
    metrics = learner.update(data)
    if (not all(math.isfinite(v) for v in metrics.values() if isinstance(v, (int, float)))
            or not all(torch.isfinite(p).all() for model in (learner.actor, learner.critic) for p in model.parameters())):
        raise FloatingPointError('Nonfinite training parameter/metric')
    return dict(collection_seconds=collection, pre_update_log_prob_error=error,
                cycle_seconds=time.perf_counter() - start,
                truncated_transitions=int(data['truncated'].sum()), **metrics)


def trim_logs(path, maximum):
    if not path.exists():
        return
    rows = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            break  # An interrupted final append is not a committed update.
        if row['update'] <= maximum:
            rows.append(row)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    temporary.replace(path)


def copy_atomic(source, destination):
    temporary = Path(destination).with_suffix('.tmp')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def train_stage(args, study, layout, agents, updates, preceding):
    stage = stage_path(args.output, args.condition, args.seed, agents)
    stage.mkdir(parents=True, exist_ok=True)
    config = config_for(args.condition, args.seed, agents, updates, args.device, args.smoke)
    input_path = (conversion(args.output, study, args.condition, args.seed, layout) if preceding is None
                  else preceding / 'best.pt')
    previous_steps = 0 if preceding is None else read_json(preceding / 'summary.json')['cumulative_env_steps']
    protocol = dict(study_sha256=fingerprint(study), config=config, input_checkpoint=str(input_path),
                    input_sha256=sha(input_path), previous_joint_steps=previous_steps)
    immutable_json(stage / 'protocol.json', protocol)
    if (stage / 'summary.json').exists():
        result = read_json(stage / 'summary.json')
        if not args.resume:
            raise ValueError('Existing run requires --resume')
        if result['completed_updates'] != updates or any(sha(stage / k) != v for k, v in result['artifacts'].items()):
            raise ValueError('Completed stage artifacts changed')
        return result
    last = stage / 'last.pt'
    if last.exists() and not args.resume:
        raise ValueError('Existing checkpoint requires --resume')
    initial = torch.load(last if last.exists() else input_path, map_location='cpu', weights_only=False)
    if initial.get('observation_spec') != SPEC or initial.get('map_sha256') != layout['sha256']:
        raise ValueError('Incompatible initial checkpoint')
    if last.exists() and initial.get('protocol_sha256') != fingerprint(protocol):
        raise ValueError('Incompatible resume protocol')
    completed = initial['updates'] if last.exists() else 0
    if not 0 <= completed <= updates:
        raise ValueError('Invalid resume update counter')
    restarts = read_rows(stage / 'restarts.jsonl')
    restart = len(restarts) + 1 if last.exists() else 0
    envs, learner = make_learner(layout, config, initial, restart)
    checkpoint_dir = stage / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    best_update, best_rank = (initial['best_update'], initial['best_rank']) if last.exists() else (0, None)
    pretraining = initial['pretraining_joint_steps']
    session_started = time.perf_counter()
    previous_elapsed = initial.get('elapsed_seconds', 0.) if last.exists() else 0.

    def save(update):
        return dict(format='navigation-training-v2', observation_spec=SPEC, state_schema=STATE_SCHEMA,
                    map_shape=layout['shape'], map_sha256=layout['sha256'], config=config,
                    actor=learner.actor.state_dict(), critic=learner.critic.state_dict(),
                    actor_optimizer=learner.actor_optimizer.state_dict(),
                    critic_optimizer=learner.critic_optimizer.state_dict(),
                    updates=update, env_steps=update * study['joint_steps_per_update'],
                    cumulative_env_steps=previous_steps + update * study['joint_steps_per_update'],
                    pretraining_joint_steps=pretraining, initialization=initial['initialization'],
                    elapsed_seconds=previous_elapsed + time.perf_counter() - session_started,
                    protocol_sha256=fingerprint(protocol), best_update=best_update, best_rank=best_rank,
                    runtime_sources_sha256=EXECUTION_SOURCES_SHA256,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(learner.device) if learner.device.type == 'cuda' else None,
                    resume_restarts_environments=True, restart_count=restart)

    try:
        if last.exists():
            learner.actor_optimizer.load_state_dict(initial['actor_optimizer'])
            learner.critic_optimizer.load_state_dict(initial['critic_optimizer'])
            learner.update_number = completed
            random.setstate(initial['python_rng'])
            np.random.set_state(initial['numpy_rng'])
            torch.set_rng_state(initial['torch_rng'])
            if learner.device.type == 'cuda':
                torch.cuda.set_rng_state(initial['cuda_rng'], learner.device)
            for name in ('metrics.jsonl', 'validation.jsonl', 'episodes.jsonl', 'ccpd_events.jsonl'):
                trim_logs(stage / name, completed)
            append(stage / 'restarts.jsonl', dict(update=completed, restart=restart, timestamp=time.time(),
                   runtime_sources_sha256=EXECUTION_SOURCES_SHA256,
                   environments_and_memory_restarted=True,
                   environment_seed=stream_seed(args.seed, agents, restart, 0xE4)))
            copy_atomic(checkpoint_dir / f'{best_update}.pt', stage / 'best.pt')
        else:
            started = time.perf_counter()
            result = validation(learner.actor, layout, agents, study)
            best_rank = checkpoint_rank(result, 0)
            write_json(stage / 'baseline.json', result)
            append(stage / 'validation.jsonl', dict(update=0, validation_seconds=time.perf_counter()-started, **result))
            saved = save(0)
            atomic_save(saved, checkpoint_dir / '0.pt')
            atomic_save(saved, last)
            copy_atomic(checkpoint_dir / '0.pt', stage / 'best.pt')
        for update in range(completed + 1, updates + 1):
            metrics = checked_update(learner, envs)
            append(stage / 'metrics.jsonl', dict(update=update, **metrics))
            for row in envs.completed:
                append(stage / 'episodes.jsonl', dict(update=update, **row))
            envs.completed.clear()
            for row in learner.ccpd_records:
                append(stage / 'ccpd_events.jsonl', dict(update=update, event=row))
            selected = False
            due = update % study['validation']['interval_updates'] == 0 or update == updates
            if due:
                started = time.perf_counter()
                result = validation(learner.actor, layout, agents, study)
                rank = checkpoint_rank(result, update)
                append(stage / 'validation.jsonl', dict(update=update, validation_seconds=time.perf_counter()-started, **result))
                if rank > best_rank:
                    best_update, best_rank, selected = update, rank, True
            saved = save(update)
            if due:
                atomic_save(saved, checkpoint_dir / f'{update}.pt')
            atomic_save(saved, last)
            if selected:
                copy_atomic(checkpoint_dir / f'{update}.pt', stage / 'best.pt')
            print(f'{args.condition} seed={args.seed} n={agents} update={update}/{updates} '
                  f'joint_steps={saved["cumulative_env_steps"]} KL={metrics["approx_kl"]:.5f}', flush=True)
            if args.stop_after == update and update != updates:
                return dict(status='paused', completed_updates=update)
        actor, _ = load_checkpoint(stage / 'best.pt')
        export_actor(actor, stage / 'actor_best.pt')
        export_actor(learner.actor, stage / 'actor_last.pt')
        if agents == 1:
            write_json(stage / 'isolated.json', isolated(actor, layout, study))
        write_json(stage / 'observation_spec.json', SPEC)
        result = dict(status='complete', progress_passed=bool(best_rank[0]), condition=args.condition,
                      seed=args.seed, agents=agents, completed_updates=updates,
                      joint_steps=updates * study['joint_steps_per_update'],
                      cumulative_env_steps=previous_steps + updates * study['joint_steps_per_update'],
                      pretraining_joint_steps=pretraining, best_update=best_update, best_rank=best_rank,
                      protocol_sha256=fingerprint(protocol), smoke=args.smoke,
                      runtime_sources_sha256=EXECUTION_SOURCES_SHA256,
                      runtime=dict(elapsed_seconds=previous_elapsed + time.perf_counter() - session_started,
                                   training_seconds=sum(r['cycle_seconds'] for r in read_rows(stage / 'metrics.jsonl')),
                                   validation_seconds=sum(r['validation_seconds'] for r in read_rows(stage / 'validation.jsonl')),
                                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                   peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(learner.device)
                                       if learner.device.type == 'cuda' else 0),
                      artifacts={name: sha(stage / name) for name in ('best.pt', 'last.pt', 'actor_best.pt', 'actor_last.pt')})
        write_json(stage / 'summary.json', result)
        return result
    finally:
        envs.close()


def train(args, study, layout):
    if not args.smoke and not getattr(args, 'skip_preflight', False):
        for agents in (20, 40, 50):
            path = profile_path(args.output, args.condition, agents, args.device)
            if not path.exists():
                raise ValueError('Production training requires --profile-all or the --campaign preflight')
            record = read_json(path)
            if (record['study_sha256'] != fingerprint(study) or record['smoke']
                    or record['status'] != 'passed'):
                raise ValueError(f'Missing valid GPU preflight: {path}')
    previous = None
    with locked(run_path(args.output, args.condition, args.seed) / 'run.lock'):
        for agents, updates in study['schedule']:
            if args.agents is None or args.agents == agents:
                try:
                    result = train_stage(args, study, layout, agents, updates, previous)
                except Exception as error:
                    write_json(run_path(args.output, args.condition, args.seed) / 'failure.json',
                               dict(status='technical_failure', agents=agents, error=repr(error), timestamp=time.time()))
                    raise
                if result['status'] == 'paused':
                    return result
            previous = stage_path(args.output, args.condition, args.seed, agents)
        failure = run_path(args.output, args.condition, args.seed) / 'failure.json'
        if failure.exists():
            write_json(failure, dict(read_json(failure), status='recovered', recovered_at=time.time()))
    return dict(status='complete')


def host_available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


class MemoryMonitor:
    """Sample host headroom while CUDA's allocator records exact allocation peaks."""
    def __enter__(self):
        self.minimum = host_available()
        self.done = threading.Event()
        def sample():
            while not self.done.wait(.1):
                self.minimum = min(self.minimum, host_available())
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.minimum = min(self.minimum, host_available())
        self.done.set()
        self.thread.join()


def profile_path(output, condition, agents, device):
    return Path(output) / 'profile' / f'{condition}_n{agents}_{device.replace(":", "_")}.json'


def profile(args, study, layout):
    device = setup_device(args.device, args.seed)
    if device.type != 'cuda' and not args.smoke:
        raise ValueError('Production preflight requires an actual CUDA device')
    path = conversion(args.output, study, args.condition, args.seed, layout)
    initial = torch.load(path, map_location='cpu', weights_only=False)
    config = config_for(args.condition, args.seed, args.agents, 2, args.device, args.smoke)
    samples = []
    with MemoryMonitor() as memory:
        envs, learner = make_learner(layout, config, initial)
        try:
            for _ in range(2):
                samples.append(checked_update(learner, envs))
                envs.completed.clear()
            started = time.perf_counter()
            evaluate(copy.deepcopy(learner.actor).cpu().eval(), layout, [100009],
                     32 if args.smoke else 500, n_agents=args.agents)
            eval_seconds = time.perf_counter() - started
            gpu = {}
            if device.type == 'cuda':
                free, total = torch.cuda.mem_get_info(device)
                reserved = torch.cuda.memory_reserved(device)
                peak = torch.cuda.max_memory_reserved(device)
                used_peak = total - free - reserved + peak
                gpu = dict(name=torch.cuda.get_device_name(device), total_bytes=total,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           peak_reserved_bytes=peak, used_peak_bytes=used_peak,
                           headroom_fraction=1 - used_peak / total)
        finally:
            envs.close()
    if sum(row['truncated_transitions'] for row in samples) < 2:
        raise AssertionError('Profile did not exercise truncation')
    passed = args.smoke or (gpu['headroom_fraction'] >= .20 and memory.minimum >= 2 * 1024**3)
    record = dict(status='passed' if passed else 'insufficient_headroom', smoke=args.smoke,
                  condition=args.condition, agents=args.agents, seed=args.seed, device=args.device,
                  study_sha256=fingerprint(study), checkpoint_sha256=sha(path), rows=samples,
                  evaluation_seconds=eval_seconds, evaluation_steps=32 if args.smoke else 500,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  minimum_host_available_bytes=memory.minimum, gpu=gpu)
    destination = profile_path(args.output, args.condition, args.agents, args.device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, record)
    if not passed:
        raise RuntimeError(f'Insufficient profiling headroom; see {destination}')
    return record


def child_command(args, operation, **options):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--' + operation,
               '--output', str(args.output), '--schedule', ','.join(f'{n}:{u}' for n, u in args.schedule)]
    if args.smoke:
        command.append('--smoke')
    for key, value in options.items():
        if value is True:
            command.append('--' + key.replace('_', '-'))
        elif value is not None and value is not False:
            command += ['--' + key.replace('_', '-'), str(value)]
    return command


def profile_all(args, study):
    for device in args.devices:
        for condition in CONDITIONS:
            for agents in (20, 40, 50):
                path = profile_path(args.output, condition, agents, device)
                if path.exists():
                    record = read_json(path)
                    if record['study_sha256'] != fingerprint(study):
                        raise ValueError(f'Stale profile: {path}')
                    if record['status'] == 'passed':
                        continue
                subprocess.run(child_command(args, 'profile', condition=condition, seed=0,
                                              agents=agents, device=device), check=True)
    estimate_cost(args.output, study, args.devices)


def estimate_cost(output, study, devices):
    runs = []
    for method in CONDITIONS:
        for device in devices:
            profiles = {n: read_json(profile_path(output, method, n, device)) for n in (20, 40, 50)}
            seconds = {n: float(np.mean([r['cycle_seconds'] for r in p['rows']])) for n, p in profiles.items()}
            estimates = {1: seconds[20], 10: seconds[20], 20: seconds[20],
                         30: (seconds[20] + seconds[40]) / 2, 40: seconds[40], 50: seconds[50]}
            training = sum(updates * estimates[n] for n, updates in study['schedule'])
            validation_seconds = 0.
            for n, updates in study['schedule']:
                p = profiles[20 if n <= 20 else 40 if n <= 40 else 50]
                count = 1 + math.ceil(updates / study['validation']['interval_updates'])
                validation_seconds += count * len(study['validation']['seeds']) * study['validation']['replicates'] * study['validation']['steps'] * p['evaluation_seconds'] / p['evaluation_steps']
            from stage4_evaluation import evaluation_suites
            final = sum(sum(s['count'] * s['replicates'] * s['steps'] for s in evaluation_suites(method, study))
                        * profiles[n]['evaluation_seconds'] / profiles[n]['evaluation_steps'] for n in (40, 50))
            iso = study['isolated']
            isolated_seconds = len(iso['seeds']) * iso['replicates'] * iso['steps'] * profiles[20]['evaluation_seconds'] / profiles[20]['evaluation_steps']
            runs.append(dict(condition=method, device=device, training_hours=training/3600,
                             validation_hours=validation_seconds/3600, isolated_hours=isolated_seconds/3600,
                             final_evaluation_hours=final/3600))
    total = sum(sum(r[k] for k in ('training_hours', 'validation_hours', 'isolated_hours', 'final_evaluation_hours'))
                for r in runs) * len(SEEDS) / len(devices)
    write_json(Path(output) / 'cost_estimate.json', dict(runs=runs, serial_campaign_hours=total,
               smoke=study['smoke'], assumptions='Two-cycle measurements, not a guarantee. '
               '1/10 use conservative 20-robot costs; 30 interpolates 20/40. Includes validation, '
               'isolated screens and final suites. File I/O, CPU contention and slower learned behavior can add cost.'))


def joint_profile(args, study):
    if len(args.devices) != 2 or any(not d.startswith('cuda:') for d in args.devices):
        raise ValueError('Joint check requires two distinct CUDA devices')
    destination = args.output / 'joint_profile'
    assignments = list(zip(CONDITIONS, args.devices))
    commands = [child_command(args, 'profile', condition=m, seed=0, agents=50, device=d)
                for m, d in assignments]
    # Run in a separate output so the isolated measurements stay intact.
    destination.mkdir(exist_ok=True)
    immutable_json(destination / 'protocol.json', study)
    if (args.output / 'runtime_repair.json').exists():
        immutable_json(destination / 'runtime_repair.json', read_json(args.output / 'runtime_repair.json'))
    for method, _ in assignments:
        conversion(destination, study, method, 0, load_layout())
    commands = [[str(destination) if x == str(args.output) else x for x in cmd] for cmd in commands]
    with MemoryMonitor() as memory:
        processes = [subprocess.Popen(cmd) for cmd in commands]
        codes = [p.wait() for p in processes]
    if any(codes):
        raise RuntimeError('Joint memory/runtime profiling failed')
    rows = [read_json(profile_path(destination, m, 50, d)) for m, d in assignments]
    ratios = []
    for (method, device), row in zip(assignments, rows):
        single = read_json(profile_path(args.output, method, 50, device))
        ratios.append(np.mean([r['cycle_seconds'] for r in row['rows']]) / np.mean([r['cycle_seconds'] for r in single['rows']]))
    passed = (memory.minimum >= 2 * 1024**3 and all(r['gpu']['headroom_fraction'] >= .2 for r in rows)
              and max(ratios) <= 1.25)
    write_json(args.output / 'concurrency.json', dict(passed=bool(passed), study_sha256=fingerprint(study),
               devices=args.devices, assignments=dict(assignments), minimum_host_available_bytes=memory.minimum, slowdown=ratios,
               jobs=2, maximum_allowed_slowdown=1.25))
    if not passed:
        raise RuntimeError('Joint check failed; use --jobs 1')


def campaign(args, study):
    with locked(args.output / 'pipeline.lock'):
        skip_preflight = getattr(args, 'skip_preflight', False)
        if skip_preflight:
            append(args.output / 'preflight_overrides.jsonl', dict(timestamp=time.time(),
                   runtime_sources_sha256=EXECUTION_SOURCES_SHA256, devices=args.devices, jobs=args.jobs))
            print('Startup profiling skipped by explicit request; training integrity checks remain enabled.', flush=True)
        else:
            profile_all(args, study)
            if args.jobs == 2:
                joint_profile(args, study)
        jobs = [(m, s) for s in SEEDS for m in CONDITIONS]
        logs = args.output / 'logs'
        logs.mkdir(exist_ok=True)
        for offset in range(0, len(jobs), args.jobs):
            running = []
            with contextlib.ExitStack() as stack:
                for slot, (method, seed) in enumerate(jobs[offset:offset + args.jobs]):
                    device = args.devices[(offset + slot) % len(args.devices)]
                    log = stack.enter_context((logs / f'{method}_seed_{seed}.log').open('a'))
                    command = child_command(args, 'train', condition=method, seed=seed, device=device,
                                            resume=True, skip_preflight=skip_preflight)
                    print('Launching', ' '.join(command), flush=True)
                    running.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
                codes = [p.wait() for p in running]
            if any(codes):
                raise RuntimeError('Training process failed; inspect logs and failure.json, then resume the campaign')
        from stage4_evaluation import lock_evaluation, evaluate_campaign, report
        with locked(args.output / 'evaluation.lock'):
            lock_evaluation(args.output, study)
            evaluate_campaign(args.output, study)
            return report(args.output, study)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for operation in ('prepare', 'train', 'profile', 'profile-all', 'profile-pair', 'campaign',
                      'lock-evaluation', 'evaluate', 'report'):
        group.add_argument('--' + operation, action='store_true')
    parser.add_argument('--condition', choices=tuple(CONDITIONS))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--source-checkpoint', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--agents', type=int, choices=tuple(dict(SCHEDULE)))
    parser.add_argument('--schedule', type=schedule_arg)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-preflight', action='store_true',
                        help='Explicitly skip startup memory/timing profiles; retain training integrity checks')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.schedule = args.schedule or tuple((n, 2 if args.smoke else u) for n, u in SCHEDULE)
    if not args.smoke and args.schedule != SCHEDULE:
        parser.error('Production schedule is frozen; custom schedules require --smoke and a separate output')
    if args.smoke and args.output == DEFAULT_OUTPUT:
        parser.error('Smoke runs require a separate --output')
    if (args.train or args.profile or args.source_checkpoint) and (args.condition is None or args.seed is None):
        parser.error('Provide --condition and --seed')
    if args.profile and args.agents is None:
        parser.error('--profile requires --agents')
    if args.train and args.agents is not None and args.agents not in dict(args.schedule):
        parser.error('--agents is not in the schedule')
    if args.stop_after is not None and (not args.train or args.stop_after < 1):
        parser.error('--stop-after requires --train and a positive update')
    if args.skip_preflight and not (args.train or args.campaign):
        parser.error('--skip-preflight applies only to --train or --campaign')
    if len(set(args.devices)) != len(args.devices) or (args.campaign and args.jobs == 2 and len(args.devices) != 2):
        parser.error('Use distinct devices; --jobs 2 requires exactly two')
    layout = load_layout()
    if args.report or args.evaluate or args.lock_evaluation:
        study = verify_study(args.output)
        from stage4_evaluation import lock_evaluation, evaluate_campaign, report
        function = report if args.report else lock_evaluation if args.lock_evaluation else evaluate_campaign
        with locked(args.output / 'evaluation.lock'):
            result = function(args.output, study)
    else:
        study = ensure_study(args, layout)
        if args.prepare:
            result = [str(conversion(args.output, study, m, s, layout)) for m in CONDITIONS for s in SEEDS]
        elif args.train:
            result = train(args, study, layout)
        elif args.profile:
            result = profile(args, study, layout)
        elif args.profile_all:
            result = profile_all(args, study)
        elif args.profile_pair:
            result = joint_profile(args, study)
        else:
            try:
                write_json(args.output / 'status.json', dict(stage='running', pid=os.getpid(), updated_at=time.time()))
                result = campaign(args, study)
                write_json(args.output / 'status.json', dict(stage='complete', updated_at=time.time()))
            except Exception as error:
                write_json(args.output / 'status.json', dict(stage='technical_failure', error=repr(error), updated_at=time.time()))
                raise
    print(json.dumps(result, default=str, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
