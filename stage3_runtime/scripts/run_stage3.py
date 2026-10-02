#!/usr/bin/env python3
"""Stage 3 fixed-map 7x7 conversion, bounded pilots, checks, and profiling."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'seac/seac'), str(ROOT / 'robotic-warehouse')]

import numpy as np
import torch

from navigation_envs import NavigationEnvs, NavigationEpisode, load_layouts, stream_seed
from navigation_policy import (GRID_WIDTH, LOCAL_SIZE, SPEC, STATE_SCHEMA, NavigationActor,
                               NavigationCritic, export_actor)
from navigation_train import (CONDITIONS, NavigationPPO, configuration, evaluate,
                              load_checkpoint, sha, source_hashes, write_json)
from train_shared import atomic_save

MANIFEST = ROOT / 'assets/stage3_40x20.json'
DEFAULT_OUTPUT = PROJECT / 'results/stage3_40x20_7x7'
OLD_RESULTS = PROJECT / 'results/decentralized_navigation/training'
TRANSFER_RESET = {'grid.5.weight', 'grid.5.bias'}
FLEETS = (1, 10, 20, 30, 40, 50)
PILOT_UPDATES = 244


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def source_checkpoint(condition):
    return OLD_RESULTS / condition / 'seed_0/train/last.pt'


def conversion_path(output, condition):
    # Keep conversions made by earlier runner versions available for their CPU profiles.
    return output / 'converted' / f'{condition}-{sha(Path(__file__))[:12]}.pt'


def converted_actor(old, communication, seed=0):
    """Reuse compatible v1 actor weights; resize only the raster projection."""
    if (old.get('format') != 'navigation-training-v1'
            or old.get('observation_spec', {}).get('schema') != 'ego-navigation-v1'
            or old['observation_spec'].get('grid_shape') != [7, 5, 5]
            or old.get('state_schema') != 'pooled-warehouse-v1'
            or old.get('config', {}).get('communication') != communication):
        raise ValueError('Source is not the expected 5x5 navigation checkpoint')
    torch.manual_seed(seed)
    actor = NavigationActor(communication)
    target = actor.state_dict()
    if set(target) != set(old['actor']):
        raise ValueError('Unexpected actor parameter names during conversion')
    for name, tensor in target.items():
        source_tensor = old['actor'][name]
        if name in TRANSFER_RESET:
            if name == 'grid.5.weight' and tensor.shape == source_tensor.shape:
                raise ValueError(f'Expected resized projection: {name}')
            continue
        if tensor.shape != source_tensor.shape:
            raise ValueError(f'Unexpected parameter shape change: {name}')
        target[name] = source_tensor.detach().clone()
    actor.load_state_dict(target)
    return actor


def load_layout():
    layouts = load_layouts(MANIFEST, n_agents=50)
    if len(layouts) != 1 or layouts[0]['shape'] != (21, 41):
        raise ValueError('Stage 3 requires the declared single 40x20 map')
    return layouts[0]


def config_for(condition, agents, smoke=False, updates=PILOT_UPDATES, device='cpu'):
    if condition not in CONDITIONS or agents not in FLEETS:
        raise ValueError('Unsupported condition or fleet size')
    config = configuration(MANIFEST, condition, 0, budget=updates * (64 if smoke else 2048), device=device)
    config.update(n_agents=agents, num_envs=2 if smoke else 8,
                  rollout_steps=32 if smoke else 256,
                  horizons=[16, 32] if smoke else [500, 500, 2000, 2000, 10000, 10000, 10000, 10000],
                  observation_spec=SPEC, state_schema=STATE_SCHEMA,
                  stage3_version=1, max_stage_updates=updates)
    return config


def conversion(condition, output, layout):
    source = source_checkpoint(condition)
    destination = conversion_path(output, condition)
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = sha(source)
    sources = source_hashes()
    if destination.exists():
        saved = torch.load(destination, map_location='cpu', weights_only=False)
        if (saved.get('conversion', {}).get('source_sha256') != source_hash
                or saved.get('conversion', {}).get('stage3_sources') != sources
                or saved.get('observation_spec') != SPEC):
            raise ValueError(f'Existing conversion is incompatible: {destination}')
        return destination
    old = torch.load(source, map_location='cpu', weights_only=False)
    actor = converted_actor(old, CONDITIONS[condition][0])
    target = actor.state_dict()
    critic = NavigationCritic(layout['shape'])
    config = config_for(condition, 1)
    record = dict(source_path=str(source.resolve()), source_sha256=source_hash,
                  stage3_sources=sources, map_sha256=layout['sha256'],
                  transferred=sorted(set(target) - TRANSFER_RESET),
                  reinitialized=sorted(TRANSFER_RESET), critic='fresh', optimizers='fresh',
                  old_env_steps=old['env_steps'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(format='navigation-training-v2', observation_spec=SPEC,
                     state_schema=STATE_SCHEMA, config=config, map_shape=layout['shape'],
                     actor=actor.state_dict(), critic=critic.state_dict(), env_steps=0,
                     updates=0, conversion=record), destination)
    write_json(destination.with_suffix('.json'), record)
    return destination


def validation(actor, layout, agents, smoke=False):
    actor.eval()
    seeds = range(100000, 100002 if smoke else 100010)
    replicates = range(1 if smoke else 3)
    horizon = 64 if smoke else 5000
    rows = [row for replicate in replicates
            for row in evaluate(actor, layout, seeds, horizon, replicate=replicate, n_agents=agents)]
    throughput = float(np.mean([r['cycles_per_1000_steps'] for r in rows]))
    ages = [age for row in rows for age in row['unfinished_task_ages']]
    complete = [cycle > 0 for row in rows for cycle in row['cycles_per_robot']]
    return dict(agents=agents, scenarios=len(rows), horizon=horizon, throughput=throughput,
                p95_unfinished_age=float(np.percentile(ages, 95)),
                robot_episode_completion_fraction=float(np.mean(complete)), rows=rows)


def passes(current, baseline):
    throughput = current['throughput']
    previous = baseline['throughput']
    improved = throughput > 0 if previous == 0 else throughput >= previous * 1.05
    return bool(improved and current['p95_unfinished_age'] <= baseline['p95_unfinished_age']
                and current['robot_episode_completion_fraction'] >= .90)


def checkpoint(learner, config, layout, conversion_record, updates, best_rank, passed_twice):
    return dict(format='navigation-training-v2', observation_spec=SPEC, state_schema=STATE_SCHEMA,
                config=config, map_shape=layout['shape'], actor=learner.actor.state_dict(),
                critic=learner.critic.state_dict(),
                actor_optimizer=learner.actor_optimizer.state_dict(),
                critic_optimizer=learner.critic_optimizer.state_dict(),
                updates=updates, env_steps=updates * config['num_envs'] * config['rollout_steps'],
                best_rank=best_rank, passed_twice=passed_twice,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), conversion=conversion_record,
                cuda_rng=torch.cuda.get_rng_state(learner.device) if learner.device.type == 'cuda' else None,
                stage3_sources=source_hashes(), map_sha256=layout['sha256'],
                manifest_sha256=sha(MANIFEST), resume_restarts_environments=True)


def stage_path(output, condition, agents):
    return output / condition / f'n{agents}'


def clean_log(path, maximum):
    if path.exists():
        lines = [line for line in path.read_text().splitlines()
                 if json.loads(line)['update'] <= maximum]
        path.write_text(''.join(line + '\n' for line in lines))


def train_stage(args, condition, agents, layout):
    stage = stage_path(args.output, condition, agents)
    stage.mkdir(parents=True, exist_ok=True)
    updates = 2 if args.smoke else args.updates
    config = config_for(condition, agents, args.smoke, updates, args.device)
    protocol = dict(config=config, manifest_sha256=sha(MANIFEST), layout_sha256=layout['sha256'],
                    sources=source_hashes(), evaluation_seeds=[100000, 100001] if args.smoke else list(range(100000, 100010)),
                    evaluation_replicates=1 if args.smoke else 3, evaluation_steps=64 if args.smoke else 5000)
    protocol_file = stage / 'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError(f'Stage protocol changed: {stage}')
    if not protocol_file.exists():
        write_json(protocol_file, protocol)
    completed = stage / 'summary.json'
    if completed.exists():
        return json.loads(completed.read_text())
    if agents == 1:
        input_path = conversion(condition, args.output, layout)
    else:
        predecessor = stage_path(args.output, condition, FLEETS[FLEETS.index(agents)-1])
        summary = json.loads((predecessor / 'summary.json').read_text())
        if summary['status'] != 'passed' and not args.smoke:
            raise ValueError('Previous fleet has not passed its learning screen')
        input_path = predecessor / 'best.pt'
    initial = torch.load(input_path, map_location='cpu', weights_only=False)
    if (initial.get('format') != 'navigation-training-v2' or initial.get('observation_spec') != SPEC
            or initial.get('state_schema') != STATE_SCHEMA or initial.get('map_sha256', layout['sha256']) != layout['sha256']):
        raise ValueError('Incompatible Stage 3 initialization')
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or device.index is None
                                  or device.index >= torch.cuda.device_count()):
        raise ValueError(f'CUDA device unavailable: {args.device}')
    if device.type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    torch.use_deterministic_algorithms(True)
    last = stage / 'last.pt'
    restart_update = (torch.load(last, map_location='cpu', weights_only=False)['updates']
                      if last.exists() else 0)
    envs = NavigationEnvs([layout], 100 + agents + 1000 * restart_update,
                          config['horizons'], config['communication'],
                          config['packet_loss'], agents)
    learner = NavigationPPO(envs.obs.shape[-1], 4, envs.state.shape[-1], config, device,
                            actor=NavigationActor(config['communication']),
                            critic=NavigationCritic(envs.map_shape))
    learner.actor.load_state_dict(initial['actor'])
    learner.critic.load_state_dict(initial['critic'])
    learner.actor.seed_streams(stream_seed(0, agents, 0xAA), len(envs.envs) * agents)
    metrics_path = stage / 'metrics.jsonl'
    validation_path = stage / 'validation.jsonl'
    episodes_path = stage / 'episodes.jsonl'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if (saved.get('format') != 'navigation-training-v2' or saved.get('config') != config
                or saved.get('stage3_sources') != protocol['sources']
                or saved.get('manifest_sha256') != protocol['manifest_sha256']):
            raise ValueError('Incompatible restart checkpoint')
        learner.actor.load_state_dict(saved['actor']); learner.critic.load_state_dict(saved['critic'])
        learner.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        learner.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'], device)
        completed_updates = saved['updates']
        learner.update_number = completed_updates
        best_rank, consecutive = saved['best_rank'], saved['passed_twice']
        learner.actor.seed_streams(stream_seed(0, agents, completed_updates, 0xAA), len(envs.envs) * agents)
        for path in (metrics_path, validation_path, episodes_path):
            clean_log(path, completed_updates)
    else:
        completed_updates, best_rank, consecutive = 0, None, 0
        base = validation(copy.deepcopy(learner.actor).cpu(), layout, agents, args.smoke)
        write_json(stage / 'baseline.json', base)
        atomic_save(checkpoint(learner, config, layout, initial['conversion'], 0, None, 0), last)
    baseline = json.loads((stage / 'baseline.json').read_text())
    update = completed_updates
    try:
        for update in range(completed_updates + 1, updates + 1):
            data, _, collection_seconds = learner.collect(envs)
            diagnostics = learner.policy_diagnostics(data)
            if diagnostics['max_log_prob_error'] > 2e-5:
                raise RuntimeError('PPO replay mismatch')
            metrics = learner.update(data)
            if not all(math.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))):
                raise FloatingPointError('Nonfinite update metric')
            if not all(torch.isfinite(p).all() for model in (learner.actor, learner.critic)
                       for p in model.parameters()):
                raise FloatingPointError('Nonfinite trained parameter')
            with metrics_path.open('a') as file:
                file.write(json.dumps(dict(update=update, collection_seconds=collection_seconds,
                                           pre_update_log_prob_error=diagnostics['max_log_prob_error'], **metrics)) + '\n')
            with episodes_path.open('a') as file:
                for row in envs.completed:
                    file.write(json.dumps(dict(update=update, **row)) + '\n')
            envs.completed.clear()
            if update % (1 if args.smoke else 49) == 0 or update == updates:
                result = validation(copy.deepcopy(learner.actor).cpu(), layout, agents, args.smoke)
                good = passes(result, baseline)
                consecutive = consecutive + 1 if good else 0
                rank = [result['throughput'], -result['p95_unfinished_age'], -update]
                with validation_path.open('a') as file:
                    file.write(json.dumps(dict(update=update, passed=good, **result)) + '\n')
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    is_best = True
                else:
                    is_best = False
            else:
                is_best = False
            saved = checkpoint(learner, config, layout, initial['conversion'], update, best_rank, consecutive)
            atomic_save(saved, last)
            if is_best:
                atomic_save(saved, stage / 'best.pt')
            print(f'{condition} n={agents} update={update}/{updates} KL={metrics["approx_kl"]:.5f}', flush=True)
            if args.stop_after == update:
                return dict(status='paused', condition=condition, agents=agents,
                            completed_updates=update, checkpoint=str(last))
            if consecutive >= 2 and not args.smoke:
                break
        summary = dict(status='passed' if consecutive >= 2 or args.smoke else 'needs_work',
                       condition=condition, agents=agents, completed_updates=update,
                       joint_steps=update * config['num_envs'] * config['rollout_steps'],
                       best_rank=best_rank, baseline_throughput=baseline['throughput'],
                       source_checkpoint_sha256=initial['conversion']['source_sha256'],
                       protocol_sha256=sha(protocol_file), schema=SPEC['schema'],
                       smoke=args.smoke)
        write_json(completed, summary)
        export_actor(learner.actor, stage / 'actor_last.pt')
        write_json(stage / 'observation_spec.json', SPEC)
        return summary
    finally:
        envs.close()


def check(layout):
    if GRID_WIDTH != 7 or LOCAL_SIZE != 361:
        raise AssertionError('Incorrect Stage 3 sensor contract')
    for agents in FLEETS:
        episode = NavigationEpisode(layout, 700 + agents, 10700 + agents, 16,
                                    layout['shape'], agents, True)
        try:
            assert episode.obs.shape == (agents, 361 + 21 * (agents - 1))
            assert episode.state.shape == (4 * 21 * 41 + agents * 368,)
            actor = NavigationActor(True)
            actor.seed_streams(0, agents)
            action, _, state = actor.act(torch.as_tensor(episode.obs))
            assert action.shape == (agents,) and torch.isfinite(state).all()
        finally:
            episode.close()
    return dict(status='passed', map_sha256=layout['sha256'], observation_spec=SPEC,
                fleets=list(FLEETS), source_hashes=source_hashes())


def profile(args, layout):
    if args.condition is None or args.agents is None:
        raise ValueError('Profiling requires --condition and --agents')
    condition, agents = args.condition, args.agents
    path = conversion(condition, args.output, layout)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = config_for(condition, agents, updates=2)
    torch.set_num_threads(1)
    envs = NavigationEnvs([layout], 900 + agents, config['horizons'], config['communication'],
                          config['packet_loss'], agents)
    learner = NavigationPPO(envs.obs.shape[-1], 4, envs.state.shape[-1], config, torch.device('cpu'),
                            actor=NavigationActor(config['communication']), critic=NavigationCritic(envs.map_shape))
    learner.actor.load_state_dict(saved['actor']); learner.critic.load_state_dict(saved['critic'])
    learner.actor.seed_streams(0, len(envs.envs) * agents)
    rows = []
    try:
        for _ in range(2):
            data, _, collection_seconds = learner.collect(envs)
            pre = learner.policy_diagnostics(data)
            assert pre['max_log_prob_error'] <= 2e-5
            started = time.perf_counter()
            metrics = learner.update(data)
            rows.append(dict(collection_seconds=collection_seconds,
                             update_seconds=time.perf_counter() - started,
                             peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                             truncated_transitions=int(data['truncated'].sum()),
                             selected_ccpd=metrics.get('ccpd_selected_samples', 0)))
    finally:
        envs.close()
    assert sum(r['truncated_transitions'] for r in rows) >= 2
    result = dict(status='passed', condition=condition, agents=agents, rows=rows,
                  map_sha256=layout['sha256'], schema=SPEC['schema'], sources=source_hashes(),
                  checkpoint_sha256=sha(path), cuda_available=torch.cuda.is_available())
    destination = args.output / 'profile' / f'{condition}_{agents}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    return result


def report(output, layout):
    summaries = [json.loads(path.read_text()) for path in sorted(output.glob('*/n*/summary.json'))]
    profiles = [json.loads(path.read_text()) for path in sorted((output / 'profile').glob('*.json'))]
    lines = [
        '# Stage 3: fixed 40×20 map with 7×7 observations', '',
        f'Map SHA-256: `{layout["sha256"]}`. Schema: `{SPEC["schema"]}`.', '',
        'The original 5×5 checkpoints are preserved. Converted checkpoints reuse compatible actor weights, '
        'initialize the resized grid projection and critic, and start with fresh optimizers.', '',
        '## Pilot status', '',
        '| Method | Robots | Status | Joint steps | Initial throughput |',
        '|---|---:|---|---:|---:|',
    ]
    for row in summaries:
        lines.append(f'| {row["condition"]} | {row["agents"]} | {row["status"]} '
                     f'| {row["joint_steps"]} | {row["baseline_throughput"]:.3f} |')
    if not summaries:
        lines.append('| Pending | — | Training has not started | — | — |')
    lines += ['', 'Each pilot is capped at 244 updates per fleet size. Stage 3 uses validation seeds '
              '100000–100009 and leaves 200000–200049 for later final evaluation. '
              'The 99% isolated-route target is reported separately and does not control pilot promotion.', '',
              '## Full rollout/update profiles', '',
              '| Method | Robots | Mean collection s | Mean update s | Peak RSS GiB |',
              '|---|---:|---:|---:|---:|']
    for row in profiles:
        samples = row['rows']
        lines.append(f'| {row["condition"]} | {row["agents"]} '
                     f'| {np.mean([r["collection_seconds"] for r in samples]):.2f} '
                     f'| {np.mean([r["update_seconds"] for r in samples]):.2f} '
                     f'| {max(r["peak_rss_mib"] for r in samples)/1024:.2f} |')
    if not profiles:
        lines.append('| Pending | — | — | — | — |')
    lines += ['', 'These are two-cycle CPU engineering measurements. Later fleet sizes require '
              'learning calibration before a Stage 4 budget is fixed.', '',
              '## Evidence', '',
              '- `converted/`: explicit 5×5-to-7×7 conversion records and checkpoint hashes.',
              '- `<method>/n<robots>/`: protocol, baseline, updates, validation, checkpoints, and summary.',
              '- `profile/`: complete rollout/update timings and peak process memory.',
              '- Source snapshot: `stage3_runtime/assets/stage1_source_snapshot.json`.', '']
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'STAGE3_REPORT.md'
    temporary = output / f'.STAGE3_REPORT.{os.getpid()}.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(report_path)
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    for flag in ('check', 'convert', 'train', 'train-both-gpus', 'evaluate', 'profile', 'report'):
        operation.add_argument('--' + flag, action='store_true')
    parser.add_argument('--condition', choices=tuple(CONDITIONS))
    parser.add_argument('--agents', type=int, choices=FLEETS)
    parser.add_argument('--curriculum', action='store_true', help='Run the bounded 1-then-10 pilot')
    parser.add_argument('--updates', type=int, default=PILOT_UPDATES)
    parser.add_argument('--device', default='cpu', help='Training device, e.g. cpu, cuda:0, or cuda:1')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--smoke', action='store_true', help='Two small updates with short evaluations')
    parser.add_argument('--stop-after', type=int, help='Stop after saving this update; next run resumes')
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if not 1 <= args.updates <= PILOT_UPDATES:
        parser.error(f'--updates must be between 1 and {PILOT_UPDATES}')
    if args.stop_after is not None and (not args.train or args.stop_after < 1
                                       or args.stop_after > (2 if args.smoke else args.updates)):
        parser.error('--stop-after must name an update in this training run')
    layout = load_layout()
    if args.train and args.condition is None:
        parser.error('--train requires --condition')
    if args.train and args.agents not in (None, 1, 10) and not args.smoke:
        parser.error('Larger fleets are Stage 3 engineering checks only; use --profile or --smoke')
    if args.curriculum and (not args.train or args.agents is not None):
        parser.error('--curriculum requires --train without --agents')
    if args.train_both_gpus:
        if args.condition is not None or args.agents is not None or args.curriculum or args.device != 'cpu':
            parser.error('--train-both-gpus takes only --updates, --output, and --smoke')
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            parser.error('--train-both-gpus requires two visible CUDA GPUs')
        jobs = []
        for condition, device in (('local', 'cuda:0'), ('communicating_ccpd', 'cuda:1')):
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--train', '--curriculum',
                       '--condition', condition, '--device', device, '--updates', str(args.updates),
                       '--output', str(args.output)]
            if args.smoke:
                command.append('--smoke')
            print(f'Starting {condition} on {device}', flush=True)
            jobs.append(subprocess.Popen(command))
        codes = [job.wait() for job in jobs]
        return 0 if all(code == 0 for code in codes) else 1
    if args.check or not any((args.convert, args.train, args.evaluate, args.profile, args.report)):
        print(json.dumps(check(layout), indent=2)); return 0
    if args.report:
        print(report(args.output, layout))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f'{args.condition or "pipeline"}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.convert:
            for condition in ((args.condition,) if args.condition else ('local', 'communicating_ccpd')):
                print(conversion(condition, args.output, layout))
        elif args.profile:
            print(json.dumps(profile(args, layout), indent=2))
            print(report(args.output, layout))
        elif args.evaluate:
            if args.condition is None or args.agents is None:
                parser.error('--evaluate requires --condition and --agents')
            path = stage_path(args.output, args.condition, args.agents) / 'best.pt'
            actor, _ = load_checkpoint(path)
            print(json.dumps(validation(actor, layout, args.agents, args.smoke), indent=2))
        else:
            for agents in ((1, 10) if args.curriculum else (args.agents or 1,)):
                result = train_stage(args, args.condition, agents, layout)
                print(json.dumps(result, indent=2), flush=True)
                if result['status'] != 'passed':
                    break
            print(report(args.output, layout))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
