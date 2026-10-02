"""Locked holdout evaluation and paired reporting for Stage 4."""
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from navigation_train import evaluate, load_checkpoint, sha, write_json
from report_navigation import differences, summary as navigation_summary
from run_stage4 import (ANALYSIS, CONDITIONS, SEEDS, append, fingerprint, immutable_json, load_layout,
                        read_json, read_rows, setup_device, stage_path, verify_study)


def evaluation_suites(method, study):
    suites = [dict(name='reference_long', kind='best', count=50, replicates=3),
              dict(name='reference_long', kind='last', count=10, replicates=1),
              dict(name='deterministic', kind='best', count=20, replicates=1, deterministic=True),
              dict(name='changed_starts', kind='best', count=20, replicates=1, profile='changed_starts'),
              dict(name='changed_tasks', kind='best', count=20, replicates=1, profile='changed_tasks')]
    if CONDITIONS[method][0]:
        suites += [dict(name=f'loss_{loss}', kind='best', count=20, replicates=1, loss=loss)
                   for loss in (0., .3, 1.)]
        suites += [dict(name='delay_2', kind='best', count=20, replicates=1, delay=2)]
    result = []
    for spec in suites:
        spec = {**dict(steps=10000, profile='reference', loss=.1, delay=1, deterministic=False), **spec}
        # Smoke evaluations must never consume the production holdout.
        if study['smoke']:
            spec.update(steps=32, count=2, replicates=1)
        result.append(spec)
    return result


def eval_seeds(study, count):
    return list(range(100500, 100500 + count)) if study['smoke'] else ANALYSIS['holdout_seeds'][:count]


def checkpoint_key(method, seed, agents, kind):
    return f'{method}/seed_{seed}/n{agents}/{kind}.pt'


def lock_evaluation(output, study):
    output = Path(output)
    if verify_study(output) != study:
        raise ValueError('Study protocol mismatch')
    if not set(ANALYSIS['final_fleets']) <= {n for n, _ in study['schedule']}:
        raise ValueError('Final evaluation requires completed 40- and 50-robot stages')
    checkpoints, summaries = {}, {}
    for method in CONDITIONS:
        for seed in SEEDS:
            cumulative = 0
            for agents, updates in study['schedule']:
                stage = stage_path(output, method, seed, agents)
                result = read_json(stage / 'summary.json')
                cumulative += updates * study['joint_steps_per_update']
                if (result['status'] != 'complete' or result['completed_updates'] != updates
                        or result['condition'] != method or result['seed'] != seed or result['agents'] != agents
                        or result['cumulative_env_steps'] != cumulative or result['smoke'] != study['smoke']
                        or result['protocol_sha256'] != fingerprint(read_json(stage / 'protocol.json'))):
                    raise ValueError(f'Incomplete/mislabeled training stage: {stage}')
                for name, digest in result['artifacts'].items():
                    if sha(stage / name) != digest:
                        raise ValueError(f'Training artifact changed: {stage / name}')
                summaries[str((stage / 'summary.json').relative_to(output))] = sha(stage / 'summary.json')
                if agents in ANALYSIS['final_fleets']:
                    for kind in ('best', 'last'):
                        checkpoints[checkpoint_key(method, seed, agents, kind)] = sha(stage / f'{kind}.pt')
    lock = dict(study_sha256=fingerprint(study), analysis=ANALYSIS, smoke=study['smoke'],
                checkpoints=checkpoints, training_summaries=summaries,
                suites={method: evaluation_suites(method, study) for method in CONDITIONS})
    immutable_json(output / 'evaluation_lock.json', lock)
    return lock


def require_lock(output, study):
    lock = read_json(Path(output) / 'evaluation_lock.json')
    if (verify_study(output) != study or lock['study_sha256'] != fingerprint(study)
            or lock['analysis'] != ANALYSIS or lock['smoke'] != study['smoke']
            or lock['suites'] != {m: evaluation_suites(m, study) for m in CONDITIONS}):
        raise ValueError('Final evaluation lock is stale')
    expected = {checkpoint_key(m, s, n, k) for m in CONDITIONS for s in SEEDS
                for n in ANALYSIS['final_fleets'] for k in ('best', 'last')}
    if set(lock['checkpoints']) != expected:
        raise ValueError('Incomplete checkpoint lock')
    for name, digest in {**lock['checkpoints'], **lock['training_summaries']}.items():
        if sha(Path(output) / name) != digest:
            raise ValueError(f'Locked artifact changed: {name}')
    return lock


def evaluation_files(output, study, lock):
    for method in CONDITIONS:
        for seed in SEEDS:
            for agents in ANALYSIS['final_fleets']:
                for spec in evaluation_suites(method, study):
                    key = checkpoint_key(method, seed, agents, spec['kind'])
                    for replicate in range(spec['replicates']):
                        folder = Path(output) / 'evaluation' / method / f'seed_{seed}' / f'n{agents}'
                        path = folder / f"{spec['kind']}_{spec['name']}_rng_{replicate}.jsonl"
                        inputs = dict(study_sha256=fingerprint(study), lock_sha256=fingerprint(lock),
                                      checkpoint_sha256=lock['checkpoints'][key], method=method,
                                      training_seed=seed, agents=agents, checkpoint_kind=spec['kind'],
                                      suite=spec['name'], action_replicate=replicate, spec=spec,
                                      seeds=eval_seeds(study, spec['count']), map_sha256=study['map_sha256'])
                        yield path, Path(output) / key, inputs


def valid_rows(rows, inputs, complete=True):
    spec = inputs['spec']
    if len(rows) > len(inputs['seeds']) or (complete and len(rows) != len(inputs['seeds'])):
        return False
    for row, seed in zip(rows, inputs['seeds']):
        def finite(value):
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, list):
                return all(finite(v) for v in value)
            return not isinstance(value, float) or math.isfinite(value)
        expected = {k: inputs[k] for k in ('method', 'training_seed', 'agents', 'checkpoint_kind',
                                           'suite', 'action_replicate', 'checkpoint_sha256')}
        expected.update(episode_id=seed, seed=seed + (100 if spec['profile'] == 'changed_starts' else 0),
                        task_seed=seed + (10050 if spec['profile'] == 'changed_tasks' else 10000),
                        steps=spec['steps'], requested_steps=spec['steps'], deterministic=spec['deterministic'],
                        profile=spec['profile'], delay=spec['delay'], geometry_sha256=inputs['map_sha256'],
                        inputs_sha256=fingerprint(inputs),
                        packet_loss=spec['loss'] if CONDITIONS[inputs['method']][0] else None)
        if (any(row.get(k) != v for k, v in expected.items())
                or len(row.get('max_task_ages', [])) != inputs['agents']
                or len(row.get('cycles_per_robot', [])) != inputs['agents']
                or 'progress_failure' not in row or not finite(row)):
            return False
    return True


def receipt_valid(path, inputs):
    marker = path.with_name(path.name + '.done.json')
    return (path.exists() and marker.exists()
            and read_json(marker) == dict(inputs=inputs, sha256=sha(path))
            and valid_rows(read_rows(path), inputs))


def evaluate_campaign(output, study):
    output = Path(output)
    lock = require_lock(output, study)
    layout = load_layout()
    setup_device('cpu', 0)
    loaded, actor = None, None
    files = list(evaluation_files(output, study, lock))
    for index, (path, checkpoint, inputs) in enumerate(files):
        path.parent.mkdir(parents=True, exist_ok=True)
        if receipt_valid(path, inputs):
            continue
        if path.exists() or path.with_name(path.name + '.done.json').exists():
            raise ValueError(f'Invalid evaluation receipt: {path}')
        if loaded != checkpoint:
            actor, _ = load_checkpoint(checkpoint)
            loaded = checkpoint
        if sha(checkpoint) != inputs['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during final evaluation')
        spec = inputs['spec']
        partial = path.with_name(path.name + '.partial')
        rows = []
        if partial.exists():
            lines = partial.read_text().splitlines()
            for i, line in enumerate(lines):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if i != len(lines) - 1:
                        raise
            if not valid_rows(rows, inputs, complete=False):
                raise ValueError(f'Stale/mislabeled partial evaluation: {partial}')
            partial.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
        for seed in inputs['seeds'][len(rows):]:
            row = evaluate(actor, layout, [seed], spec['steps'], replicate=inputs['action_replicate'],
                           n_agents=inputs['agents'], deterministic=spec['deterministic'],
                           profile=spec['profile'], loss=spec['loss'], delay=spec['delay'],
                           allow_stage4_holdout=not study['smoke'])[0]
            row.update({k: inputs[k] for k in ('method', 'training_seed', 'agents', 'checkpoint_kind',
                                              'suite', 'checkpoint_sha256')})
            row.update(delay=spec['delay'], inputs_sha256=fingerprint(inputs))
            append(partial, row)
            rows.append(row)
            write_json(output / 'evaluation_status.json', dict(file=index+1, files=len(files),
                       path=str(path.relative_to(output)), episodes=len(rows), timestamp=time.time()))
        if not valid_rows(rows, inputs):
            raise ValueError(f'Incomplete evaluation: {path}')
        partial.replace(path)
        write_json(path.with_name(path.name + '.done.json'), dict(inputs=inputs, sha256=sha(path)))
        print(f'Evaluated {path.relative_to(output)}', flush=True)
    return dict(status='complete', files=len(files), smoke=study['smoke'])


def summary(rows):
    result = navigation_summary(rows)
    durations = [d for r in rows for d in r['cycle_durations']]
    result.update(p99_cycle_time=float(np.percentile(durations, 99)) if durations else None,
                  p95_unfinished_age=float(np.percentile([a for r in rows for a in r['unfinished_task_ages']], 95)),
                  max_task_age=max(max(r['max_task_ages']) for r in rows),
                  mean_max_task_age=float(np.mean([max(r['max_task_ages']) for r in rows])),
                  mean_per_robot_max_task_age=np.mean([r['max_task_ages'] for r in rows], axis=0).tolist(),
                  progress_failure=float(np.mean([r['progress_failure'] for r in rows])),
                  individual_progress_failure=float(np.mean([r['individual_progress_failure'] for r in rows])),
                  robot_failure_fraction=float(np.mean([r['robot_failure_fraction'] for r in rows])))
    return result


def paired_progress(left, right, draws=10000):
    fields = ('progress_failure', 'individual_progress_failure', 'robot_failure_fraction', 'aged_robot_time_fraction')
    a = {(r['training_seed'], r['episode_id'], r['action_replicate']): r for r in left}
    b = {(r['training_seed'], r['episode_id'], r['action_replicate']): r for r in right}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError('Duplicate/unmatched paired evaluation identities')
    seeds = sorted({k[0] for k in a})
    episodes = sorted({k[1] for k in a})
    reps = sorted({k[2] for k in a})
    if len(a) != len(seeds) * len(episodes) * len(reps):
        raise ValueError('Incomplete paired evaluation block')
    delta = np.array([[[[float(a[s,e,r][f]) - float(b[s,e,r][f]) for f in fields]
                        for r in reps] for e in episodes] for s in seeds]).mean(2)
    rng = np.random.default_rng(ANALYSIS['bootstrap_seed'])
    samples = []
    n, e, _ = delta.shape
    for start in range(0, draws, 250):
        count = min(250, draws-start)
        training = rng.integers(n, size=(count, n))
        scenario = rng.integers(e, size=(count, n, e))
        samples.append(delta[training[..., None], scenario].mean((1, 2)))
    intervals = np.percentile(np.concatenate(samples), [2.5, 97.5], axis=0)
    point = delta.mean((0, 1))
    return {name: dict(difference=float(point[i]), lower=float(intervals[0, i]), upper=float(intervals[1, i]))
            for i, name in enumerate(fields)}


def comparison(candidate_rows, baseline_rows, draws=10000):
    metrics = dict(differences(candidate_rows, baseline_rows, draws),
                   **paired_progress(candidate_rows, baseline_rows, draws))
    safeguards = []
    for seed in SEEDS:
        a = summary([r for r in candidate_rows if r['training_seed'] == seed])
        b = summary([r for r in baseline_rows if r['training_seed'] == seed])
        safeguards.append(dict(seed=seed, no_progress_failure=a['progress_failure'] == 0,
            throughput_noninferior=a['cycles_per_1000_steps'] >= .98 * b['cycles_per_1000_steps'],
            no_worse_starvation=all(a[k] <= b[k] for k in (
                'robot_failure_fraction', 'aged_robot_time_fraction', 'p95_unfinished_age', 'max_task_age'))))
    base = float(np.mean([summary([r for r in baseline_rows if r['training_seed'] == seed])['cycles_per_1000_steps']
                          for seed in SEEDS]))
    throughput, failure = metrics['cycles_per_1000_steps'], metrics['progress_failure']
    productivity = throughput['difference'] >= .03 * base and throughput['lower'] > 0
    reliability = (failure['difference'] < 0 and failure['upper'] < 0
                   and throughput['lower'] >= -.02 * base)
    passed = all(all(v for k, v in row.items() if k != 'seed') for row in safeguards)
    return dict(metrics=metrics, per_seed_safeguards=safeguards, throughput_route=bool(productivity),
                reliability_route=bool(reliability), qualifies=bool(passed and (productivity or reliability)))


def report(output, study):
    output = Path(output)
    stages, faults, isolated = [], [], []
    for method in CONDITIONS:
        for seed in SEEDS:
            for agents, updates in study['schedule']:
                path = stage_path(output, method, seed, agents) / 'summary.json'
                stages.append(read_json(path) if path.exists() else dict(condition=method, seed=seed,
                              agents=agents, status='pending', completed_updates=0, expected_updates=updates))
            fault = output / method / f'seed_{seed}' / 'failure.json'
            if fault.exists():
                faults.append(dict(condition=method, seed=seed, **read_json(fault)))
            single = stage_path(output, method, seed, 1) / 'isolated.json'
            if single.exists():
                isolated.append(dict(condition=method, seed=seed,
                    **{k: v for k, v in read_json(single).items() if k != 'rows'}))
    result = dict(complete=False, smoke=study['smoke'], study_sha256=fingerprint(study),
                  scope=ANALYSIS['scope'], stages=stages, technical_failures=faults,
                  isolated_screens=isolated,
                  per_training_seed=[], paired_comparisons=[], decisions=[])
    groups = {}
    missing = []
    if (output / 'evaluation_lock.json').exists():
        lock = require_lock(output, study)
        expected = list(evaluation_files(output, study, lock))
        known = {path for path, _, _ in expected}
        if set((output / 'evaluation').rglob('*.jsonl')) - known:
            raise ValueError('Unknown evaluation files cannot be included in the study')
        for path, _, inputs in expected:
            if not path.exists():
                missing.append(str(path.relative_to(output)))
                continue
            if not receipt_valid(path, inputs):
                raise ValueError(f'Invalid evaluation receipt or labels: {path}')
            key = (inputs['method'], inputs['agents'], inputs['checkpoint_kind'], inputs['suite'])
            groups.setdefault(key, []).extend(read_rows(path))
        result['complete'] = not missing
    else:
        missing = ['evaluation_lock.json: training and checkpoint locking must finish first']
    result['missing_evaluation'] = missing
    if result['complete']:
        for (method, agents, kind, suite), rows in groups.items():
            for seed in SEEDS:
                result['per_training_seed'].append(dict(method=method, agents=agents, checkpoint_kind=kind,
                    suite=suite, training_seed=seed, **summary([r for r in rows if r['training_seed'] == seed])))
        for candidate, baseline in ANALYSIS['comparisons']:
            both = []
            for agents in ANALYSIS['final_fleets']:
                values = comparison(groups[candidate, agents, 'best', 'reference_long'],
                                    groups[baseline, agents, 'best', 'reference_long'], ANALYSIS['bootstrap_draws'])
                for guard in values['per_seed_safeguards']:
                    guard['validation_progress_passed'] = next(r['progress_passed'] for r in stages
                        if r['condition'] == candidate and r['seed'] == guard['seed'] and r['agents'] == agents)
                values['qualifies'] = values['qualifies'] and all(
                    guard['validation_progress_passed'] for guard in values['per_seed_safeguards'])
                both.append(values['qualifies'])
                result['paired_comparisons'].append(dict(candidate=candidate, baseline=baseline,
                                                        agents=agents, **values))
            result['decisions'].append(dict(candidate=candidate, baseline=baseline,
                recommended=all(both) and not study['smoke'], interpretation='Smoke data only; no research recommendation'
                if study['smoke'] else 'Supported at both target fleets' if all(both) else
                'Inconclusive about added complexity; retain the simpler reference provisionally'))
        if any(f['status'] != 'recovered' for f in faults):
            for decision in result['decisions']:
                decision.update(recommended=False, interpretation='Unresolved technical failure; study not promotable')
    if (output / 'cost_estimate.json').exists():
        result['cost_estimate'] = read_json(output / 'cost_estimate.json')
    if (output / 'runtime_repair.json').exists():
        result['runtime_repair'] = read_json(output / 'runtime_repair.json')
    write_json(output / 'summary.json', result)
    lines = ['# Stage 4 fixed-map adaptation study', '',
             '**SMOKE DATA — not research evidence.**' if study['smoke'] else ANALYSIS['scope'], '',
             f'Final evaluation complete: {result["complete"]}.', '',
             '## Training status', '', '| Method | Seed | Robots | Status | Updates | Progress screen |',
             '|---|---:|---:|---|---:|---|']
    for row in stages:
        lines.append(f'| {row["condition"]} | {row["seed"]} | {row["agents"]} | {row["status"]} '
                     f'| {row["completed_updates"]} | {row.get("progress_passed", "pending")} |')
    if result['complete']:
        lines += ['', '## Primary held-out evaluation', '',
                  '| Method | Robots | Seed | Cycles/1,000 | Progress failure | p95 cycle | p99 cycle | Max task age |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in result['per_training_seed']:
            if row['checkpoint_kind'] == 'best' and row['suite'] == 'reference_long':
                lines.append(f'| {row["method"]} | {row["agents"]} | {row["training_seed"]} '
                             f'| {row["cycles_per_1000_steps"]:.3f} | {row["progress_failure"]:.3f} '
                             f'| {row["p95_cycle_time"]} | {row["p99_cycle_time"]} | {row["max_task_age"]} |')
        lines += ['', '## Paired comparisons', '', ANALYSIS['uncertainty'] + '.', '',
                  '| Candidate − baseline | Robots | Throughput difference [95% CI] | Qualifies |',
                  '|---|---:|---:|---|']
        for row in result['paired_comparisons']:
            d = row['metrics']['cycles_per_1000_steps']
            lines.append(f'| {row["candidate"]} − {row["baseline"]} | {row["agents"]} '
                         f'| {d["difference"]:.3f} [{d["lower"]:.3f}, {d["upper"]:.3f}] | {row["qualifies"]} |')
        lines += ['', 'All checkpoints, robustness suites, per-robot service, incomplete-task ages, '
                  'conflicts per navigation decision, and safeguards are retained in summary.json. '
                  'Completed-cycle percentiles exclude unfinished tasks. Zero observed failures do not '
                  'establish zero risk; action replicates are not independent scenario seeds.']
    else:
        lines += ['', 'Final comparisons are withheld until every required suite has a valid receipt.']
    if faults:
        lines += ['', '## Technical failures', '', *[f'- {r}' for r in faults]]
    if 'runtime_repair' in result:
        lines += ['', '## Recorded runtime correction', '', result['runtime_repair']['reason'],
                  'Original protocol and checkpoints were preserved. Source hashes and verification '
                  'evidence are recorded in runtime_repair.json; resumed checkpoints identify their runtime.']
    (output / 'STAGE4_REPORT.md').write_text('\n'.join(lines) + '\n')
    return dict(complete=result['complete'], report=str(output / 'STAGE4_REPORT.md'),
                completed_stages=sum(r['status'] == 'complete' for r in stages))
