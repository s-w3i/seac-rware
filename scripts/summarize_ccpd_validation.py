#!/usr/bin/env python3
"""Summarize the frozen diagnostic suite without treating episodes as training seeds."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from validate_ccpd import METHODS, SCENARIOS, SEEDS, training_dir, write_json

FIELDS = ('steps', 'completed_cycles', 'conflict_attempts', 'movement_denied', 'wait_steps',
          'navigation_decisions', 'team_stall_events', 'deliveries', 'pickups')
RATE_NAMES = ('cycles_per_1000_steps', 'conflicts_per_1000_steps', 'denied_per_1000_steps',
              'wait_per_1000_steps', 'team_stalls_per_1000_steps', 'deliveries_per_1000_steps',
              'pickups_per_1000_steps', 'conflicts_per_navigation', 'denied_per_navigation')


def rates(totals):
    totals = np.asarray(totals)
    steps, nav = totals[..., 0], totals[..., 5]
    if np.any(steps <= 0):
        raise ValueError('Cannot aggregate zero observation time')
    return np.stack([*(1000 * totals[..., i] / steps for i in (1, 2, 3, 4, 6, 7, 8)),
                     np.divide(totals[..., 2], nav, out=np.zeros_like(nav, dtype=float), where=nav > 0),
                     np.divide(totals[..., 3], nav, out=np.zeros_like(nav, dtype=float), where=nav > 0)], axis=-1)


def aggregate(rows):
    totals = np.array([[r.get(k, 0) for k in FIELDS] for r in rows], dtype=float).sum(axis=0)
    durations = [v for r in rows for v in r['cycle_durations']]
    result = dict(zip(RATE_NAMES, rates(totals).tolist()))
    result.update(episodes=len(rows), actual_steps=int(totals[0]),
                  mean_cycle_time=float(np.mean(durations)) if durations else None,
                  p95_cycle_time=float(np.percentile(durations, 95)) if durations else None,
                  completed_cycle_count=len(durations),
                  mean_per_robot_completions=np.mean([r['cycles_per_robot'] for r in rows], axis=0).tolist(),
                  mean_per_robot_unfinished_age=np.mean([r['unfinished_task_ages'] for r in rows], axis=0).tolist(),
                  max_unfinished_task_age=max(r['max_unfinished_task_age'] for r in rows),
                  mean_unfinished_task_age=float(np.mean([r['unfinished_task_ages'] for r in rows])),
                  zero_cycle_fraction=float(np.mean([r['zero_cycle_fraction'] for r in rows])),
                  mean_unfinished_tasks=float(np.mean([r['unfinished_tasks'] for r in rows])))
    return result


def paired_bootstrap(left, right, draws=10000):
    """Each array is training seed x matched episode seed x additive statistic."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 3 or left.shape[-1] != len(FIELDS):
        raise ValueError('Paired arrays must have identical seed/episode/statistic axes')
    rng = np.random.default_rng(1729)
    n, episodes, _ = left.shape
    samples = []
    for start in range(0, draws, 250):
        count = min(250, draws-start)
        training = rng.integers(n, size=(count, n))
        episode = rng.integers(episodes, size=(count, n, episodes))
        a = left[training[..., None], episode].sum(axis=2)
        b = right[training[..., None], episode].sum(axis=2)
        samples.append((rates(a) - rates(b)).mean(axis=1))
    samples = np.concatenate(samples)
    point = (rates(left.sum(axis=1)) - rates(right.sum(axis=1))).mean(axis=0)
    interval = np.percentile(samples, [2.5, 97.5], axis=0)
    return {name: dict(difference=float(point[i]), lower=float(interval[0, i]), upper=float(interval[1, i]))
            for i, name in enumerate(RATE_NAMES)}


def load_group(output, method, kind, scenario):
    groups = []
    for seed in range(3):
        path = output / 'evaluation' / method / f'seed_{seed}' / f'{kind}_{scenario}.jsonl'
        if not path.exists():
            return None
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 50 or [r['seed'] for r in rows] != SEEDS:
            raise ValueError(f'Incomplete/duplicate episode suite: {path}')
        if any(r['training_seed'] != seed or r['method'] != method or r['scenario'] != scenario
               or r['checkpoint_kind'] != kind for r in rows):
            raise ValueError(f'Mislabeled evaluation: {path}')
        groups.append(rows)
    return groups


def matrix(groups):
    return np.array([[[r.get(k, 0) for k in FIELDS] for r in rows] for rows in groups], dtype=float)


def training_diagnostics(output, methods):
    records = []
    for method in methods:
        for seed in range(3):
            directory = training_dir(method, seed, output)
            if not (directory / 'summary.json').exists():
                continue
            rows = [json.loads(s) for s in (directory / 'metrics.jsonl').read_text().splitlines()]
            rows = [r for r in rows if r.get('kind') == 'learning']
            record = dict(method=method, training_seed=seed, summary=json.loads((directory / 'summary.json').read_text()))
            for key in ('ccpd_events', 'ccpd_successful_events', 'ccpd_failed_events', 'ccpd_censored_events', 'ccpd_selected_samples'):
                record[key + '_total'] = sum(r.get(key, 0) for r in rows)
            for key in ('ccpd_auxiliary_loss', 'ccpd_weighted_auxiliary_loss', 'ccpd_selected_fraction', 'ccpd_selected_noop_fraction'):
                record[key + '_update_mean'] = float(np.mean([r.get(key, 0) for r in rows]))
            records.append(record)
    return records


def summarize(output):
    methods = (*METHODS, 'random', 'all_conflict')
    report = dict(complete=False, per_seed=[], differences=[], transfer=[],
                  estimand='Equal-weight mean over training seeds; rates pooled over episodes within each training seed. Cycle durations pooled within each training seed, then seed statistics averaged.',
                  uncertainty='10,000 paired hierarchical bootstrap resamples, RNG seed 1729; matched training seeds then matched episode seeds. Pointwise 95% intervals; no familywise guarantee; only three training seeds.',
                  training=training_diagnostics(output, methods))
    groups = {}
    missing = []
    for kind in ('last', 'best'):
        for scenario in SCENARIOS if kind == 'last' else ('reference',):
            for method in methods:
                value = load_group(output, method, kind, scenario)
                key = (kind, scenario, method)
                if value is None:
                    missing.append('/'.join(key))
                    continue
                groups[key] = value
                for seed, rows in enumerate(value):
                    report['per_seed'].append(dict(method=method, training_seed=seed, checkpoint_kind=kind,
                                                   scenario=scenario, **aggregate(rows)))
            for baseline in ('mappo', 'shared_ppo', 'random', 'all_conflict'):
                a, b = groups.get((kind, scenario, 'ccpd')), groups.get((kind, scenario, baseline))
                if a is not None and b is not None:
                    report['differences'].append(dict(checkpoint_kind=kind, scenario=scenario, comparison=f'ccpd - {baseline}',
                                                       metrics=paired_bootstrap(matrix(a), matrix(b))))
    for (kind, scenario, method), value in groups.items():
        reference = groups.get((kind, 'reference', method))
        if scenario != 'reference' and reference is not None:
            for seed in range(3):
                old = aggregate(reference[seed])['cycles_per_1000_steps']
                new = aggregate(value[seed])['cycles_per_1000_steps']
                report['transfer'].append(dict(method=method, training_seed=seed, scenario=scenario,
                                               throughput_change=new-old, throughput_change_percent=100*(new/old-1) if old else None))
    report.update(complete=not missing, missing=missing)
    write_json(output / 'summary.json', report)
    lines = ['# CCPD diagnostic comparison', '',
             '**Complete suite.**' if report['complete'] else '**Interim report: pending experiments are not treated as results.**', '',
             'Final checkpoints are primary; validation-selected best checkpoints are separate. Seeds 2000–2049 are diagnostic held-out episodes. Reserved seeds 3000–3049 have not been used.', '',
             'The old comparison used training validation seeds 1000–1003. Geometric reflections/rotation do not establish generalization to arbitrary warehouse topologies. Team-stall events are a proxy, not proven deadlocks.', '',
             report['estimand'], '', report['uncertainty'], '',
             'Cycle-time statistics exclude unfinished tasks; inspect unfinished ages and per-robot completions in summary.json. Latency is excluded from rankings because these CPU evaluations may run concurrently.', '']
    def fmt(x):
        return 'undefined' if x is None else f'{x:.3f}'
    for kind in ('last', 'best'):
        for scenario in SCENARIOS if kind == 'last' else ('reference',):
            rows = [r for r in report['per_seed'] if r['checkpoint_kind'] == kind and r['scenario'] == scenario]
            if not rows:
                continue
            lines += [f'## {kind}: {scenario}', '',
                      '| Method | Training seed | Cycles/1k | Conflicts/1k | Denied/1k | Wait/1k | Mean cycle | P95 cycle | Stall/1k | Max unfinished age |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for r in rows:
                keys = ('cycles_per_1000_steps', 'conflicts_per_1000_steps', 'denied_per_1000_steps', 'wait_per_1000_steps',
                        'mean_cycle_time', 'p95_cycle_time', 'team_stalls_per_1000_steps', 'max_unfinished_task_age')
                lines.append(f"| {r['method']} | {r['training_seed']} | " + ' | '.join(fmt(r[k]) for k in keys) + ' |')
            lines += ['', '| CCPD minus comparator | Throughput difference [95% CI] | Conflict-rate difference [95% CI] |', '|---|---:|---:|']
            for d in report['differences']:
                if d['checkpoint_kind'] == kind and d['scenario'] == scenario:
                    values = [d['metrics'][m] for m in ('cycles_per_1000_steps', 'conflicts_per_1000_steps')]
                    lines.append(f"| {d['comparison']} | " + ' | '.join(f"{v['difference']:.3f} [{v['lower']:.3f}, {v['upper']:.3f}]" for v in values) + ' |')
            lines.append('')
    lines += ['## Decision', '']
    if missing:
        lines += [f'{len(missing)} method/checkpoint/scenario groups remain pending. Do not improve or select the model using this partial report.']
    else:
        # A conservative directional decision: a zero-crossing throughput interval
        # does not prove equivalence, and nonsignificant control differences do not prove a match.
        comparisons = [d for d in report['differences'] if d['checkpoint_kind'] == 'last']
        adverse = [d for d in comparisons if d['metrics']['cycles_per_1000_steps']['upper'] < 0
                   or d['metrics']['conflicts_per_1000_steps']['lower'] > 0]
        robust = all(d['metrics']['conflicts_per_1000_steps']['upper'] < 0
                     and d['metrics']['cycles_per_1000_steps']['lower'] >= 0 for d in comparisons)
        if adverse:
            lines += ['A throughput loss or conflict increase is supported in at least one diagnostic comparison. Investigate those failure scenarios before changing the architecture:', '']
            lines += [f"- {d['scenario']}: {d['comparison']}" for d in adverse]
        elif robust:
            lines += ['The directional benefit survives every tested scenario and auxiliary control. Preserve the mechanism; prioritize additional training seeds and broader topology tests.']
        else:
            lines += ['The complete suite does not establish consistent superiority over both baselines and auxiliary controls. Prioritize additional independent training seeds and inspect scenario-specific failures. Intervals crossing zero do not establish equivalence or show that controls match CCPD.']
        lines += ['', 'These are exploratory pointwise comparisons. No practical throughput-equivalence margin was specified, so similar point estimates alone are not called preserved throughput.']
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ccpd_diagnostic')
    args = parser.parse_args()
    summarize(args.output.resolve())
