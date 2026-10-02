"""Paired reports for the communication x CCPD factorial experiment."""
import json
from pathlib import Path

import numpy as np

from summarize_ccpd_validation import aggregate, paired_bootstrap, FIELDS
from navigation_train import CONDITIONS, write_json


def summary(rows):
    result = aggregate(rows)
    for name in ('fleet_failure', 'fleet_completion_gaps', 'max_fleet_completion_gap',
                 'aged_endpoint_fraction', 'aged_robot_time_fraction', 'inference_ms_per_fleet_step'):
        result[name] = float(np.mean([r[name] for r in rows]))
    result['packets_per_fleet_step'] = sum(r['packets_sent'] for r in rows) / sum(r['steps'] for r in rows)
    result['mean_longest_stationary'] = np.mean([r['longest_stationary'] for r in rows], axis=0).tolist()
    result['repetition_windows_per_1000_steps'] = 1000 * sum(sum(r['repetition_windows']) for r in rows) / sum(r['steps'] for r in rows)
    result['denial_reasons_per_1000_steps'] = {k: 1000 * sum(r['denial_reasons'][k] for r in rows) / sum(r['steps'] for r in rows)
                                            for k in ('boundary', 'rack', 'traffic')}
    return result


def paired_arrays(left, right):
    """Keep the three action realizations together inside each matched episode."""
    def grouped(rows):
        lookup = {(r['training_seed'], r['episode_id'], r['action_replicate']): r for r in rows}
        if len(lookup) != len(rows):
            raise ValueError('Duplicate evaluation identity')
        return lookup
    a, b = grouped(left), grouped(right)
    if set(a) != set(b):
        raise ValueError('Unmatched evaluation identities')
    seeds = sorted({k[0] for k in a})
    episodes = sorted({k[1] for k in a})
    reps = sorted({k[2] for k in a})
    if len(a) != len(seeds)*len(episodes)*len(reps):
        raise ValueError('Incomplete factorial evaluation group')
    def array(d, fields):
        return np.array([[[[d[s, e, r].get(k, 0) for k in fields] for r in reps] for e in episodes] for s in seeds], float).mean(2)
    return array(a, FIELDS), array(b, FIELDS), array(a, ['fleet_failure', 'aged_endpoint_fraction']), array(b, ['fleet_failure', 'aged_endpoint_fraction'])


def differences(left, right, draws=10000):
    a, b, failures_a, failures_b = paired_arrays(left, right)
    result = paired_bootstrap(a, b, draws)
    rng = np.random.default_rng(1729)
    n, e, _ = failures_a.shape
    values = []
    for start in range(0, draws, 250):
        count = min(250, draws-start)
        training = rng.integers(n, size=(count, n))
        episode = rng.integers(e, size=(count, n, e))
        values.append((failures_a[training[..., None], episode] - failures_b[training[..., None], episode]).mean((1, 2)))
    bounds = np.percentile(np.concatenate(values), [2.5, 97.5], axis=0)
    for i, name in enumerate(('fleet_failure', 'aged_endpoint_fraction')):
        result[name] = dict(difference=float((failures_a-failures_b)[..., i].mean()), lower=float(bounds[0, i]), upper=float(bounds[1, i]))
    return result


def recommendation(groups, per_seed):
    decisions = []
    for candidate, baseline, factor in [('communicating', 'local', 'communication'),
            ('communicating_ccpd', 'local_ccpd', 'communication'),
            ('local_ccpd', 'local', 'ccpd'), ('communicating_ccpd', 'communicating', 'ccpd')]:
        screen = []
        for seed in range(3):
            a = per_seed[candidate, seed, 'reference_long']
            b = per_seed[baseline, seed, 'reference_long']
            short_a = per_seed[candidate, seed, 'reference_short']
            short_b = per_seed[baseline, seed, 'reference_short']
            screen.append(dict(seed=seed, operational=a['fleet_failure'] <= .01,
                throughput=a['cycles_per_1000_steps'] >= .98*b['cycles_per_1000_steps'] and short_a['cycles_per_1000_steps'] >= .98*short_b['cycles_per_1000_steps'],
                unfinished=all(a[k] <= b[k] for k in ('aged_endpoint_fraction', 'aged_robot_time_fraction',
                                                     'mean_unfinished_task_age', 'max_unfinished_task_age')),
                lower_conflicts=short_a['conflicts_per_1000_steps'] < short_b['conflicts_per_1000_steps'] and a['conflicts_per_1000_steps'] < b['conflicts_per_1000_steps'],
                counterfactual=all(per_seed[candidate, seed, profile]['cycles_per_1000_steps'] >=
                    .95*summary([r for r in groups[candidate, 'reference_long'] if r['training_seed']==seed and r['episode_id']<2020])['cycles_per_1000_steps']
                    for profile in ('changed_starts', 'changed_tasks'))))
        diff = differences(groups[candidate, 'reference_long'], groups[baseline, 'reference_long'])
        safeguards = all(all(r[k] for k in ('operational', 'throughput', 'unfinished', 'counterfactual')) for r in screen)
        if factor == 'communication':
            base = np.mean([per_seed[baseline, s, 'reference_long']['cycles_per_1000_steps'] for s in range(3)])
            improvement = (diff['cycles_per_1000_steps']['difference'] >= .03*base and diff['cycles_per_1000_steps']['lower'] > 0) or (
                diff['fleet_failure']['difference'] <= -.02 and diff['fleet_failure']['upper'] < 0)
        else:
            improvement = all(r['lower_conflicts'] for r in screen)
        decisions.append(dict(candidate=candidate, comparator=baseline, factor=factor, per_seed=screen,
                              differences=diff, promising=bool(safeguards and improvement),
                              interpretation='promising; requires further independent confirmation' if safeguards and improvement else
                              'prefer simpler condition provisionally or collect more evidence; not an equivalence claim'))
    return decisions


def generate_report(output):
    output = Path(output)
    groups, records, per_seed = {}, [], {}
    # Reporting must not silently include partial, stale or mislabeled evaluations.
    from run_decentralized_navigation import suites, receipt_valid
    for method, (communication, _) in CONDITIONS.items():
        for seed in range(3):
            for kind in ('last', 'best'):
                expected = {f"{kind}_{s['suite']}_rng_{s['replicate']}.jsonl": s for s in suites(communication, kind)}
                folder = output / 'evaluation' / method / f'seed_{seed}'
                actual = {p.name for p in folder.glob(kind+'_*.jsonl')}
                if actual != set(expected):
                    raise ValueError(f'Incomplete/unknown evaluation files in {folder}: {kind}')
                for name, spec in expected.items():
                    path = folder / name
                    marker = path.with_name(path.name+'.done.json')
                    if not marker.exists() or not receipt_valid(path, json.loads(marker.read_text())['inputs']):
                        raise ValueError(f'Incomplete evaluation receipt: {path}')
                    rows = [json.loads(l) for l in path.read_text().splitlines()]
                    if [r['episode_id'] for r in rows] != list(range(2000, 2000+spec['count'])) or any(
                        r['method'] != method or r['training_seed'] != seed or r['checkpoint_kind'] != kind or
                        r['suite'] != spec['suite'] or r['action_replicate'] != spec['replicate'] or
                        r['steps'] != spec['steps'] for r in rows):
                        raise ValueError(f'Mislabeled/incomplete evaluation: {path}')
    for method in CONDITIONS:
        for seed in range(3):
            for f in sorted((output / 'evaluation' / method / f'seed_{seed}').glob('*.jsonl')):
                rows = [json.loads(l) for l in f.read_text().splitlines()]
                by = {}
                for r in rows:
                    key = (r['checkpoint_kind'], r['suite'])
                    by.setdefault(key, []).append(r)
                for (kind, suite), subset in by.items():
                    groups.setdefault((kind, method, suite), []).extend(subset)
    for (kind, method, suite), rows in sorted(groups.items()):
        for seed in range(3):
            subset = [r for r in rows if r['training_seed']==seed]
            value = summary(subset)
            records.append(dict(checkpoint_kind=kind, method=method, training_seed=seed, suite=suite, **value))
            if kind == 'last':
                per_seed[method, seed, suite] = value
    last = {(m, suite): rows for (kind, m, suite), rows in groups.items() if kind == 'last'}
    decisions = recommendation(last, per_seed)
    comparisons = []
    for kind, candidate, suite in groups:
        # Compare each factor at fixed settings of the other factor.
        comparators = {'communicating': ['local'], 'local_ccpd': ['local'],
                       'communicating_ccpd': ['local_ccpd', 'communicating']}.get(candidate, [])
        for baseline in comparators:
            if (kind, baseline, suite) in groups:
                comparisons.append(dict(checkpoint_kind=kind, candidate=candidate, comparator=baseline, suite=suite,
                                        metrics=differences(groups[kind, candidate, suite], groups[kind, baseline, suite])))
        if suite in ('changed_starts', 'changed_tasks') or suite.startswith('loss_'):
            rows = groups[kind, candidate, suite]
            identities = {r['episode_id'] for r in rows}
            reference = [r for r in groups[kind, candidate, 'reference_long'] if r['episode_id'] in identities]
            comparisons.append(dict(checkpoint_kind=kind, candidate=candidate, comparator=candidate,
                                    suite=suite+' - reference_long', metrics=differences(rows, reference)))
    report = dict(complete=True, per_training_seed=records, decisions=decisions,
                  paired_comparisons=comparisons,
                  uncertainty='10000 paired hierarchical resamples: matched training seeds then episode IDs; action RNG replicates retained within episode blocks. Pointwise 95% intervals; only three training seeds.',
                  scope='Original layout only. No cross-layout generalization claim. CCPD comparisons concern net usefulness, not success-selection causality.',
                  reserved_seeds=list(range(3000, 3050)))
    write_json(output / 'summary.json', report)
    lines = ['# Decentralized navigation comparison', '', report['scope'], '', report['uncertainty'], '',
             'Rates are per 1,000 fleet steps. Failure means a fleet completion gap ≥500 steps anywhere in the episode. Final and validation-selected checkpoints are separate.',
             'The simulator deadlock counter is a team-stall proxy. Cycle times exclude unfinished cycles; empty cycle statistics are null. Per-robot service, gap counts, stationary intervals, repetition, denial reasons and communication volume are retained in summary.json.', '']
    for kind in ('last', 'best'):
        lines += [f'## {kind} checkpoints', '', '| Suite | Method | Seed | Throughput | Conflicts | Denied | Fleet failure | Aged endpoints | Actor ms/fleet step |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in records:
            if r['checkpoint_kind'] == kind:
                lines.append(f"| {r['suite']} | {r['method']} | {r['training_seed']} | {r['cycles_per_1000_steps']:.3f} | {r['conflicts_per_1000_steps']:.3f} | {r['denied_per_1000_steps']:.3f} | {r['fleet_failure']:.3f} | {r['aged_endpoint_fraction']:.3f} | {r['inference_ms_per_fleet_step']:.3f} |")
        lines.append('')
    lines += ['## Paired long-run differences', '', '| Candidate − comparator | Throughput [95% CI] | Fleet failure [95% CI] | Promising |', '|---|---:|---:|---|']
    for d in decisions:
        a, b = d['differences']['cycles_per_1000_steps'], d['differences']['fleet_failure']
        lines.append(f"| {d['candidate']} − {d['comparator']} | {a['difference']:.3f} [{a['lower']:.3f}, {a['upper']:.3f}] | {b['difference']:.4f} [{b['lower']:.4f}, {b['upper']:.4f}] | {d['promising']} |")
    (output / 'comparison.md').write_text('\n'.join(lines)+'\n')
    next_steps = ['# Next-step recommendation', '', 'These are screening decisions, not proof of superiority or equivalence.', '']
    for d in decisions:
        next_steps.append(f"- {d['candidate']} versus {d['comparator']}: {d['interpretation']}.")
    next_steps += ['', 'Inspect per-seed safeguards in summary.json and packet-loss/outage suites before selecting deployment settings. Do not automatically expand training, enable other layouts, or inspect reserved seeds.']
    (output / 'next_steps.md').write_text('\n'.join(next_steps)+'\n')
    return report
