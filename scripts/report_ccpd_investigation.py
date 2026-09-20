"""Existing-evidence tables, pilot comparisons, and predeclared screening decisions."""
import csv
import json
from pathlib import Path

import numpy as np

from investigate_ccpd import METHODS, MODES, SCENARIOS, original_train, read_rows, write_json
from summarize_ccpd_validation import aggregate, paired_bootstrap, matrix

DIAGNOSTICS = ('ccpd_selected_fraction', 'ccpd_selected_noop_fraction', 'ccpd_auxiliary_loss',
               'ccpd_weighted_auxiliary_loss', 'ccpd_events', 'ccpd_successful_events',
               'ccpd_failed_events', 'ccpd_censored_events', 'approx_kl', 'entropy')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def analyze_existing(prior, output):
    existing = json.loads((prior / 'summary.json').read_text())
    if not existing['complete']:
        raise ValueError('Cannot analyze unfinished campaign as final evidence')
    curves, diagnostic_curves, summaries = [], [], []
    for method in METHODS:
        for seed in range(3):
            directory = original_train(prior, method, seed)
            rows = [r for r in read_rows(directory / 'metrics.jsonl') if r.get('kind') == 'learning']
            validation = read_rows(directory / 'evaluation.jsonl')
            for step in sorted({r['env_steps'] for r in validation}):
                episodes = [r for r in validation if r['env_steps'] == step]
                curves.append(dict(method=method, training_seed=seed, env_steps=step,
                                   validation_cycles_per_1000_steps=float(np.mean([r['cycles_per_1000_steps'] for r in episodes])),
                                   validation_conflicts_per_1000_steps=1000*sum(r['conflict_attempts'] for r in episodes)/sum(r['steps'] for r in episodes)))
            buckets = {}
            for r in rows:
                # Shared 250k-step bins; these are descriptive update means.
                buckets.setdefault((r['env_steps']-1)//250000, []).append(r)
            for bucket, window in sorted(buckets.items()):
                diagnostic_curves.append(dict(method=method, training_seed=seed, bin_start=bucket*250000,
                    bin_end=(bucket+1)*250000, updates=len(window),
                    **{k: float(np.mean([r.get(k, 0) for r in window])) for k in DIAGNOSTICS}))
            selected = sum(r.get('ccpd_selected_samples', 0) for r in rows)
            summary = json.loads((directory / 'summary.json').read_text())
            summaries.append(dict(method=method, training_seed=seed, selected_actions=selected,
                                  selected_noop_fraction=sum(r.get('ccpd_selected_samples', 0)*r.get('ccpd_selected_noop_fraction', 0) for r in rows)/selected if selected else None,
                                  wall_seconds=summary['wall_seconds'], env_steps=summary['env_steps'],
                                  **{k: float(np.mean([r.get(k, 0) for r in rows])) for k in DIAGNOSTICS if k != 'ccpd_selected_noop_fraction'}))
    write_csv(output / 'analysis/learning_curves.csv', curves)
    write_csv(output / 'analysis/auxiliary_curves.csv', diagnostic_curves)
    write_json(output / 'analysis/existing_evidence.json', dict(training=summaries, comparison=existing))
    lines = ['# Existing 20M-step evidence', '',
             'Validation learning curves and auxiliary diagnostics are in the adjacent CSV files. Compare common environment-step values and bins; these are not additional held-out results.', '',
             '| Hypothesis | Evidence to inspect | Limitation |', '|---|---|---|',
             '| Layout-specific learning | Original versus transformed-layout throughput for every method and seed | Reflections and rotation do not establish arbitrary-topology transfer |',
             '| Starvation hidden by fleet averages | Per-robot completions, final unfinished ages, and replay timelines | A large cycle age alone is not proof of a deadlock |',
             '| Auxiliary selection/action mix | Selected action fractions, loss, entropy and KL histories; same-rollout selection audit | The team-advantage filter does not identify which robot caused resolution |', '',
             '| Method | Seed | Selected actions | Selected NOOP fraction | Training hours |', '|---|---:|---:|---:|---:|']
    for r in summaries:
        noop = 'N/A' if r['selected_noop_fraction'] is None else f"{r['selected_noop_fraction']:.4f}"
        lines.append(f"| {r['method']} | {r['training_seed']} | {r['selected_actions']} | {noop} | {r['wall_seconds']/3600:.2f} |")
    (output / 'analysis/evidence.md').write_text('\n'.join(lines)+'\n')


def summarize_episode_group(rows):
    return dict(aggregate(rows), starvation_fraction=float(np.mean(np.asarray([r['unfinished_task_ages'] for r in rows]) >= 500)))


def screening(per_seed):
    by_key = {(r['method'], r['training_seed'], r['scenario']): r for r in per_seed}
    evidence = []
    for seed in range(2):
        success = by_key['successful', seed, 'reference']
        long_success = by_key['successful', seed, 'long_run']
        for baseline in ('random', 'random_action_matched'):
            control = by_key[baseline, seed, 'reference']
            long_control = by_key[baseline, seed, 'long_run']
            evidence.append(dict(training_seed=seed, comparator=baseline,
                fewer_conflicts=success['conflicts_per_1000_steps'] < control['conflicts_per_1000_steps'],
                throughput_within_tolerance=success['cycles_per_1000_steps'] >= .98 * control['cycles_per_1000_steps'],
                starvation_not_increased=long_success['starvation_fraction'] <= long_control['starvation_fraction'],
                throughput_difference=success['cycles_per_1000_steps']-control['cycles_per_1000_steps'],
                conflict_difference=success['conflicts_per_1000_steps']-control['conflicts_per_1000_steps'],
                starvation_difference=long_success['starvation_fraction']-long_control['starvation_fraction']))
    passes = lambda r: all(r[k] for k in ('fewer_conflicts', 'throughput_within_tolerance', 'starvation_not_increased'))
    if all(passes(r) for r in evidence):
        conclusion = 'promote_full_budget_action_matched_comparison'
    elif all(passes(r) for r in evidence if r['comparator'] == 'random') and not all(passes(r) for r in evidence if r['comparator'] == 'random_action_matched'):
        conclusion = 'action_mix_is_a_plausible_explanation'
    else:
        conclusion = 'inconclusive_or_adverse_pilot'
    return dict(conclusion=conclusion, evidence=evidence, tolerance=.02,
                interpretation='Directional screening in two seeds; not equivalence, causality, or publication-level superiority.')


def mean_table(records):
    lines = ['| Scenario | Method | Cycles/1k | Conflicts/1k | Denied/1k | Wait/1k | Mean cycle | P95 cycle | Max unfinished age |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scenario in SCENARIOS:
        for method in dict.fromkeys(r['method'] for r in records):
            group = [r for r in records if r['method'] == method and r['scenario'] == scenario]
            if not group:
                continue
            values = []
            for key in ('cycles_per_1000_steps', 'conflicts_per_1000_steps', 'denied_per_1000_steps', 'wait_per_1000_steps', 'mean_cycle_time', 'p95_cycle_time'):
                # An undefined seed duration must not silently disappear from the mean.
                values.append('undefined' if any(r[key] is None for r in group) else f'{np.mean([r[key] for r in group]):.3f}')
            lines.append(f'| {scenario} | {method} | ' + ' | '.join(values) + f" | {max(r['max_unfinished_task_age'] for r in group)} |")
    return lines


def generate_report(prior, output):
    from run_ccpd_investigation import evaluation_protocol
    existing = json.loads((output / 'analysis/existing_evidence.json').read_text())
    failures = json.loads((output / 'analysis/failure_summary.json').read_text())
    per_seed, differences, groups, training = [], [], {}, []
    for mode in MODES:
        for seed in range(2):
            directory = output / 'training' / mode / f'seed_{seed}' / 'train'
            rows = [r for r in read_rows(directory / 'metrics.jsonl') if r.get('kind') == 'learning']
            training.append(dict(method=mode, training_seed=seed, summary=json.loads((directory / 'summary.json').read_text()),
                                 selected_actions=sum(r.get('ccpd_selected_samples', 0) for r in rows),
                                 **{k: float(np.mean([r.get(k, 0) for r in rows])) for k in DIAGNOSTICS}))
            for scenario in SCENARIOS:
                path = output / 'evaluation' / mode / f'seed_{seed}' / f'last_{scenario}.jsonl'
                rows = read_rows(path)
                seeds, horizon = evaluation_protocol(scenario)
                if [r['seed'] for r in rows] != seeds or any(r['training_seed'] != seed or r['method'] != mode
                        or r['scenario'] != scenario or r['steps'] != horizon for r in rows):
                    raise ValueError(f'Incomplete or mislabeled pilot group: {path}')
                groups[mode, seed, scenario] = rows
                per_seed.append(dict(method=mode, training_seed=seed, scenario=scenario, **summarize_episode_group(rows)))
    for scenario in SCENARIOS:
        success = matrix([groups['successful', s, scenario] for s in range(2)])
        for mode in ('off', 'random', 'random_action_matched'):
            comparator = matrix([groups[mode, s, scenario] for s in range(2)])
            differences.append(dict(scenario=scenario, comparator=mode, metrics=paired_bootstrap(success, comparator)))
    decision = screening(per_seed)
    selection = []
    for method in ('ccpd', 'mappo', 'random', 'all_conflict'):
        for seed in range(3):
            value = json.loads((output / 'selection' / method / f'seed_{seed}.json').read_text())
            if not value['parameters_unchanged'] or len(value['records']) != 16:
                raise ValueError('Incomplete selection audit')
            selection.append(dict(method=method, training_seed=seed, **value))
    summary = dict(complete=True, existing_20m=existing, pilot_10m=dict(per_seed=per_seed, differences=differences, training=training),
                   failure_replays=failures, selection_audits=selection, decision=decision,
                   uncertainty='10,000 paired hierarchical bootstrap draws (training seeds then episode seeds), pointwise 95% intervals. Only two pilot training seeds; descriptive screening only.',
                   reserved_seeds='3000–3049 unused',
                   selection_caveat='Same-rollout action quotas match exactly. Independently trained policies generate different trajectories and successful pools.')
    write_json(output / 'summary.json', summary)
    lines = ['# CCPD investigation and comparison', '',
             'The existing 20M campaign and new 10M pilot are separate experiments. Only final checkpoints are used for the pilot. All evaluation scenarios here are development diagnostics.', '',
             '## Existing 20M results — three training seeds', '',
             *mean_table([r for r in existing['comparison']['per_seed'] if r['checkpoint_kind'] == 'last']), '',
             '## Matched 10M pilot — two training seeds', '', *mean_table(per_seed), '',
             'Cells show equal-weight training-seed means, except maximum unfinished age. Duration statistics include completed cycles only. All individual training seeds and per-robot values are retained in summary.json.', '',
             '## Successful selection minus control', '', summary['uncertainty'], '',
             '| Scenario | Comparator | Throughput difference [95% CI] | Conflict difference [95% CI] |', '|---|---|---:|---:|']
    for d in differences:
        values = [d['metrics'][k] for k in ('cycles_per_1000_steps', 'conflicts_per_1000_steps')]
        lines.append(f"| {d['scenario']} | {d['comparator']} | " + ' | '.join(f"{v['difference']:.3f} [{v['lower']:.3f}, {v['upper']:.3f}]" for v in values) + ' |')
    lines += ['', '## Long-run starvation screening', '',
              '| Mode | Training seed | Fraction of robot episode endpoints with cycle age ≥500 |', '|---|---:|---:|']
    for r in per_seed:
        if r['scenario'] == 'long_run':
            lines.append(f"| {r['method']} | {r['training_seed']} | {r['starvation_fraction']:.3f} |")
    lines += ['', '## Failure replay', '',
              'Cases were selected from adverse CCPD episodes, then replayed with every method. These cases explain behavior; their frequency is not an unbiased estimate of failure probability. Repetition is observable state/action repetition, not proof of repeated GRU state.', '',
              '| Method | Seed | Scenario / episode | Max cycle interval | Repetition windows | Other-robot completion steps during starvation |', '|---|---:|---|---:|---:|---:|']
    for r in failures:
        lines.append(f"| {r['method']} | {r['training_seed']} | {r['scenario']} / {r['episode_seed']} | {max(r['longest_cycle_interval'])} | {sum(r['repetition_windows'])} | {sum(r['other_robot_completion_steps_while_age_ge_500'])} |")
    lines += ['', '## Same-rollout selection audit', '',
              '| Checkpoint family | Seed | Selection mode | Selected actions | NOOP | FORWARD | LEFT | RIGHT |', '|---|---:|---|---:|---:|---:|---:|---:|']
    for entry in selection:
        for mode in ('successful', 'random', 'all_conflict', 'random_action_matched'):
            records = [r for r in entry['records'] if r['mode'] == mode]
            counts = [sum(r['breakdown']['action'].get(str(a), 0) for r in records) for a in range(4)]
            lines.append(f"| {entry['method']} | {entry['training_seed']} | {mode} | {sum(counts)} | " + ' | '.join(map(str, counts)) + ' |')
    lines += ['', summary['selection_caveat'], '',
              'Learning histories: [validation curves](analysis/learning_curves.csv), [auxiliary curves](analysis/auxiliary_curves.csv), and [evidence notes](analysis/evidence.md). Team advantage and event labels do not establish individual causal credit.', '',
              '## Decision', '', decision['conclusion'], '', 'See [next steps](next_steps.md) for the screening evidence and recommendation. No additional full training is launched automatically.']
    (output / 'comparison.md').write_text('\n'.join(lines)+'\n')
    advice = {
        'promote_full_budget_action_matched_comparison': 'The pilot passes the predeclared screen. Recommend a separate 20M-step, three-seed action-matched comparison before claiming a successful-event contribution.',
        'action_mix_is_a_plausible_explanation': 'The screen passes against random selection but not against action-matched random selection. Investigate action mix as a plausible explanation; this does not prove equivalence or causality.',
        'inconclusive_or_adverse_pilot': 'The pilot does not pass the predeclared screen. Examine the failed comparisons below and the replay traces. Treat seed disagreement or immature learning as inconclusive; do not extend the training budget automatically.'}
    next_lines = ['# Recommended next steps', '', advice[decision['conclusion']], '',
                  '| Seed | Comparator | Fewer conflicts | Throughput within 2% | Starvation not increased |', '|---:|---|---|---|---|']
    for r in decision['evidence']:
        next_lines.append(f"| {r['training_seed']} | {r['comparator']} | {r['fewer_conflicts']} | {r['throughput_within_tolerance']} | {r['starvation_not_increased']} |")
    next_lines += ['', 'The gate uses original-map conflicts and throughput in both seeds, and the fraction of long-run robot endpoints with unfinished-cycle age ≥500. The 2% allowance is a pilot screening tolerance, not a noninferiority claim.', '',
                   'Two seeds at 10M steps cannot establish publication-level superiority. Interval overlap is not equivalence. Preserve unseen layouts and seeds 3000–3049 for later final evaluation.']
    (output / 'next_steps.md').write_text('\n'.join(next_lines)+'\n')
    return summary
