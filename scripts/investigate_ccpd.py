"""Read-only CCPD evidence collection and shared investigation file helpers."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/home/utar/seac-rware')
sys.path[:0] = [str(ROOT / 'robotic-warehouse'), str(ROOT / 'seac/seac')]

import numpy as np
import torch
import rware
from evaluate_shared import evaluate_actor, load_actor
from shared_ccpd import select_samples
from shared_envs import SharedEnvs
from shared_ppo import SharedPPO
from train_shared import DEFAULTS
from validate_ccpd import layouts, write_json

METHODS = ('ccpd', 'mappo', 'shared_ppo', 'random', 'all_conflict')
MODES = ('off', 'successful', 'random', 'random_action_matched')
SELECTION_MODES = ('successful', 'random', 'all_conflict', 'random_action_matched')
SCENARIOS = ('reference', 'horizontal', 'vertical', 'rotation', 'long_run')


def assert_isolation():
    expected = ROOT / 'robotic-warehouse/rware/__init__.py'
    if Path(rware.__file__).resolve() != expected:
        raise RuntimeError(f'Wrong simulator imported: {rware.__file__}; expected {expected}')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    # Full artifacts only: truncation is an error, not a silently missing episode.
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def original_train(prior, method, seed):
    if method in ('random', 'all_conflict'):
        return prior / 'training' / method / f'seed_{seed}' / 'train'
    runs = json.loads((prior / 'audit.json').read_text())['runs']
    return Path(next(r['directory'] for r in runs if r['method'] == method and r['training_seed'] == seed))


def episode_path(prior, method, seed, scenario):
    return prior / 'evaluation' / method / f'seed_{seed}' / f'last_{scenario}.jsonl'


def cached(path, inputs):
    path = Path(path)
    receipt = path.with_name(path.name + '.done.json')
    if path.with_name(path.name + '.partial').exists():
        raise ValueError(f'Interrupted artifact requires explicit recovery: {path}.partial')
    if not path.exists() and not receipt.exists():
        return False
    if not path.exists() or not receipt.exists():
        raise ValueError(f'Incomplete artifact requires explicit recovery: {path}')
    saved = json.loads(receipt.read_text())
    if saved != dict(inputs=inputs, sha256=sha(path)):
        raise ValueError(f'Incompatible or modified artifact: {path}')
    return True


def finish(path, inputs):
    temporary = path.with_name(path.name + '.partial')
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)
    write_json(path.with_name(path.name + '.done.json'), dict(inputs=inputs, sha256=sha(path)))


def save_artifact(path, value, inputs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + '.partial').open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    finish(path, inputs)


def select_cases(prior):
    cases = set()
    for training_seed in range(3):
        rows = read_rows(episode_path(prior, 'ccpd', training_seed, 'long_run'))
        worst_age = sorted(rows, key=lambda r: (-r['max_unfinished_task_age'], r['seed']))[:2]
        worst_rate = sorted(rows, key=lambda r: (r['cycles_per_1000_steps'], r['seed']))[:2]
        for row in worst_age + worst_rate:
            cases.add((training_seed, 'long_run', row['seed']))
        for scenario in ('horizontal', 'vertical', 'rotation'):
            rows = read_rows(episode_path(prior, 'ccpd', training_seed, scenario))
            row = min(rows, key=lambda r: (r['cycles_per_1000_steps'], r['seed']))
            cases.add((training_seed, scenario, row['seed']))
    return [dict(training_seed=s, scenario=c, episode_seed=e) for s, c, e in sorted(cases)]


def check_replay(actual, archived):
    for key, value in actual.items():
        if key == 'inference_ms_per_fleet_step':
            continue
        if key not in archived:
            raise ValueError(f'Archived episode missing {key}')
        if value is None or isinstance(value, (bool, str)):
            equal = value == archived[key]
        else:
            equal = np.allclose(value, archived[key], rtol=1e-10, atol=1e-10)
        if not equal:
            raise ValueError(f'Replay differs from archived episode: {key}')


def replay_case(checkpoint, archived, trace_path):
    inputs = dict(checkpoint_sha256=sha(checkpoint), scenario=archived['scenario'], seed=archived['seed'],
                  archived_episode_sha256=hashlib.sha256(json.dumps(archived, sort_keys=True).encode()).hexdigest())
    summary_path = trace_path.with_name(trace_path.name + '.summary.json')
    if cached(trace_path, inputs):
        if not cached(summary_path, inputs):
            raise ValueError(f'Missing replay summary: {summary_path}')
        return json.loads(summary_path.read_text())
    actor, saved = load_actor(checkpoint)
    logits = []
    hook = actor.head.register_forward_hook(lambda module, args, result: logits.append(result.detach()))
    scenario = archived['scenario']
    layout = layouts().get(scenario) if scenario not in ('reference', 'long_run') else None
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    streaks = {k: np.zeros(5, dtype=int) for k in ('no_translation', 'no_progress', 'no_phase_progress')}
    longest = {k: np.zeros(5, dtype=int) for k in streaks}
    histories = [[] for _ in range(5)]
    repetition = np.zeros(5, dtype=int)
    since_completion = np.zeros(5, dtype=int)
    max_since_completion = np.zeros(5, dtype=int)
    unfair_steps = np.zeros(5, dtype=int)
    try:
        with gzip.open(trace_path.with_name(trace_path.name + '.partial'), 'xt') as handle:
            def record(t, warehouse, action, mask, info):
                trace = info['coordination']
                after_phase = np.array([task.phase.value if task else -1 for task in warehouse.task_manager.tasks])
                completed = np.asarray(info['completed_cycles'])
                before_completion = since_completion.copy() + 1
                max_since_completion[:] = np.maximum(max_since_completion, before_completion)
                since_completion[:] = np.where(completed > 0, 0, before_completion)
                conditions = dict(no_translation=np.asarray(trace['movement_success']) == 0,
                                  no_progress=np.asarray(trace['progress']) <= 0,
                                  no_phase_progress=after_phase == trace['task_phase'])
                for key, condition in conditions.items():
                    streaks[key][:] = np.where(condition, streaks[key] + 1, 0)
                    longest[key][:] = np.maximum(longest[key], streaks[key])
                for robot in range(5):
                    if not mask[robot]:
                        histories[robot].clear()
                        continue
                    signature = (*np.asarray(trace['position'][robot]).tolist(), int(trace['orientation'][robot]),
                                 int(trace['task_phase'][robot]), *np.asarray(trace['target'][robot]).tolist(), int(action[robot]))
                    histories[robot].append(signature)
                    histories[robot] = histories[robot][-32:]
                    h = histories[robot]
                    if len(h) == 32 and any(all(h[i] == h[i-period] for i in range(period, 32)) for period in range(1, 9)):
                        repetition[robot] += 1
                unfair_steps[:] += ((since_completion >= 500) & ((completed.sum() - completed) > 0)).astype(int)
                values = {k: np.asarray(v).tolist() for k, v in trace.items()}
                values.update(step=t, requested_action=action.tolist(), eligible=mask.tolist(),
                              action_probabilities=torch.softmax(logits.pop(), -1).cpu().tolist(),
                              unfinished_cycle_ages=warehouse._cycle_steps.tolist(),
                              completed_cycles=completed.tolist(), phase_after=after_phase.tolist())
                handle.write(json.dumps(values, allow_nan=False) + '\n')
            actual, = evaluate_actor(actor, saved['config']['env_name'], [archived['seed']], archived['requested_steps'],
                                     archived['continuous'], True, layout=layout, trace_callback=record)
        check_replay(actual, archived)
        if sha(checkpoint) != inputs['checkpoint_sha256'] or not all(
                torch.equal(value, saved['actor'][key]) for key, value in actor.state_dict().items()):
            raise ValueError('Actor parameters changed during replay')
        result = dict(method=archived['method'], training_seed=archived['training_seed'], scenario=scenario,
                      episode_seed=archived['seed'], replay_verified=True, parameters_unchanged=True,
                      metrics=actual, longest_streaks={k: v.tolist() for k, v in longest.items()},
                      longest_cycle_interval=max_since_completion.tolist(), repetition_windows=repetition.tolist(),
                      other_robot_completion_steps_while_age_ge_500=unfair_steps.tolist(), trace=str(trace_path))
        finish(trace_path, inputs)
        save_artifact(summary_path, result, inputs)
        return result
    finally:
        hook.remove()


def replay_failures(prior, output, log):
    cases = select_cases(prior)
    write_json(output / 'analysis/failure_cases.json', dict(selection='Adverse CCPD cases; diagnostic, not an unbiased sample', cases=cases))
    results = []
    for case in cases:
        for method in METHODS:
            seed, scenario, episode_seed = case['training_seed'], case['scenario'], case['episode_seed']
            archived = next(r for r in read_rows(episode_path(prior, method, seed, scenario)) if r['seed'] == episode_seed)
            path = output / 'traces' / method / f'seed_{seed}' / f'{scenario}_{episode_seed}.jsonl.gz'
            results.append(replay_case(original_train(prior, method, seed) / 'last.pt', archived, path))
            log(f'Replay verified: {method} seed {seed}, {scenario}, episode {episode_seed}')
    write_json(output / 'analysis/failure_summary.json', results)
    return results


def selection_breakdown(data, weights, events):
    trace = data['coordination']
    selected = weights.cpu().numpy().reshape(data['actions'].shape) > 0
    shape = selected.shape
    outcome = np.full(shape, 'outside_event', dtype='<U20')
    quality = np.full(shape, np.nan)
    for event in events:
        idx = (slice(event['start'], event['core_end'] + 1), event['env'], event['robot'])
        outcome[idx] = event['outcome']
        if event['quality'] is not None:
            quality[idx] = event['quality']
    result = {}
    for name, values in dict(action=data['actions'].cpu().numpy(), robot=trace['agent_id'],
                             phase=trace['task_phase'], density=trace['local_robot_count'], outcome=outcome).items():
        unique, count = np.unique(values[selected], return_counts=True)
        result[name] = {str(k): int(v) for k, v in zip(unique, count)}
    q = quality[selected & np.isfinite(quality)]
    result['known_event_quality_mean'] = float(q.mean()) if len(q) else None
    result['selected_with_known_quality'] = len(q)
    result['selected_total'] = int(selected.sum())
    return result


def audit_checkpoint(checkpoint, target):
    inputs = dict(checkpoint_sha256=sha(checkpoint), rollouts=4, environments=8, rollout_steps=256, seed=2000)
    if cached(target, inputs):
        return json.loads(target.read_text())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = dict(DEFAULTS, **saved['config'])
    config.update(ccpd_mode='successful', seed=2000, device='cpu', num_envs=8, rollout_steps=256)
    envs = SharedEnvs(config['env_name'], 8, 2000, config['time_limit'], coordination_trace=True)
    records = []
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2000)
            a = saved['architecture']
            learner = SharedPPO(a['obs_size'], a['actions'], a['state_size'], config, torch.device('cpu'))
            learner.actor.load_state_dict(saved['actor'])
            learner.critic.load_state_dict(saved['critic'])
            for rollout in range(4):
                data, _, _ = learner.collect(envs)
                selections = {}
                for mode in SELECTION_MODES:
                    weights, metrics, events = select_samples(data, dict(config, ccpd_mode=mode), rollout)
                    selections[mode] = weights
                    records.append(dict(rollout=rollout, mode=mode, diagnostics=metrics,
                                        breakdown=selection_breakdown(data, weights, events)))
                success = selections['successful'].cpu().numpy()
                matched = selections['random_action_matched'].cpu().numpy()
                actions = data['actions'].cpu().numpy().reshape(-1)
                if not np.array_equal(np.bincount(actions[success > 0], minlength=4), np.bincount(actions[matched > 0], minlength=4)):
                    raise ValueError('Action quotas differ')
                if not np.array_equal(np.sort(success[success > 0]), np.sort(matched[matched > 0])):
                    raise ValueError('Weight multisets differ')
            for name in ('actor', 'critic'):
                if not all(torch.equal(value, saved[name][key]) for key, value in getattr(learner, name).state_dict().items()):
                    raise ValueError('Read-only selection audit changed parameters')
        if sha(checkpoint) != inputs['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during selection audit')
    finally:
        envs.close()
    result = dict(checkpoint=str(checkpoint), parameters_unchanged=True, records=records)
    save_artifact(target, result, inputs)
    return result


def audit_selections(prior, output, log):
    for method in ('ccpd', 'mappo', 'random', 'all_conflict'):
        for seed in range(3):
            audit_checkpoint(original_train(prior, method, seed) / 'last.pt', output / 'selection' / method / f'seed_{seed}.json')
            log(f'Selection audit: {method} seed {seed}')
