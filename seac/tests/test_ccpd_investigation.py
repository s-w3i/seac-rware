"""Action-matched selection, replay fidelity, and investigation orchestration."""
import copy
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'robotic-warehouse'), str(ROOT / 'scripts')]
import investigate_ccpd as evidence
import run_ccpd_investigation as runner
import report_ccpd_investigation as report
from shared_ccpd import select_samples
from test_shared_ccpd import example, config

torch.set_num_threads(1)


def selection_data():
    data = example()
    trace = data['coordination']
    for name in ('robot_blocked', 'conflict_attempts', 'movement_denied'):
        trace[name][1, :, :] = True
    trace['movement_success'][1, :, :] = False
    for robot in range(5):
        trace['progress'][5:13, :, robot] = .2 + .1*robot
    data['actions'] = torch.arange(48*5).reshape(48, 1, 5) % 4
    data['advantages'][10] = -1
    data['eligible'][12] = False
    return data


def test_action_matched_exact_counts_weights_and_rng():
    data = selection_data()
    np.random.seed(123)
    torch.manual_seed(123)
    old_np, old_torch = np.random.get_state(), torch.get_rng_state().clone()
    success, _, _ = select_samples(data, config(ccpd_mode='successful'), 2)
    matched, metrics, _ = select_samples(data, config(ccpd_mode='random_action_matched'), 2)
    actions = data['actions'].flatten().numpy()
    left, right = success.numpy(), matched.numpy()
    assert np.count_nonzero(right) > 0
    np.testing.assert_array_equal(np.bincount(actions[left > 0], minlength=4), np.bincount(actions[right > 0], minlength=4))
    np.testing.assert_array_equal(np.sort(left[left > 0]), np.sort(right[right > 0]))
    selected = np.flatnonzero(right)
    assert len(set(selected)) == len(selected)
    assert data['eligible'].flatten()[selected].all()
    assert not data['coordination']['movement_denied'].reshape(-1)[selected].any()
    adv = np.broadcast_to(data['advantages'].numpy()[..., None], data['actions'].shape).reshape(-1)
    assert (adv[selected] > 0).all()
    assert metrics['ccpd_selected_noop_fraction'] <= .25
    assert torch.equal(old_torch, torch.get_rng_state())
    current_np = np.random.get_state()
    assert old_np[0] == current_np[0] and old_np[2:] == current_np[2:]
    np.testing.assert_array_equal(old_np[1], current_np[1])
    empty = copy.deepcopy(data)
    empty['advantages'].fill_(-1)
    weights, diagnostics, _ = select_samples(empty, config(ccpd_mode='random_action_matched'), 2)
    assert not weights.any() and diagnostics['ccpd_selected_samples'] == 0


@pytest.mark.parametrize('mode', ['off', 'successful', 'random', 'all_conflict'])
def test_existing_selector_is_identical(tmp_path, mode):
    path = tmp_path / 'legacy.py'
    path.write_bytes(subprocess.check_output(['git', 'show', 'b52f1e4:seac/seac/shared_ccpd.py'], cwd=ROOT))
    spec = importlib.util.spec_from_file_location('legacy_ccpd', path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    data = selection_data()
    old = legacy.select_samples(data, config(ccpd_mode=mode), 7)
    new = select_samples(data, config(ccpd_mode=mode), 7)
    torch.testing.assert_close(old[0], new[0], rtol=0, atol=0)
    assert old[1:] == new[1:]


def test_replay_matches_real_archived_episode(tmp_path):
    prior = evidence.ORIGINAL / 'results/ccpd_diagnostic'
    archived = evidence.read_rows(evidence.episode_path(prior, 'ccpd', 0, 'reference'))[0]
    checkpoint = evidence.original_train(prior, 'ccpd', 0) / 'last.pt'
    trace = tmp_path / 'trace.jsonl.gz'
    result = evidence.replay_case(checkpoint, archived, trace)
    assert result['replay_verified'] and result['parameters_unchanged']
    with gzip.open(trace, 'rt') as handle:
        records = [json.loads(s) for s in handle]
    assert len(records) == archived['steps']
    np.testing.assert_allclose(np.sum(records[0]['action_probabilities'], axis=1), 1, atol=1e-6)
    assert records[-1]['unfinished_cycle_ages'] == archived['unfinished_task_ages']
    assert evidence.replay_case(checkpoint, archived, trace) == result
    broken = dict(archived, completed_cycles=archived['completed_cycles'] + 1)
    with pytest.raises(ValueError, match='Replay differs'):
        evidence.replay_case(checkpoint, broken, tmp_path / 'bad.jsonl.gz')


def test_case_selection_ties_and_deduplication(tmp_path):
    for seed in range(3):
        for scenario in ('long_run', 'horizontal', 'vertical', 'rotation'):
            path = evidence.episode_path(tmp_path, 'ccpd', seed, scenario)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(dict(seed=s, max_unfinished_task_age=500, cycles_per_1000_steps=0))+'\n' for s in (2002, 2000, 2001)))
    cases = evidence.select_cases(tmp_path)
    assert len(cases) == 15  # two deduplicated long runs + three layouts per training seed
    assert {r['episode_seed'] for r in cases if r['scenario'] == 'long_run'} == {2000, 2001}
    assert {r['episode_seed'] for r in cases if r['scenario'] != 'long_run'} == {2000}


def test_prerequisite_active_complete_failed_and_incomplete(tmp_path):
    lock = tmp_path / 'pipeline.lock'
    lock.touch()
    def state(value):
        (tmp_path / 'status.json').write_text(json.dumps(dict(stage=value)))
    state('ablation_training')
    with lock.open('r') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert runner.prerequisite_state(tmp_path) == 'active'
    with pytest.raises(RuntimeError, match='stopped incomplete'):
        runner.prerequisite_state(tmp_path)
    state('failed')
    with pytest.raises(RuntimeError, match='Prerequisite failed'):
        runner.prerequisite_state(tmp_path)
    state('complete')
    with pytest.raises(RuntimeError, match='report'):
        runner.prerequisite_state(tmp_path)
    (tmp_path / 'summary.json').write_text('{"complete": true}')
    (tmp_path / 'comparison.md').write_text('complete')
    assert runner.prerequisite_state(tmp_path) == 'complete'


def test_artifact_integrity_and_partial_rejection(tmp_path):
    path = tmp_path / 'data.json'
    inputs = dict(seed=2000)
    assert not evidence.cached(path, inputs)
    evidence.save_artifact(path, {'value': 1}, inputs)
    assert evidence.cached(path, inputs)
    with pytest.raises(ValueError, match='Incompatible'):
        evidence.cached(path, dict(seed=2001))
    path.write_text('{}')
    with pytest.raises(ValueError, match='modified'):
        evidence.cached(path, inputs)
    path.with_name(path.name + '.partial').touch()
    with pytest.raises(ValueError, match='Interrupted'):
        evidence.cached(path, inputs)


def test_free_gpu_selection_respects_busy_and_inherited_visibility(monkeypatch):
    def query(command, **kwargs):
        return '0, GPU-A\n1, GPU-B\n' if '--query-gpu=index,uuid' in command else 'GPU-A, 100\n'
    monkeypatch.setattr(runner.subprocess, 'check_output', query)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert runner.free_gpus() == [('1', 'GPU-B')]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    assert runner.free_gpus() == []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(RuntimeError, match='No CUDA'):
        runner.free_gpus()


def test_training_schedule_single_gpu_and_exact_configs(tmp_path, monkeypatch):
    prior = evidence.ORIGINAL / 'results/ccpd_diagnostic'
    monkeypatch.setattr(runner, 'free_gpus', lambda: [('1', 'GPU-B')])
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs: None)
    waves = []
    monkeypatch.setattr(runner, 'run_wave', lambda jobs: waves.append(jobs) or 0)
    monkeypatch.setattr(runner, 'validate_training', lambda *args, **kwargs: None)
    runner.train_pilot(prior, tmp_path, lambda *args: None, lambda: None)
    assert len(waves) == 8 and all(len(w) == 1 for w in waves)
    configs = [json.loads(Path(w[0]['command'][-1]).read_text()) for w in waves]
    assert {(c['ccpd_mode'], c['seed']) for c in configs} == {(m, s) for m in runner.MODES for s in range(2)}
    assert sum(c['num_env_steps'] for c in configs) == 80_000_000
    assert all(c['resume'] is None and c['ccpd_coef'] == .01 for c in configs)
    assert all(w[0]['environment']['CUDA_VISIBLE_DEVICES'] == 'GPU-B' for w in waves)
    assert all(str(ROOT / 'robotic-warehouse') in w[0]['environment']['PYTHONPATH'] for w in waves)
    with pytest.raises(ValueError, match='Incomplete training'):
        # Invoke the real validator, not the scheduling test double.
        import importlib
        fresh = importlib.util.spec_from_file_location('other_runner', ROOT / 'scripts/run_ccpd_investigation.py')
        module = importlib.util.module_from_spec(fresh)
        fresh.loader.exec_module(module)
        module.validate_training(tmp_path / 'missing/train', configs[0])


def test_pipeline_order_and_evaluation_seeds(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(report, 'analyze_existing', lambda *args: order.append('existing_evidence'))
    monkeypatch.setattr(runner, 'replay_failures', lambda *args: order.append('failure_replay'))
    monkeypatch.setattr(runner, 'audit_selections', lambda *args: order.append('selection_audit'))
    monkeypatch.setattr(runner, 'train_pilot', lambda *args: order.append('pilot_training'))
    monkeypatch.setattr(runner, 'evaluate_pilot', lambda *args: order.append('pilot_evaluation'))
    monkeypatch.setattr(report, 'generate_report', lambda *args: order.append('report'))
    runner.pipeline(tmp_path, tmp_path, lambda *args: None, lambda: None)
    assert order == list(runner.STAGES)
    for scenario, count, horizon in [('reference', 50, 500), ('rotation', 20, 500), ('long_run', 10, 10000)]:
        seeds, steps = runner.evaluation_protocol(scenario)
        assert seeds == list(range(2000, 2000+count)) and steps == horizon
        assert not set(seeds) & set(range(3000, 3050))
    evidence.assert_isolation()


def screening_rows():
    return [dict(method=m, training_seed=s, scenario=scenario,
                 conflicts_per_1000_steps=8 if m == 'successful' else 10,
                 cycles_per_1000_steps=98 if m == 'successful' else 100,
                 starvation_fraction=0)
            for m in runner.MODES for s in range(2) for scenario in ('reference', 'long_run')]


def test_screening_gate_requires_every_comparison():
    rows = screening_rows()
    assert report.screening(rows)['conclusion'] == 'promote_full_budget_action_matched_comparison'
    next(r for r in rows if r['method'] == 'random_action_matched' and r['scenario'] == 'reference')['conflicts_per_1000_steps'] = 8
    assert report.screening(rows)['conclusion'] == 'action_mix_is_a_plausible_explanation'
    next(r for r in rows if r['method'] == 'successful' and r['scenario'] == 'long_run')['starvation_fraction'] = .02
    assert report.screening(rows)['conclusion'] == 'inconclusive_or_adverse_pilot'


def episode(method, seed, scenario, episode_seed):
    horizon = runner.evaluation_protocol(scenario)[1]
    return dict(method=method, training_seed=seed, scenario=scenario, seed=episode_seed, steps=horizon,
                completed_cycles=5, conflict_attempts=1, movement_denied=1, wait_steps=2, navigation_decisions=100,
                team_stall_events=0, pickups=5, deliveries=5, cycle_durations=[10]*5, cycles_per_robot=[1]*5,
                unfinished_task_ages=[1]*5, max_unfinished_task_age=1, zero_cycle_fraction=0, unfinished_tasks=5)


def test_final_report_requires_complete_groups_and_writes_decision(tmp_path):
    (tmp_path / 'analysis').mkdir()
    (tmp_path / 'analysis/existing_evidence.json').write_text(json.dumps(dict(comparison=dict(per_seed=[]))))
    (tmp_path / 'analysis/failure_summary.json').write_text('[]')
    for method in ('ccpd', 'mappo', 'random', 'all_conflict'):
        for seed in range(3):
            path = tmp_path / 'selection' / method / f'seed_{seed}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(parameters_unchanged=True, records=[dict(mode=m, breakdown=dict(action={})) for m in evidence.SELECTION_MODES for _ in range(4)])))
    for mode in runner.MODES:
        for seed in range(2):
            directory = tmp_path / 'training' / mode / f'seed_{seed}' / 'train'
            directory.mkdir(parents=True)
            (directory / 'summary.json').write_text('{"env_steps": 10000384}')
            (directory / 'metrics.jsonl').write_text('{"kind": "learning"}\n')
            for scenario in runner.SCENARIOS:
                path = tmp_path / 'evaluation' / mode / f'seed_{seed}' / f'last_{scenario}.jsonl'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(''.join(json.dumps(episode(mode, seed, scenario, e))+'\n' for e in runner.evaluation_protocol(scenario)[0]))
    summary = report.generate_report(tmp_path, tmp_path)
    assert summary['complete'] and len(summary['pilot_10m']['per_seed']) == 40
    assert (tmp_path / 'comparison.md').exists() and (tmp_path / 'next_steps.md').exists()
    assert 'inconclusive' in summary['decision']['conclusion']  # equal conflict counts never pass
    path = tmp_path / 'evaluation/off/seed_0/last_reference.jsonl'
    path.write_text(path.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError, match='Incomplete'):
        report.generate_report(tmp_path, tmp_path)


def test_completed_training_requires_receipt_and_rejects_changed_weights(tmp_path):
    from shared_models import Actor
    import shutil
    c = runner.pilot_config(config(), tmp_path, 'successful', 0)
    directory = Path(c['run_dir'])
    (directory / 'source').mkdir(parents=True)
    for name in ('shared_ccpd.py', 'shared_ppo.py', 'shared_models.py', 'shared_envs.py', 'evaluate_shared.py', 'train_shared.py', 'shared_storage.py', 'warehouse.py'):
        current = ROOT / ('robotic-warehouse/rware' if name == 'warehouse.py' else 'seac/seac') / name
        shutil.copy2(current, directory / 'source' / name)
    (directory / 'summary.json').write_text(json.dumps(dict(env_steps=10000384)))
    (directory / 'provenance.json').write_text(json.dumps(dict(config=c)))
    (directory.parent / 'launch.json').write_text('{"exit_status": 0}')
    for name in ('metrics.jsonl', 'evaluation.jsonl'):
        (directory / name).write_text('{}\n')
    saved = dict(config=c, architecture=dict(obs_size=199, actions=4, state_size=320, recurrent=True),
                 env_steps=10000384, actor=Actor(199, 4, recurrent=True).state_dict())
    torch.save(saved, directory / 'last.pt')
    with pytest.raises(ValueError, match='Unverified training'):
        runner.validate_training(directory, c)
    first = runner.validate_training(directory, c, record=True)
    assert runner.validate_training(directory, c) == first
    saved['actor']['head.bias'].add_(.1)
    torch.save(saved, directory / 'last.pt')
    with pytest.raises(ValueError, match='artifacts changed'):
        runner.validate_training(directory, c)
