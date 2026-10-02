"""Stage 4 budgets, progress, conversion, restart, holdout and report contracts."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_stage4 as runner
import stage4_evaluation as evaluation
from navigation_envs import NavigationEpisode, STAGE4_RESERVED_SEEDS, stream_seed
from navigation_policy import NavigationActor, PolicyRuntime, export_actor
from navigation_train import load_checkpoint

torch.set_num_threads(1)


def args_for(path, schedule=((1, 2), (10, 2))):
    return NS(output=path, schedule=schedule, smoke=True, source_checkpoint=None,
              condition='local', seed=0, device='cpu', resume=False, stop_after=None, agents=None)


def test_budget_and_selection():
    assert sum(u * 2048 for _, u in runner.SCHEDULE) == 4001792
    assert tuple(runner.CONDITIONS) == ('local', 'communicating_ccpd')
    assert runner.SEEDS == (0, 1)
    assert sum(u * 2048 for _, u in runner.SCHEDULE) * len(runner.CONDITIONS) * len(runner.SEEDS) == 16007168
    for n, u in runner.SCHEDULE:
        config = runner.config_for('local', 1, n, u, 'cuda:1')
        assert config['num_envs'] == 8 and config['rollout_steps'] == 256
        assert config['seed'] == 1 and config['num_env_steps'] == u * 2048
    good = dict(progress_passed=True, failure_episode_fraction=0., throughput=50., p95_unfinished_age=100.)
    bad = dict(good, progress_passed=False, failure_episode_fraction=.1, throughput=500.)
    assert runner.checkpoint_rank(good, 98) > runner.checkpoint_rank(bad, 49)
    assert runner.checkpoint_rank(good, 49) > runner.checkpoint_rank(good, 98)
    assert runner.checkpoint_rank(dict(bad, failure_episode_fraction=.05), 98) > runner.checkpoint_rank(bad, 49)
    assert runner.checkpoint_rank(dict(good, throughput=51.), 98) > runner.checkpoint_rank(good, 49)


def test_all_four_conversions_and_actor_exports(tmp_path):
    args = args_for(tmp_path)
    layout = runner.load_layout()
    study = runner.ensure_study(args, layout)
    projections = {}
    for method in runner.CONDITIONS:
        for seed in runner.SEEDS:
            source = runner.source_path(method, seed)
            original_hash = runner.sha(source)
            path = runner.conversion(tmp_path, study, method, seed, layout)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            old = torch.load(source, map_location='cpu', weights_only=False)
            assert saved['config']['seed'] == seed
            assert saved['config']['condition'] == method
            assert 'actor_optimizer' not in saved and saved['optimizers'] == 'fresh'
            for name in saved['transferred']:
                torch.testing.assert_close(saved['actor'][name], old['actor'][name], rtol=0, atol=0)
            assert saved['pretraining_joint_steps'] == 20000768
            projections[method, seed] = saved['actor']['grid.5.weight']
            actor, _ = load_checkpoint(path)
            exported = tmp_path / f'{method}_{seed}_actor.pt'
            export_actor(actor, exported)
            PolicyRuntime(exported)
            assert runner.sha(source) == original_hash
            assert runner.conversion(tmp_path, study, method, seed, layout) == path
    assert not torch.equal(projections['local', 0], projections['local', 1])
    torch.testing.assert_close(projections['local', 1], projections['communicating_ccpd', 1])
    altered = copy.deepcopy(study)
    altered['initial_checkpoints']['local/seed_1'] = study['initial_checkpoints']['local/seed_0']
    with pytest.raises(ValueError, match='condition/training seed'):
        runner.conversion(tmp_path / 'wrong', altered, 'local', 1, layout)


def test_recovered_individual_stall_does_not_hide_in_productive_fleet(monkeypatch):
    layout = runner.load_layout()
    env = NavigationEpisode(layout, 100005, 110005, 100, layout['shape'], 2)
    original = env.env.step
    def recovered(actions):
        obs, rewards, term, trunc, info = original(actions)
        info['completed_cycles'][0] = 1
        info['cycle_time'][0] = 500
        env.w._cycle_steps[0] = 0
        return obs, rewards, term, trunc, info
    monkeypatch.setattr(env.env, 'step', recovered)
    try:
        env.step(np.zeros(2, dtype=int))
        row = env.metrics()
        assert row['unfinished_task_ages'][0] == 0
        assert row['max_task_ages'][0] == 500
        assert row['individual_progress_failure'] and row['progress_failure']
        assert not row['fleet_failure']
        assert row['robot_failure_fraction'] == .5
        result = runner.progress_summary([row])
        assert not result['progress_passed'] and result['failure_episode_fraction'] == 1
        monkeypatch.setattr(env.env, 'step', original)
        env.step(np.zeros(2, dtype=int))
        assert env.metrics()['max_task_ages'][0] == 500
    finally:
        env.close()


def test_holdout_and_unfinished_isolated_tasks(tmp_path):
    layout = runner.load_layout()
    for seed in (300000, 300049, 300100, 310000, 310050, stream_seed(300000, 0, 0xC4)):
        assert seed in STAGE4_RESERVED_SEEDS
        with pytest.raises(ValueError, match='Reserved Stage 4'):
            NavigationEpisode(layout, seed, 17, 32, layout['shape'], 1)
        with pytest.raises(ValueError, match='Reserved Stage 4'):
            NavigationEpisode(layout, 17, seed, 32, layout['shape'], 1)
    actor = NavigationActor(False)
    with torch.no_grad():
        actor.head.weight.zero_()
        actor.head.bias.fill_(-100)
        actor.head.bias[0] = 100
    study = runner.ensure_study(args_for(tmp_path), layout)
    result = runner.isolated(actor, layout, study)
    assert not result['passed'] and result['completion_fraction'] == 0
    assert all(row['steps'] == study['isolated']['steps'] for row in result['rows'])
    assert not set(evaluation.eval_seeds(study, 2)) & STAGE4_RESERVED_SEEDS
    with pytest.raises(FileNotFoundError):
        evaluation.require_lock(tmp_path, study)


def test_resume_and_fixed_budget_despite_learning_result(tmp_path, monkeypatch):
    arguments = ['--train', '--smoke', '--schedule', '1:2,10:2', '--condition', 'local', '--seed', '1',
                 '--device', 'cpu', '--output', str(tmp_path)]
    original = runner.validation
    def always_failing(*args):
        value = original(*args)
        value.update(progress_passed=False, failure_episode_fraction=1.)
        return value
    monkeypatch.setattr(runner, 'validation', always_failing)
    runner.main(arguments + ['--stop-after', '1'])
    first = runner.stage_path(tmp_path, 'local', 1, 1)
    assert not (first / 'summary.json').exists()
    # Simulate a crash after writing an uncommitted log and a torn final append.
    with (first / 'metrics.jsonl').open('a') as file:
        file.write('{"update": 200}\n{"update":')
    with pytest.raises(ValueError, match='requires --resume'):
        runner.main(arguments)
    runner.main(arguments + ['--resume'])
    final = runner.stage_path(tmp_path, 'local', 1, 10)
    for folder in (first, final):
        result = runner.read_json(folder / 'summary.json')
        assert result['completed_updates'] == 2 and not result['progress_passed']
        assert [r['update'] for r in runner.read_rows(folder / 'metrics.jsonl')] == [1, 2]
        assert (folder / 'actor_best.pt').exists() and (folder / 'actor_last.pt').exists()
        saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['config']['seed'] == 1 and saved['updates'] == 2
    assert runner.read_json(final / 'summary.json')['cumulative_env_steps'] == 256
    assert len(runner.read_rows(first / 'restarts.jsonl')) == 1
    initialized = torch.load(final / 'checkpoints/0.pt', map_location='cpu', weights_only=False)
    assert initialized['actor_optimizer']['state'] == {}
    before = runner.sha(final / 'last.pt')
    runner.main(arguments + ['--resume'])
    assert runner.sha(final / 'last.pt') == before
    with pytest.raises(ValueError, match='Frozen protocol'):
        runner.main([*arguments, '--resume', '--schedule', '1:3,10:2'])


def fake_finished_study(tmp_path):
    args = args_for(tmp_path, ((1, 1), (40, 1), (50, 1)))
    study = runner.ensure_study(args, runner.load_layout())
    for method in runner.CONDITIONS:
        for seed in runner.SEEDS:
            total = 0
            for agents, updates in study['schedule']:
                folder = runner.stage_path(tmp_path, method, seed, agents)
                folder.mkdir(parents=True)
                protocol = dict(study_sha256=runner.fingerprint(study), method=method, seed=seed, agents=agents)
                runner.write_json(folder / 'protocol.json', protocol)
                artifacts = {}
                for name in ('best.pt', 'last.pt', 'actor_best.pt', 'actor_last.pt'):
                    (folder / name).write_bytes(f'{method}:{seed}:{agents}:{name}'.encode())
                    artifacts[name] = runner.sha(folder / name)
                total += updates * study['joint_steps_per_update']
                runner.write_json(folder / 'summary.json', dict(status='complete', condition=method, seed=seed,
                    agents=agents, completed_updates=updates, cumulative_env_steps=total, smoke=True,
                    protocol_sha256=runner.fingerprint(protocol), artifacts=artifacts, progress_passed=True))
    return study


def test_locked_report_requires_complete_receipts_and_matches_seeds(tmp_path):
    study = fake_finished_study(tmp_path)
    lock = evaluation.lock_evaluation(tmp_path, study)
    assert evaluation.require_lock(tmp_path, study) == lock
    layout = runner.load_layout()
    env = NavigationEpisode(layout, 100500, 110500, 32, layout['shape'], 50)
    try:
        env.step(np.zeros(50, dtype=int))
        base = env.metrics()
    finally:
        env.close()
    for path, _, inputs in evaluation.evaluation_files(tmp_path, study, lock):
        spec, n = inputs['spec'], inputs['agents']
        path.parent.mkdir(parents=True, exist_ok=True)
        for seed in inputs['seeds']:
            row = dict(base, **{k: inputs[k] for k in ('method', 'training_seed', 'agents', 'checkpoint_kind',
                                   'suite', 'action_replicate', 'checkpoint_sha256')})
            row.update(episode_id=seed, seed=seed+(100 if spec['profile']=='changed_starts' else 0),
                       task_seed=seed+(10050 if spec['profile']=='changed_tasks' else 10000),
                       steps=spec['steps'], requested_steps=spec['steps'], deterministic=spec['deterministic'],
                       profile=spec['profile'], delay=spec['delay'], geometry_sha256=inputs['map_sha256'],
                       inputs_sha256=runner.fingerprint(inputs),
                       packet_loss=spec['loss'] if runner.CONDITIONS[inputs['method']][0] else None,
                       completed_cycles=100, cycles_per_1000_steps=100000/spec['steps'],
                       cycles_per_robot=[2]*n, cycle_durations=[10]*100,
                       max_task_ages=[10]*n, unfinished_task_ages=[1]*n,
                       longest_stationary=[1]*n, repetition_windows=[0]*n,
                       max_unfinished_task_age=1, navigation_decisions=n*spec['steps'],
                       inference_ms_per_fleet_step=0., unfinished_tasks=n)
            runner.append(path, row)
        runner.write_json(path.with_name(path.name+'.done.json'), dict(inputs=inputs, sha256=runner.sha(path)))
        assert evaluation.receipt_valid(path, inputs)
    result = evaluation.report(tmp_path, study)
    assert result['complete']
    summary = runner.read_json(tmp_path / 'summary.json')
    assert len(summary['paired_comparisons']) == 2
    assert all(r['candidate'] == 'communicating_ccpd' and r['baseline'] == 'local'
               for r in summary['paired_comparisons'])
    assert not any(d['recommended'] for d in summary['decisions'])
    bad = runner.read_rows(path)
    bad[0]['training_seed'] = 999
    path.write_text(''.join(json.dumps(r)+'\n' for r in bad))
    runner.write_json(path.with_name(path.name+'.done.json'), dict(inputs=inputs, sha256=runner.sha(path)))
    with pytest.raises(ValueError, match='Invalid evaluation receipt'):
        evaluation.report(tmp_path, study)
    path.unlink()
    assert not evaluation.report(tmp_path, study)['complete']
    checkpoint = next(iter(lock['checkpoints']))
    (tmp_path / checkpoint).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Locked artifact'):
        evaluation.require_lock(tmp_path, study)


def test_evaluation_resumes_partial_episode_files_without_opening_holdout(tmp_path, monkeypatch):
    study = fake_finished_study(tmp_path)
    lock = evaluation.lock_evaluation(tmp_path, study)
    layout = runner.load_layout()
    env = NavigationEpisode(layout, 100500, 110500, 32, layout['shape'], 1)
    try:
        env.step(np.zeros(1, dtype=int))
        base = env.metrics()
    finally:
        env.close()
    calls = []
    interrupt = [True]
    monkeypatch.setattr(evaluation, 'load_checkpoint', lambda path: (path.parts[-4], {}))
    def fake_evaluate(actor, layout, seeds, steps, **kwargs):
        if interrupt[0] and len(calls) == 3:
            raise RuntimeError('Simulated interrupted evaluation')
        seed, n, profile = seeds[0], kwargs['n_agents'], kwargs['profile']
        calls.append(seed)
        assert not kwargs['allow_stage4_holdout']
        assert seed not in STAGE4_RESERVED_SEEDS
        row = dict(base, episode_id=seed, seed=seed+(100 if profile == 'changed_starts' else 0),
                   task_seed=seed+(10050 if profile == 'changed_tasks' else 10000),
                   steps=steps, requested_steps=steps, action_replicate=kwargs['replicate'],
                   deterministic=kwargs['deterministic'], profile=profile,
                   packet_loss=kwargs['loss'] if runner.CONDITIONS[actor][0] else None,
                   cycles_per_robot=[0]*n, max_task_ages=[steps]*n, unfinished_task_ages=[steps]*n,
                   longest_stationary=[steps]*n, repetition_windows=[0]*n,
                   inference_ms_per_fleet_step=0.)
        return [row]
    monkeypatch.setattr(evaluation, 'evaluate', fake_evaluate)
    with pytest.raises(RuntimeError, match='Simulated interrupted'):
        evaluation.evaluate_campaign(tmp_path, study)
    assert len(list((tmp_path/'evaluation').rglob('*.done.json'))) == 1
    assert len(list((tmp_path/'evaluation').rglob('*.partial'))) == 1
    interrupt[0] = False
    evaluation.evaluate_campaign(tmp_path, study)
    specs = list(evaluation.evaluation_files(tmp_path, study, lock))
    assert len(calls) == sum(len(inputs['seeds']) for _, _, inputs in specs)
    assert all(evaluation.receipt_valid(path, inputs) for path, _, inputs in specs)
    previous_calls = len(calls)
    evaluation.evaluate_campaign(tmp_path, study)
    assert len(calls) == previous_calls


def test_production_training_cannot_skip_gpu_preflight(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='requires --profile-all'):
        runner.main(['--train', '--condition', 'local', '--seed', '0', '--output', str(tmp_path)])
    assert not list(tmp_path.rglob('last.pt'))
    args = args_for(tmp_path)
    args.smoke, args.skip_preflight = False, True
    calls = []
    monkeypatch.setattr(runner, 'train_stage', lambda *values: calls.append(values[3]) or {'status': 'complete'})
    assert runner.train(args, dict(schedule=args.schedule), runner.load_layout())['status'] == 'complete'
    assert calls == [1, 10]


def test_runtime_repair_preserves_protocol_and_rejects_unrecorded_changes(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    layout = runner.load_layout()
    study = runner.ensure_study(args, layout)
    protocol_hash = runner.sha(tmp_path / 'protocol.json')
    changed = dict(study['sources'], **{'seac/seac/navigation_policy.py': 'corrected-runtime'})
    monkeypatch.setattr(runner, 'source_hashes', lambda: changed)
    with pytest.raises(ValueError, match='runtime repair receipt'):
        runner.verify_study(tmp_path)
    runner.write_json(tmp_path / 'runtime_repair.json', dict(study_sha256=runner.fingerprint(study),
                      sources_before=study['sources'], sources_after=changed, reason='Verified numerical correction'))
    assert runner.ensure_study(args, layout) == study
    assert runner.sha(tmp_path / 'protocol.json') == protocol_hash
    evaluation.report(tmp_path, study)
    assert runner.read_json(tmp_path / 'summary.json')['runtime_repair']['sources_after'] == changed
    changed = dict(changed, unexpected='unreviewed')
    with pytest.raises(ValueError, match='runtime repair receipt'):
        runner.verify_study(tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA batch-shape regression')
@pytest.mark.parametrize('communication', [False, True])
def test_cuda_actor_recurrence_is_independent_of_batch_partition(communication):
    from navigation_policy import LOCAL_SIZE, PACKET_SIZE
    device = runner.setup_device('cuda:0', 17)
    actor = NavigationActor(communication).to(device)
    width = LOCAL_SIZE + (3 * (PACKET_SIZE + 1) if communication else 0)
    hidden = actor.initial_state((133,), device)
    partitioned = hidden.clone()
    with torch.no_grad():
        for _ in range(8):
            obs = torch.randn(133, width, device=device)
            reset = torch.rand(133, device=device) < .1
            logits, hidden = actor(obs, hidden, reset)
            parts = [actor(obs[i:i+37], partitioned[i:i+37], reset[i:i+37]) for i in range(0, 133, 37)]
            partitioned = torch.cat([state for _, state in parts])
            torch.testing.assert_close(logits, torch.cat([value for value, _ in parts]), rtol=0, atol=0)
            torch.testing.assert_close(hidden, partitioned, rtol=0, atol=0)


def test_joint_preflight_propagates_verified_runtime_repair(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    args.devices = ['cuda:0', 'cuda:1']
    study = runner.ensure_study(args, runner.load_layout())
    patched = dict(study['sources'], **{'seac/seac/navigation_policy.py': 'patched'})
    repair = dict(study_sha256=runner.fingerprint(study), sources_before=study['sources'], sources_after=patched)
    runner.write_json(tmp_path / 'runtime_repair.json', repair)
    monkeypatch.setattr(runner, 'source_hashes', lambda: patched)
    monkeypatch.setattr(runner, 'host_available', lambda: 4 * 1024**3)
    monkeypatch.setattr(runner, 'conversion', lambda *unused: None)
    profile = dict(rows=[dict(cycle_seconds=1.)], gpu=dict(headroom_fraction=.5))
    for method, device in zip(runner.CONDITIONS, args.devices):
        path = runner.profile_path(tmp_path, method, 50, device)
        path.parent.mkdir(exist_ok=True)
        runner.write_json(path, profile)
    def spawn(command):
        output = Path(command[command.index('--output')+1])
        assert runner.verify_study(output) == study
        assert runner.read_json(output / 'runtime_repair.json') == repair
        method, device = (command[command.index(flag)+1] for flag in ('--condition', '--device'))
        path = runner.profile_path(output, method, 50, device)
        path.parent.mkdir(exist_ok=True)
        runner.write_json(path, profile)
        return NS(wait=lambda: 0)
    monkeypatch.setattr(runner.subprocess, 'Popen', spawn)
    runner.joint_profile(args, study)
    result = runner.read_json(tmp_path / 'concurrency.json')
    assert result['passed'] and result['assignments'] == dict(zip(runner.CONDITIONS, args.devices))


@pytest.mark.parametrize('skip_preflight', [False, True])
def test_campaign_assigns_two_methods_and_two_seeds_to_distinct_gpus(tmp_path, monkeypatch, skip_preflight):
    args = args_for(tmp_path)
    args.jobs, args.devices = 2, ['cuda:0', 'cuda:1']
    args.skip_preflight = skip_preflight
    study = runner.ensure_study(args, runner.load_layout())
    commands, checks = [], []
    monkeypatch.setattr(runner, 'profile_all', lambda *unused: checks.append('single'))
    monkeypatch.setattr(runner, 'joint_profile', lambda *unused: checks.append('joint'))
    def spawn(command, **unused):
        assert checks == ([] if skip_preflight else ['single', 'joint'])
        commands.append(command)
        return NS(wait=lambda: 0)
    monkeypatch.setattr(runner.subprocess, 'Popen', spawn)
    monkeypatch.setattr(evaluation, 'lock_evaluation', lambda *unused: None)
    monkeypatch.setattr(evaluation, 'evaluate_campaign', lambda *unused: None)
    monkeypatch.setattr(evaluation, 'report', lambda *unused: {'complete': True})
    assert runner.campaign(args, study)['complete']
    actual = [(c[c.index('--condition')+1], c[c.index('--seed')+1], c[c.index('--device')+1])
              for c in commands]
    assert actual == [('local', '0', 'cuda:0'), ('communicating_ccpd', '0', 'cuda:1'),
                      ('local', '1', 'cuda:0'), ('communicating_ccpd', '1', 'cuda:1')]
    assert all('--resume' in c for c in commands)
    assert all(('--skip-preflight' in c) == skip_preflight for c in commands)
    assert (tmp_path / 'preflight_overrides.jsonl').exists() == skip_preflight
    with pytest.raises(SystemExit):
        runner.main(['--prepare', '--seed', '2', '--output', str(tmp_path)])
    with pytest.raises(SystemExit):
        runner.main(['--prepare', '--condition', 'local_ccpd', '--output', str(tmp_path)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA replay regression requires a GPU')
def test_fifty_robot_cuda_replay_uses_consistent_precision(tmp_path):
    layout = runner.load_layout()
    study = runner.ensure_study(args_for(tmp_path), layout)
    path = runner.conversion(tmp_path, study, 'communicating_ccpd', 0, layout)
    initial = torch.load(path, map_location='cpu', weights_only=False)
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = True
    config = runner.config_for('communicating_ccpd', 0, 50, 2, 'cuda:0', True)
    envs, learner = runner.make_learner(layout, config, initial)
    try:
        for _ in range(2):
            metrics = runner.checked_update(learner, envs)
            assert metrics['pre_update_log_prob_error'] <= 2e-5
    finally:
        envs.close()
        torch.backends.cudnn.allow_tf32 = previous
