"""Diagnostic protocol checks: historical parity, maps and paired uncertainty."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from validate_ccpd import layouts, validate_layout, normalized_simulator, git_source
from summarize_ccpd_validation import aggregate, paired_bootstrap, FIELDS
from evaluate_shared import evaluate_actor
from shared_models import Actor
from shared_envs import SharedEnvs
from shared_ppo import SharedPPO
from train_shared import DEFAULTS


def historical_module(tmp_path, filename):
    path = tmp_path / filename
    path.write_bytes(git_source('2e8d522', 'seac/seac/' + filename))
    spec = importlib.util.spec_from_file_location('historical_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_evaluation_parity_and_recurrent_reset(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    actor = Actor(199, 4, recurrent=True)
    weights = {k: v.clone() for k, v in actor.state_dict().items()}
    rng = torch.get_rng_state().clone()
    old = historical_module(tmp_path, 'evaluate_shared.py').evaluate_actor
    expected = old(actor, DEFAULTS['env_name'], [2000, 2001], 25)
    actual = evaluate_actor(actor, DEFAULTS['env_name'], [2000, 2001], 25)
    single = evaluate_actor(actor, DEFAULTS['env_name'], [2001], 25)[0]
    for a, b in zip(expected, actual):
        for key in a:
            if key != 'inference_ms_per_fleet_step':
                assert a[key] == b[key]
    for key in single:
        if key != 'inference_ms_per_fleet_step':
            assert single[key] == actual[1][key]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(weights[k], v) for k, v in actor.state_dict().items())
    continuous = evaluate_actor(actor, DEFAULTS['env_name'], [2000], 501, continuous=True)[0]
    assert continuous['steps'] == 501 and continuous['continuous']
    with pytest.raises(ValueError, match='incompatible'):
        evaluate_actor(Actor(79, 4), DEFAULTS['env_name'], [2000], 1)
    with pytest.raises(ValueError, match='incompatible'):
        evaluate_actor(Actor(199, 5), DEFAULTS['env_name'], [2000], 1)


def test_layout_validity_and_actor_compatibility():
    torch.manual_seed(31)
    actor = Actor(199, 4, recurrent=True)
    variants = layouts()
    assert len(set(variants.values())) == 4
    original = variants['reference'].strip().splitlines()
    assert variants['horizontal'].strip().splitlines() == [r[::-1] for r in original]
    assert variants['vertical'].strip().splitlines() == original[::-1]
    assert variants['rotation'].strip().splitlines() == [r[::-1] for r in original[::-1]]
    for value in variants.values():
        assert validate_layout(value)['all_routes_valid']
        assert evaluate_actor(actor, DEFAULTS['env_name'], [2000], 4, layout=value)[0]['steps'] == 4
    with pytest.raises(ValueError):
        validate_layout('...\n')


def test_historical_disabled_learner_parity(tmp_path):
    old = historical_module(tmp_path, 'shared_ppo.py').SharedPPO
    results = []
    c = dict(DEFAULTS, method='mappo', recurrent=True, ccpd_mode='off',
             num_envs=1, rollout_steps=8, ppo_epochs=2, num_minibatches=2, sequence_length=3, burn_in=2)
    for cls in (old, SharedPPO):
        torch.manual_seed(8)
        envs = SharedEnvs(c['env_name'], 1, 5, time_limit=7)
        try:
            learner = cls(199, 4, 320, c, torch.device('cpu'))
            for _ in range(3):
                data, _, _ = learner.collect(envs)
                learner.update(data)
            results.append(([p.detach().clone() for m in (learner.actor, learner.critic) for p in m.parameters()],
                            torch.get_rng_state().clone(), envs.obs.copy()))
        finally:
            envs.close()
    assert all(torch.equal(a, b) for a, b in zip(results[0][0], results[1][0]))
    assert torch.equal(results[0][1], results[1][1])
    np.testing.assert_array_equal(results[0][2], results[1][2])
    assert normalized_simulator(git_source('2e8d522', 'robotic-warehouse/rware/warehouse.py')) == normalized_simulator(
        (ROOT / 'robotic-warehouse/rware/warehouse.py').read_bytes())


def test_aggregation_zero_cycles_and_paired_bootstrap():
    row = dict(steps=500, completed_cycles=0, conflict_attempts=10, movement_denied=20,
               wait_steps=25, navigation_decisions=1000, team_stall_events=1, deliveries=0, pickups=0,
               cycle_durations=[], cycles_per_robot=[0]*5, unfinished_task_ages=[500]*5,
               max_unfinished_task_age=500, zero_cycle_fraction=1, unfinished_tasks=5)
    result = aggregate([row, row])
    assert result['cycles_per_1000_steps'] == 0
    assert result['conflicts_per_1000_steps'] == 20
    assert result['conflicts_per_navigation'] == .01
    assert result['mean_cycle_time'] is None and result['p95_cycle_time'] is None
    array = np.array([[[row[k] for k in FIELDS]]*4]*3, dtype=float)
    identical = paired_bootstrap(array, array)
    assert all(v == dict(difference=0., lower=0., upper=0.) for v in identical.values())
    right = array.copy()
    right[..., 1] = 5
    difference = paired_bootstrap(array, right)['cycles_per_1000_steps']
    assert difference == dict(difference=-10., lower=-10., upper=-10.)
    # Seed effects must survive episode resampling; more episodes are not more runs.
    right[1, :, 1], right[2, :, 1] = 10, 15
    interval = paired_bootstrap(array, right)['cycles_per_1000_steps']
    assert interval['difference'] == -20 and interval['lower'] < -20 < interval['upper']
    # Duration mean is completion-weighted, not the average of episode means.
    a = dict(row, cycle_durations=[10], completed_cycles=1)
    b = dict(row, cycle_durations=[20, 30, 40], completed_cycles=3)
    assert aggregate([a, b])['mean_cycle_time'] == 25


def test_control_schedule_uses_matched_configs_and_one_job_per_gpu(tmp_path, monkeypatch):
    import json
    import validate_ccpd as protocol
    import run_shared_baselines as launcher
    waves = []
    monkeypatch.setattr(launcher, 'prepare_jobs', lambda jobs, gpus, busy, python: None)
    monkeypatch.setattr(launcher, 'run_wave', lambda jobs: waves.append(jobs) or 0)
    protocol.train_controls(tmp_path)
    assert len(waves) == 3
    template = json.loads((protocol.training_dir('ccpd', 0, tmp_path) / 'provenance.json').read_text())['config']
    for seed, jobs in enumerate(waves):
        assert [j['physical_gpu'] for j in jobs] == ['0', '1']
        for job, mode in zip(jobs, ('random', 'all_conflict')):
            config = json.loads(Path(job['command'][-1]).read_text())
            assert config == dict(template, seed=seed, ccpd_mode=mode,
                                  run_dir=str(tmp_path / 'training' / mode / f'seed_{seed}' / 'train'), device='cuda:0')
            assert config['resume'] is None and config['num_env_steps'] == 20_000_000


def test_evaluation_cache_rejects_partial_or_changed_protocol(tmp_path, monkeypatch):
    import json
    import evaluate_shared
    import validate_ccpd as protocol
    directory = tmp_path / 'checkpoint'
    directory.mkdir()
    actor = Actor(199, 4, recurrent=True)
    torch.save(dict(actor=actor.state_dict(), architecture=dict(obs_size=199, actions=4, recurrent=True),
                    config=dict(env_name=DEFAULTS['env_name'], seed=0), env_steps=20_000_768), directory / 'last.pt')
    calls = []
    def fake(actor, env_name, seeds, horizon, continuous, deterministic, layout, metadata):
        calls.append(1)
        return [dict(metadata, seed=s, requested_steps=horizon, steps=horizon,
                     continuous=continuous, deterministic=deterministic) for s in seeds]
    monkeypatch.setattr(evaluate_shared, 'evaluate_actor', fake)
    job = ('ccpd', 0, 'last', 'reference', directory, tmp_path)
    path = Path(protocol.evaluation_job(job))
    protocol.evaluation_job(job)
    assert len(calls) == 1
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert rows[0]['environment']['conflict_cost'] == .1
    assert rows[0]['environment']['sensor_range'] == 2
    assert rows[0]['training_seed'] == 0 and rows[0]['checkpoint_env_steps'] == 20_000_768
    # Legacy actor-only exports intentionally have no recorded step counter.
    saved = torch.load(directory / 'last.pt', weights_only=False)
    saved.pop('env_steps')
    torch.save(saved, directory / 'actor.pt')
    assert evaluate_shared.checkpoint_metadata(directory / 'actor.pt', saved, 'cpu')['checkpoint_env_steps'] is None
    path.write_text(json.dumps(rows[0]) + '\n')
    with pytest.raises(ValueError, match='incompatible provenance'):
        protocol.evaluation_job(job)
