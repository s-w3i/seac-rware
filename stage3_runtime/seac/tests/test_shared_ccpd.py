"""CCPD event labels, isolation from PPO, and recurrent auxiliary integration."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

from rware.warehouse import Action, Direction, TaskPhase
from shared_ccpd import DEFAULTS as CCPD_DEFAULTS, detect_events, select_samples, capped_sample, event_records
from shared_envs import SharedEnvs
from shared_ppo import SharedPPO
from shared_storage import replay
from train_shared import DEFAULTS, validate, checkpoint, restore, atomic_save
from evaluate_shared import load_actor

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)


def config(**overrides):
    return dict(DEFAULTS, method='mappo', recurrent=True, **overrides)


def example(T=48, E=1, N=5):
    shape = (T, E, N)
    trace = dict(robot_blocked=np.zeros(shape, bool), conflict_attempts=np.zeros(shape, bool),
                 movement_denied=np.zeros(shape, bool), movement_success=np.ones(shape, bool),
                 progress=np.ones(shape), progress_valid=np.ones(shape, bool),
                 requested_action=np.ones(shape, int), eligible=np.ones(shape, bool))
    for key in ('robot_blocked', 'conflict_attempts', 'movement_denied'):
        trace[key][1, :, 0] = True
    trace['movement_success'][1, :, 0] = False
    return dict(coordination=trace, actions=torch.ones(shape, dtype=torch.long),
                eligible=torch.ones(shape, dtype=torch.bool), advantages=torch.ones(T, E),
                terminated=torch.zeros(T, E, dtype=torch.bool), truncated=torch.zeros(T, E, dtype=torch.bool))


def events(data, **settings):
    return detect_events(data['coordination'], (data['terminated'] | data['truncated']).numpy(), config(**settings))


def sync(w):
    w._recalc_grid()
    w._distances = np.array([w._static_distance((a.x, a.y), *w._routing_context(i))
                             for i, a in enumerate(w.agents)])


def test_tracing_has_no_simulation_or_rng_effect_and_survives_reset():
    a, b = [SharedEnvs(DEFAULTS['env_name'], 2, 42, time_limit=7, coordination_trace=flag)
            for flag in (False, True)]
    try:
        rng = np.random.default_rng(7)
        for step in range(30):
            action = rng.integers(4, size=(2, 5))
            before = [(e.unwrapped._cur_steps, [(r.x, r.y) for r in e.unwrapped.agents]) for e in b.envs]
            left, right = a.step(action), b.step(action)
            for key in ('next_obs', 'final_obs', 'next_state', 'final_state', 'rewards',
                        'terminated', 'truncated', 'decision_mask'):
                np.testing.assert_array_equal(getattr(left, key), getattr(right, key))
            for i, (x, y) in enumerate(zip(a.envs, b.envs)):
                assert x.unwrapped.np_random.bit_generator.state == y.unwrapped.np_random.bit_generator.state
                trace = right.infos[i]['coordination']
                assert 'coordination' not in left.infos[i]
                np.testing.assert_array_equal(trace['position'], before[i][1])
                np.testing.assert_array_equal(trace['episode_step'], [before[i][0]] * 5)
                np.testing.assert_array_equal(trace['requested_action'], action[i])
                np.testing.assert_array_equal(trace['eligible'], right.decision_mask[i])
                np.testing.assert_allclose(trace['progress'] * .1, right.infos[i]['reward_progress'])
    finally:
        a.close()
        b.close()


def test_trace_service_target_change_wall_and_unreachable_distance():
    env = gym.make(DEFAULTS['env_name'], n_agents=1, coordination_trace_enabled=True)
    try:
        env.reset(seed=5)
        w = env.unwrapped
        robot, task = w.agents[0], w.task_manager.tasks[0]
        robot.x, robot.y = task.rack_position
        task.phase = TaskPhase.AUTO_PICKUP
        sync(w)
        old_target = task.rack_position
        _, _, _, _, info = env.step([Action.RIGHT])
        trace = info['coordination']
        assert not trace['eligible'][0] and trace['progress'][0] == 0
        assert trace['requested_action'][0] == Action.RIGHT.value
        assert trace['distance_before'][0] == trace['distance_after'][0] == 0
        np.testing.assert_array_equal(trace['target'][0], old_target)
        assert task.phase == TaskPhase.DELIVER
        assert w._distances[0] > 0  # The new target distance never enters old-context progress.
        robot.x, robot.y, robot.dir = 0, 0, Direction.LEFT
        sync(w)
        _, _, _, _, info = env.step([1])
        assert info['coordination']['movement_denied'][0]
        assert not info['coordination']['robot_blocked'][0]
        task.workstation_position = (-1, 0)
        sync(w)
        _, _, _, _, info = env.step([0])
        assert not info['coordination']['progress_valid'][0]
        assert info['coordination']['progress'][0] == 0
    finally:
        env.close()


def test_success_and_window_boundaries():
    data = example()
    e, = events(data)
    assert (e['start'], e['core_end'], e['end'], e['duration']) == (1, 4, 12, 4)
    assert e['quality'] == pytest.approx(3.6) and e['outcome'] == 'successful'
    # Complete exactly at termination/truncation is allowed; one missing step is censored.
    for kind in ('terminated', 'truncated'):
        for boundary in (11, 12):
            cut = copy.deepcopy(data)
            cut[kind][boundary, 0] = True
            e, = events(cut)
            assert e['outcome'] == ('censored' if boundary == 11 else 'successful')
    short = example(T=12)
    assert events(short)[0]['outcome'] == 'censored'
    # No detector state carries from the censored rollout into the next.
    empty = example()
    empty['coordination']['robot_blocked'].fill(False)
    assert events(empty) == []


@pytest.mark.parametrize('case,reason', [('wait', 'timeout'), ('blocked', 'timeout'),
    ('no_progress', 'insufficient_progress'), ('invalid', 'invalid_progress'),
    ('recurrence', 'late_conflict'), ('quality', 'low_quality')])
def test_failed_events(case, reason):
    data = example()
    trace = data['coordination']
    if case == 'wait':
        trace['movement_success'][2:, 0, 0] = False
    elif case == 'blocked':
        trace['robot_blocked'][2:33, 0, 0] = True
    elif case == 'no_progress':
        trace['progress'][5:13, 0, 0] = 0
    elif case == 'invalid':
        trace['progress_valid'][6, 0, 0] = False
    elif case == 'recurrence':
        trace['robot_blocked'][12, 0, 0] = True
    elif case == 'quality':
        trace['conflict_attempts'][5:9, 0, 0] = True
    e = events(data)[0]
    assert e['outcome'] == 'failed' and e['reason'] == reason
    if case == 'quality':
        assert e['recurrence'] == 4 and e['quality'] < 0


def test_events_do_not_overlap_and_confirmation_is_not_teaching():
    data = example()
    for key in ('robot_blocked', 'conflict_attempts'):
        data['coordination'][key][7, 0, 0] = True
        data['coordination'][key][20, 0, 0] = True
    es = events(data)
    assert len(es) == 2 and es[0]['end'] < es[1]['start']
    weights, _, _ = select_samples(data, config(ccpd_mode='successful'), 0)
    selected = weights.reshape(data['actions'].shape) > 0
    assert not selected[5:13].any() and not selected[24 + 1:33].any()
    assert es[0]['recurrence'] == 1


def test_selection_masks_raw_advantage_caps_and_matched_controls():
    data = example(T=48, E=2)
    data['eligible'][2, 0, 0] = False
    data['advantages'].fill_(5.)
    data['advantages'][3, 0] = -1
    data['advantages'][4, 0] = 1  # Positive before normalization, negative after it.
    data['actions'][2, 1, 0] = 0
    global_numpy = copy.deepcopy(np.random.get_state())
    global_torch = torch.get_rng_state().clone()
    weights_by_mode = []
    for mode in ('successful', 'all_conflict', 'random'):
        c = config(ccpd_mode=mode)
        weights, metrics, _ = select_samples(data, c, 4)
        again, _, _ = select_samples(data, c, 4)
        torch.testing.assert_close(weights, again, rtol=0, atol=0)
        selected = weights.reshape(data['actions'].shape) > 0
        assert not selected[~data['eligible']].any()
        assert not selected[data['advantages'][..., None].expand_as(selected) <= 0].any()
        assert not selected[torch.as_tensor(data['coordination']['movement_denied'])].any()
        assert metrics['ccpd_selected_noop_fraction'] <= .25
        assert metrics['ccpd_selected_fraction'] <= .10 and weights.max() <= 2
        weights_by_mode.append(sorted(weights[weights > 0].tolist()))
    assert weights_by_mode[0] and weights_by_mode[0] == weights_by_mode[1] == weights_by_mode[2]
    np.testing.assert_array_equal(np.random.get_state()[1], global_numpy[1])
    assert np.random.get_state()[2:] == global_numpy[2:]
    assert torch.equal(global_torch, torch.get_rng_state())
    # Positive raw advantage, even below the rollout mean, remains eligible.
    assert select_samples(data, config(ccpd_mode='successful'), 0)[0].reshape(48, 2, 5)[4, 0, 0] > 0
    actions = np.array([0] * 95 + [1] * 5)
    selected = capped_sample(np.ones(100, bool), actions, 20, .25, np.random.default_rng(1))
    assert len(selected) == 6 and (actions[selected] == 0).sum() <= 1


def test_nonpositive_advantage_and_no_events_produce_zero_samples():
    data = example()
    data['advantages'].zero_()
    weights, metrics, _ = select_samples(data, config(ccpd_mode='successful'), 0)
    assert not weights.any() and metrics['ccpd_selected_samples'] == 0
    data['coordination']['robot_blocked'].fill(False)
    assert select_samples(data, config(ccpd_mode='all_conflict'), 0)[1]['ccpd_events'] == 0


def test_empty_active_selection_matches_plain_ppo_and_trace_limit():
    c = config(ccpd_mode='successful', num_envs=1, rollout_steps=16, ppo_epochs=2,
               num_minibatches=2, sequence_length=5, burn_in=2)
    torch.manual_seed(9)
    envs = SharedEnvs(c['env_name'], 1, 1, coordination_trace=True)
    try:
        learner = SharedPPO(199, 4, envs.state.shape[-1], c, torch.device('cpu'))
        data, _, _ = learner.collect(envs)
        data['coordination']['robot_blocked'].fill(False)
        baseline = copy.deepcopy(learner)
        baseline.config = dict(c, ccpd_mode='off')
        rng = torch.get_rng_state().clone()
        learner.update(data)
        torch.set_rng_state(rng)
        baseline.update(data)
        for model in ('actor', 'critic'):
            assert all(torch.equal(a, b) for a, b in zip(getattr(learner, model).parameters(),
                                                       getattr(baseline, model).parameters()))
        synthetic = example(E=30)
        assert len(list(event_records(synthetic, events(synthetic)))) == 20
    finally:
        envs.close()


def test_disabled_is_identical_over_multiple_updates():
    results = []
    for mode in ('off', 'successful'):
        c = config(ccpd_mode=mode, ccpd_coef=0., num_envs=1, rollout_steps=8,
                   ppo_epochs=2, num_minibatches=2, sequence_length=3, burn_in=2)
        torch.manual_seed(8)
        envs = SharedEnvs(c['env_name'], 1, 5, time_limit=7)
        try:
            learner = SharedPPO(199, 4, envs.state.shape[-1], c, torch.device('cpu'))
            for _ in range(3):
                data, _, _ = learner.collect(envs)
                assert 'coordination' not in data
                learner.update(data)
            results.append(([p.detach().clone() for m in (learner.actor, learner.critic) for p in m.parameters()],
                            torch.get_rng_state().clone(), envs.obs.copy()))
        finally:
            envs.close()
    assert all(torch.equal(a, b) for a, b in zip(results[0][0], results[1][0]))
    assert torch.equal(results[0][1], results[1][1])
    np.testing.assert_array_equal(results[0][2], results[1][2])


def test_auxiliary_gradient_uses_recurrent_context_and_export(tmp_path):
    c = config(ccpd_mode='successful', ccpd_trace_events=True, num_envs=1, rollout_steps=16,
               ppo_epochs=1, num_minibatches=1, sequence_length=5, burn_in=2, entropy_coef=0.)
    torch.manual_seed(9)
    envs = SharedEnvs(c['env_name'], 1, 1, time_limit=8, coordination_trace=True)
    try:
        learner = SharedPPO(199, 4, envs.state.shape[-1], c, torch.device('cpu'))
        data, _, _ = learner.collect(envs)
        # Controlled successful event amid real recurrent inputs. Suppress PPO's
        # advantage/entropy gradient so only the selected-action loss moves actor weights.
        synthetic = example(T=16)
        for key in ('coordination', 'actions', 'eligible', 'advantages', 'terminated', 'truncated'):
            data[key] = synthetic[key]
        with torch.no_grad():
            data['log_probs'] = learner.actor.evaluate_actions(data['obs'], data['actions'],
                data['actor_states'], data['resets'])[0]
        selected, _, _ = select_samples(data, c, 0)
        expected = []
        for indices, batch in learner.minibatches(data, True, shuffle=False):
            valid = indices >= 0
            log = torch.distributions.Categorical(logits=replay(learner.actor, batch)[valid]).log_prob(
                data['actions'].flatten()[indices[valid]])
            w = selected[indices[valid]]
            expected.extend((-w[w > 0] * log[w > 0]).detach().tolist())
        original = [p.detach().clone() for p in learner.actor.parameters()]
        metrics = learner.update(data)
        assert metrics['ccpd_auxiliary_loss'] == pytest.approx(np.mean(expected), rel=1e-5)
        assert metrics['ccpd_auxiliary_loss'] > 0 and metrics['ccpd_selected_samples'] == 3
        assert metrics['actor_main_loss'] == 0.
        assert any(not torch.equal(a, b) for a, b in zip(original, learner.actor.parameters()))
        assert len(learner.ccpd_records) == 1
        json.dumps(learner.ccpd_records, allow_nan=False)
        architecture = dict(obs_size=199, actions=4, state_size=envs.state.shape[-1], recurrent=True)
        path = tmp_path / 'ccpd.pt'
        atomic_save(checkpoint(learner, c, architecture, 16, 1, 0.), path)
        actor, _ = load_actor(path)
        torch.testing.assert_close(actor(data['obs'][0])[0], learner.actor(data['obs'][0])[0])
        restored = SharedPPO(199, 4, envs.state.shape[-1], c, torch.device('cpu'))
        restore(restored, path, c, architecture)
        assert restored.update_number == 1 and restored.actor_history is None
        for key, value in (('ccpd_coef', .02), ('ccpd_confirmation_steps', 9), ('ccpd_mode', 'random')):
            with pytest.raises(ValueError, match='Incompatible'):
                restore(restored, path, dict(c, **{key: value}), architecture)
        # A legacy checkpoint without any CCPD keys still resumes as plain MAPPO.
        saved = torch.load(path, weights_only=False)
        saved['config'] = {k: v for k, v in c.items() if k not in CCPD_DEFAULTS}
        atomic_save(saved, path)
        restore(restored, path, dict(c, **CCPD_DEFAULTS), architecture)
        with pytest.raises(ValueError, match='Incompatible'):
            restore(restored, path, c, architecture)
    finally:
        envs.close()


@pytest.mark.parametrize('override', [dict(ccpd_mode='typo'), dict(ccpd_coef=-1),
    dict(ccpd_coef=float('nan')), dict(ccpd_event_horizon=0), dict(ccpd_clear_steps=9),
    dict(ccpd_confirmation_steps=True), dict(ccpd_max_sample_fraction=1.1), dict(ccpd_trace_events=1)])
def test_invalid_ccpd_config(override):
    with pytest.raises(ValueError):
        validate(config(**override))


def test_named_entrypoint_records_provenance_and_resumes(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    base = [sys.executable, str(ROOT / 'seac/seac/train_shared.py'), 'with', 'mappo_gru_ccpd_routing',
            'device=cpu', 'num_envs=1', 'rollout_steps=8', 'ppo_epochs=1', 'num_minibatches=1',
            'eval_steps=2', 'eval_seeds=[1000]', 'ccpd_trace_events=True']
    for directory, steps, extra in ((first, 16, []), (second, 24, [f'resume={first}/last.pt'])):
        subprocess.run(base + [f'run_dir={directory}', f'num_env_steps={steps}'] + extra, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        provenance = json.loads((directory / 'provenance.json').read_text())
        assert provenance['ccpd_detector_version'] == 'ccpd-v0-1'
        assert (directory / 'source/shared_ccpd.py').exists() and (directory / 'source/warehouse.py').exists()
        row = json.loads((directory / 'metrics.jsonl').read_text().splitlines()[-1])
        assert np.isfinite(row['ccpd_auxiliary_loss'])
    assert json.loads((second / 'summary.json').read_text())['updates'] == 3
