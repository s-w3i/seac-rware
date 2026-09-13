"""Algorithm/data-flow regression checks for all four shared baselines."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

from evaluate_shared import evaluate_actor, load_actor
from shared_envs import SharedEnvs, centralized_state, decision_mask
from shared_models import Actor, Network
from shared_ppo import SharedPPO
from shared_storage import batches, gae, replay, retain_history
from train_shared import DEFAULTS, atomic_save, checkpoint, restore, validate
from rware.warehouse import Action, Direction, TaskPhase

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)


def config(**overrides):
    return dict(DEFAULTS, **overrides)


def test_gae_bootstrap_trace_and_local_broadcast():
    rewards = torch.tensor([[1.], [2.], [3.]])
    values = torch.tensor([[10.], [20.], [30.]])
    next_values = torch.tensor([[20.], [100.], [40.]])
    term = torch.tensor([[False], [False], [True]])
    trunc = torch.tensor([[False], [True], [False]])
    adv, returns = gae(rewards, values, next_values, term, trunc, .5, 1.)
    torch.testing.assert_close(adv[:, 0], torch.tensor([17., 32., -27.]))
    torch.testing.assert_close(returns[:, 0], torch.tensor([27., 52., 3.]))
    a, r = gae(rewards, values[..., None].expand(3, 1, 2),
               next_values[..., None].expand(3, 1, 2), term, trunc, .5, 1.)
    torch.testing.assert_close(a[..., 0], adv)
    torch.testing.assert_close(r[..., 1], returns)
    a, _ = gae(rewards[:1], values[:1], next_values[:1], term[:1], trunc[:1], .5, 1.)
    assert a.item() == 1.  # Rollout boundary retains bootstrap.


def test_time_limits_and_before_reset_states():
    envs = SharedEnvs(DEFAULTS['env_name'], 1, 7, time_limit=1)
    try:
        before = envs.obs.copy()
        transition = envs.step(np.zeros((1, 5), dtype=int))
        assert transition.truncated.tolist() == [True]
        assert not transition.terminated.any()
        assert not np.array_equal(transition.final_obs, transition.next_obs)
        assert not np.array_equal(transition.final_state, transition.next_state)
        assert before.shape == transition.final_obs.shape
    finally:
        envs.close()
    for limits, expected in ((dict(max_steps=1), (False, True)),
                             (dict(max_steps=None, max_inactivity_steps=1), (True, False)),
                             (dict(max_steps=1, max_inactivity_steps=1), (True, True))):
        env = gym.make(DEFAULTS['env_name'], **limits)
        try:
            env.reset(seed=1)
            _, _, terminated, truncated, _ = env.step([0] * 5)
            assert (terminated, truncated) == expected
        finally:
            env.close()


def test_state_encoder_is_read_only_and_seeded_transitions_unchanged():
    a, b = [SharedEnvs(DEFAULTS['env_name'], 1, 42) for _ in range(2)]
    try:
        w = a.envs[0].unwrapped
        rng = copy.deepcopy(w.np_random.bit_generator.state)
        encoded = centralized_state(w)
        assert encoded.shape == (3 * 60 + 5 * 28,)
        assert np.isfinite(encoded).all()
        for _ in range(4):
            np.testing.assert_array_equal(encoded, centralized_state(w))
        assert rng == w.np_random.bit_generator.state
        for actions in ([1] * 5, [2] * 5, [3] * 5):
            left, right = a.step([actions]), b.step([actions])
            np.testing.assert_array_equal(left.next_obs, right.next_obs)
            np.testing.assert_array_equal(left.rewards, right.rewards)
            np.testing.assert_array_equal(left.next_state, right.next_state)
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize('method,recurrent', [('shared_ppo', False), ('shared_ppo', True), ('mappo', False), ('mappo', True)])
def test_four_variants_likelihoods_updates_exports_resume(tmp_path, method, recurrent):
    c = config(method=method, recurrent=recurrent, rollout_steps=8, num_envs=2,
               num_minibatches=2, ppo_epochs=2, sequence_length=3, burn_in=2)
    envs = SharedEnvs(c['env_name'], 2, 10, time_limit=3)
    try:
        learner = SharedPPO(199, 4, envs.state.shape[-1], c, torch.device('cpu'))
        assert not ({id(p) for p in learner.actor.parameters()} & {id(p) for p in learner.critic.parameters()})
        data, _, _ = learner.collect(envs)
        diagnostics = learner.policy_diagnostics(data)
        assert diagnostics['max_log_prob_error'] < 1e-5
        expected = (8, 2) if method == 'mappo' else (8, 2, 5)
        assert data['returns'].shape == expected
        original = [p.clone() for p in learner.actor.parameters()]
        critic = [p.clone() for p in learner.critic.parameters()]
        metrics = learner.update(data)
        assert all(np.isfinite(v) for v in metrics.values())
        assert any(not torch.equal(a, b) for a, b in zip(original, learner.actor.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(critic, learner.critic.parameters()))
        architecture = dict(obs_size=199, actions=4, state_size=envs.state.shape[-1], recurrent=recurrent)
        path = tmp_path / 'last.pt'
        atomic_save(checkpoint(learner, c, architecture, 16, 1, 0.), path)
        actor, _ = load_actor(path)
        obs = torch.as_tensor(envs.obs)
        torch.testing.assert_close(actor(obs)[0], learner.actor(obs)[0])
        separate = torch.stack([actor(obs[:, i])[0] for i in range(5)], dim=1)
        torch.testing.assert_close(actor(obs)[0], separate)
        restored = SharedPPO(199, 4, envs.state.shape[-1], c, torch.device('cpu'))
        assert restore(restored, path, c, architecture) == (16, 1, 0.)
        assert restored.reset is None and restored.actor_history is None
        assert restored.actor_optimizer.state_dict()['state']
        torch.testing.assert_close(restored.critic(data['critic_input'] if method == 'mappo' else data['obs'])[0],
                                   learner.critic(data['critic_input'] if method == 'mappo' else data['obs'])[0])
        with pytest.raises(ValueError, match='Incompatible'):
            restore(restored, path, dict(c, gamma=.5), architecture)
    finally:
        envs.close()


def test_service_actions_advance_memory_and_only_critic_updates():
    c = config(recurrent=True, num_envs=1, rollout_steps=2, ppo_epochs=1, num_minibatches=1)
    envs = SharedEnvs(c['env_name'], 1, 0)
    try:
        w = envs.envs[0].unwrapped
        task, robot = w.task_manager.tasks[0], w.agents[0]
        robot.x, robot.y = task.rack_position
        task.phase = TaskPhase.AUTO_PICKUP
        w._recalc_grid()
        assert not decision_mask(w)[0]
        transition = envs.step([[3] * 5])
        assert not transition.decision_mask[0, 0]
        assert task.phase == TaskPhase.DELIVER
        assert robot.carrying_shelf is not None
        learner = SharedPPO(199, 4, envs.state.shape[-1], c, torch.device('cpu'))
        data, _, _ = learner.collect(envs)
        assert not torch.equal(data['actor_states'][0], data['actor_states'][1])
        data['eligible'].zero_()
        actor_before = [p.clone() for p in learner.actor.parameters()]
        critic_before = [p.clone() for p in learner.critic.parameters()]
        metrics = learner.update(data)
        assert metrics['actor_epochs'] == 0 and metrics['entropy'] == 0
        assert all(torch.equal(a, b) for a, b in zip(actor_before, learner.actor.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(critic_before, learner.critic.parameters()))
    finally:
        envs.close()


def test_denied_forward_retains_proposal_and_likelihood():
    c = config(num_envs=1, rollout_steps=1)
    envs = SharedEnvs(c['env_name'], 1, 0)
    try:
        w = envs.envs[0].unwrapped
        w.agents[0].x, w.agents[0].y, w.agents[0].dir = 0, 0, Direction.LEFT
        w._recalc_grid()
        learner = SharedPPO(199, 4, envs.state.shape[-1], c, torch.device('cpu'))
        with torch.no_grad():
            learner.actor.head.weight.zero_()
            learner.actor.head.bias.fill_(-100)
            learner.actor.head.bias[Action.FORWARD.value] = 100
        data, infos, _ = learner.collect(envs)
        assert data['actions'][0, 0, 0] == 1
        assert infos[0]['movement_denied'][0] == 1
        assert learner.policy_diagnostics(data)['max_log_prob_error'] < 1e-6
    finally:
        envs.close()


def test_recurrent_chunk_replay_padding_history_and_resets():
    torch.manual_seed(2)
    model = Network(3, 2, 7, recurrent=True)
    obs = torch.randn(13, 4, 3)
    reset = torch.zeros(13, 4, dtype=torch.bool)
    reset[4, 0] = True
    reset[8, 2] = True
    state = torch.zeros(4, 7)
    states, outputs = [], []
    with torch.no_grad():
        for t in range(13):
            states.append(state)
            output, state = model(obs[t], state, reset[t])
            outputs.append(output)
    states, expected = torch.stack(states), torch.stack(outputs)
    history = retain_history(obs[:5], states[:5], reset[:5], None, 3)
    replayed = torch.empty_like(expected[5:]).flatten(0, 1)
    counts = torch.zeros(8 * 4, dtype=torch.long)
    for indices, batch in batches(obs[5:], states[5:], reset[5:], True, 3, 3, 3, history):
        valid = indices >= 0
        replayed[indices[valid]] = replay(model, batch)[valid]
        counts[indices[valid]] += 1
    assert (counts == 1).all()
    torch.testing.assert_close(replayed, expected[5:].flatten(0, 1))
    changed = states[7].clone()
    changed[0] += 100
    normal, _ = model(obs[7], states[7])
    other, _ = model(obs[7], changed)
    torch.testing.assert_close(normal[1:], other[1:])
    a, _ = model(obs[7], changed, torch.ones(4, dtype=torch.bool))
    b, _ = model(obs[7], torch.zeros_like(changed))
    torch.testing.assert_close(a, b)


def test_evaluation_rng_and_continuous_no_reset():
    actor = Actor(199, 4, recurrent=True)
    before = torch.get_rng_state().clone()
    rows = evaluate_actor(actor, DEFAULTS['env_name'], [1000], steps=501, continuous=True, deterministic=False)
    assert torch.equal(before, torch.get_rng_state())
    assert rows[0]['steps'] == 501 and rows[0]['max_unfinished_task_age'] == 501
    assert len(rows[0]['reward_per_robot']) == 5


def test_collector_truncation_bootstraps_with_final_value():
    c = config(num_envs=1, rollout_steps=1)
    envs = SharedEnvs(c['env_name'], 1, 0, time_limit=1)
    try:
        learner = SharedPPO(199, 4, envs.state.shape[-1], c, torch.device('cpu'))
        captured = []
        original = envs.step
        def capture(actions):
            result = original(actions)
            captured.append(result)
            return result
        envs.step = capture
        data, _, _ = learner.collect(envs)
        with torch.no_grad():
            final = learner.critic(torch.as_tensor(captured[0].final_obs))[0].squeeze(-1)
            reset = learner.critic(torch.as_tensor(captured[0].next_obs))[0].squeeze(-1)
        torch.testing.assert_close(data['bootstrap'][0], final)
        assert not torch.equal(final, reset)
    finally:
        envs.close()


def test_invalid_config():
    for override in (dict(num_envs=0), dict(recurrent='yes'), dict(method='ccpd'), dict(burn_in=-1), dict(actor_lr=float('nan'))):
        with pytest.raises(ValueError):
            validate(config(**override))


def test_kl_stopping_does_not_stop_critic_epochs():
    c = config(num_envs=1, rollout_steps=4, ppo_epochs=3, num_minibatches=1, target_kl=1e-15)
    envs = SharedEnvs(c['env_name'], 1, 0)
    try:
        learner = SharedPPO(199, 4, envs.state.shape[-1], c, torch.device('cpu'))
        data, _, _ = learner.collect(envs)
        metrics = learner.update(data)
        assert metrics['actor_epochs'] == 1 and metrics['critic_epochs'] == 3
        assert int(next(iter(learner.critic_optimizer.state.values()))['step']) == 3
    finally:
        envs.close()


def test_masked_service_rewards_still_enter_gae_and_shared_robot_gradients():
    rewards = torch.tensor([[1.], [10.]])
    zeros = torch.zeros(2, 1)
    done = torch.zeros(2, 1, dtype=torch.bool)
    adv, _ = gae(rewards, zeros, zeros, done, done, 1., 1.)
    assert adv[0].item() == 11  # No decision mask may cut the reward trace.
    actor = Actor(3, 4)
    for robot_obs in (torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 1., 0.]])):
        actor.zero_grad()
        log, _, _ = actor.evaluate_actions(robot_obs, torch.tensor([1]))
        (-log.mean()).backward()
        assert actor.head.weight.grad.abs().sum() > 0


def test_recurrent_final_bootstrap_does_not_advance_live_state_twice():
    c = config(recurrent=True, num_envs=1, rollout_steps=1)
    envs = SharedEnvs(c['env_name'], 1, 0, time_limit=1)
    try:
        learner = SharedPPO(199, 4, envs.state.shape[-1], c, torch.device('cpu'))
        captured = []
        original = envs.step
        def capture(actions):
            result = original(actions)
            captured.append(result)
            return result
        envs.step = capture
        data, _, _ = learner.collect(envs)
        with torch.no_grad():
            _, after = learner.critic(data['obs'][0], data['critic_states'][0], data['resets'][0])
            bootstrap, _ = learner.critic(torch.as_tensor(captured[0].final_obs), after)
        torch.testing.assert_close(data['bootstrap'][0], bootstrap.squeeze(-1))
        torch.testing.assert_close(learner.critic_state, after)
        assert learner.reset.all()
    finally:
        envs.close()


def test_sacred_entrypoint_metadata_resume_and_overwrite(tmp_path):
    first, resumed = tmp_path / 'first', tmp_path / 'resumed'
    cmd = [sys.executable, str(ROOT / 'seac/seac/train_shared.py'), 'with', 'mappo_gru_routing',
           'device=cpu', 'num_envs=1', 'rollout_steps=4', 'ppo_epochs=1', 'num_minibatches=1',
           'eval_steps=2', 'eval_seeds=[1000]']
    subprocess.run(cmd + ['num_env_steps=8', f'run_dir={first}'], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    provenance = json.loads((first / 'provenance.json').read_text())
    assert provenance['actual_layout']['grid_size'] == [10, 6]
    assert len(provenance['map_sha256']) == 64
    assert list((first / 'tensorboard').glob('events.*'))
    subprocess.run(cmd + ['num_env_steps=16', f'run_dir={resumed}', f'resume={first}/last.pt'], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    summary = json.loads((resumed / 'summary.json').read_text())
    assert summary['env_steps'] == 16 and summary['updates'] == 4
    assert json.loads((resumed / 'provenance.json').read_text())['inherited_best']
    assert (first / 'best.pt').read_bytes() == (resumed / 'best.pt').read_bytes()
    again = tmp_path / 'again'
    subprocess.run(cmd + ['num_env_steps=20', f'run_dir={again}', f'resume={resumed}/last.pt'], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert json.loads((again / 'provenance.json').read_text())['inherited_best']
    assert (first / 'best.pt').read_bytes() == (again / 'best.pt').read_bytes()
    result = subprocess.run(cmd + ['num_env_steps=8', f'run_dir={first}'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode != 0 and 'FileExistsError' in result.stdout


def test_cpu_evaluation_does_not_seed_other_cuda_generators(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('CPU evaluation must not reseed CUDA generators')
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', fail)
    rows = evaluate_actor(Actor(199, 4), DEFAULTS['env_name'], [1000], steps=2)
    assert rows[0]['steps'] == 2
