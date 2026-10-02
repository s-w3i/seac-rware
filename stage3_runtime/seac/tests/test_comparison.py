from pathlib import Path
import importlib.util

import gymnasium as gym
import pytest
import torch

from shared_a2c import SharedA2C
from teams import SEACTeam


def load_script(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spaces(agents=2):
    return (
        [gym.spaces.Box(-1, 1, (6,), dtype=float) for _ in range(agents)],
        [gym.spaces.Discrete(4) for _ in range(agents)],
    )


def shared(recurrent=False, agents=2):
    obs, actions = spaces(agents)
    return SharedA2C(obs, actions, 3e-4, 1e-3, recurrent, 2, 1, "cpu")


def seac(recurrent=False, agents=2):
    obs, actions = spaces(agents)
    return SEACTeam(obs, actions, 3e-4, 1e-3, recurrent, 2, 1, "cpu")


def fill_rollout(team):
    observations = [torch.zeros(1, 6) for _ in team.storages]
    team.initialize(observations)
    for step in range(2):
        with torch.no_grad():
            values, actions, log_probs, states = team.act(step)
        team.insert(
            observations, states, actions, log_probs, values,
            torch.ones(1, len(observations)), torch.ones(1, 1),
            torch.ones(1, 1),
        )
    team.compute_returns(
        use_gae=False, gamma=.99, gae_lambda=.95,
        use_proper_time_limits=True,
    )


def test_shared_has_one_model_and_optimizer():
    team = shared()
    assert len(team.models) == 1
    assert len({id(team.model) for _ in team.storages}) == 1
    assert team.optimizer is not None


def test_shared_update_has_no_seac_terms_and_changes_parameters():
    team = shared()
    fill_rollout(team)
    before = [parameter.detach().clone() for parameter in team.model.parameters()]
    metrics = team.update(value_loss_coef=.5, entropy_coef=.01, max_grad_norm=.5)
    assert not any("seac" in key or "importance" in key for key in metrics)
    assert any(not torch.equal(old, new) for old, new in zip(before, team.model.parameters()))


def test_shared_gru_streams_are_independent_and_masks_reset():
    team = shared(recurrent=True)
    states = team.eval_states(1, "cpu")
    observations = [torch.ones(1, 6), torch.zeros(1, 6)]
    with torch.no_grad():
        _, next_states = team.eval_act(observations, states, torch.ones(1, 1))
        _, reset_states = team.eval_act(observations, next_states, torch.zeros(1, 1))
        _, fresh_states = team.eval_act(observations, states, torch.zeros(1, 1))
    assert not torch.equal(next_states[0], next_states[1])
    assert torch.equal(reset_states[0], fresh_states[0])
    assert torch.equal(reset_states[1], fresh_states[1])


def test_seac_gru_tracks_evaluator_robot_matrix():
    team = seac(recurrent=True)
    team.initialize([torch.ones(1, 6), torch.zeros(1, 6)])
    with torch.no_grad():
        team.act(0)
    assert len(team.cross_states) == 2
    assert len(team.cross_states[0]) == 2
    assert not torch.equal(team.cross_states[0][0], team.cross_states[0][1])
    assert team.cross_rollout_starts[0][1].shape == team.cross_states[0][1].shape


@pytest.mark.parametrize("factory,filename", [
    (shared, "shared_a2c.pt"), (seac, "seac.pt"),
])
def test_checkpoint_roundtrip(tmp_path, factory, filename):
    original = factory(True)
    original.save(tmp_path, {"update": 7})
    restored = factory(True)
    state = restored.restore(tmp_path, "cpu")
    assert state["update"] == 7
    assert (tmp_path / filename).exists()
    for expected, actual in zip(original.models, restored.models):
        for left, right in zip(expected.parameters(), actual.parameters()):
            assert torch.equal(left, right)


def test_launcher_assigns_two_methods_to_each_gpu(tmp_path):
    launcher = load_script("run_comparison")
    commands = [launcher.command(tmp_path, method, 0, str(index % 2),
                                 tmp_path / "results", "attempt_1")
                for index, method in enumerate(launcher.METHODS)]
    devices = [next(value for value in command if value.startswith("algorithm.device="))
               for command in commands]
    assert devices.count("algorithm.device=cuda:0") == 2
    assert devices.count("algorithm.device=cuda:1") == 2


def test_student_t_interval_is_used_for_five_seeds():
    report = load_script("summarize_comparison")
    mean, half_width = report.ci([1, 2, 3, 4, 5])
    assert mean == 3
    assert half_width == pytest.approx(2.776 * torch.std(torch.tensor(
        [1., 2., 3., 4., 5.]
    ), unbiased=True).item() / 5 ** .5)
