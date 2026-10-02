"""Stage 3 conversion and restart contracts for the new sensor schema."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from navigation_policy import LOCAL_SIZE, NavigationActor, NavigationCritic, PolicyRuntime, SPEC
from navigation_train import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('stage3_runner', ROOT / 'scripts/run_stage3.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('condition', ['local', 'communicating_ccpd'])
def test_conversion_is_explicit_and_preserves_compatible_weights(tmp_path, condition):
    path = runner.conversion(condition, tmp_path, runner.load_layout())
    old = torch.load(runner.source_checkpoint(condition), map_location='cpu', weights_only=False)
    new = torch.load(path, map_location='cpu', weights_only=False)
    assert LOCAL_SIZE == 361 and SPEC['grid_shape'] == [7, 7, 7]
    assert new['format'] == 'navigation-training-v2'
    assert new['conversion']['reinitialized'] == ['grid.5.bias', 'grid.5.weight']
    assert new['conversion']['critic'] == 'fresh'
    assert 'actor_optimizer' not in new and 'critic_optimizer' not in new
    for key in new['conversion']['transferred']:
        torch.testing.assert_close(new['actor'][key], old['actor'][key], rtol=0, atol=0)
    assert new['actor']['grid.5.weight'].shape[1] == 32 * 49
    assert new['critic']['robot.0.weight'].shape[1] == LOCAL_SIZE + 6
    assert runner.conversion(condition, tmp_path, runner.load_layout()) == path
    actor, _ = load_checkpoint(path)
    assert actor.communication == runner.CONDITIONS[condition][0]
    with pytest.raises(ValueError, match='Incompatible'):
        load_checkpoint(runner.source_checkpoint(condition))


def test_restart_requires_frozen_protocol_and_keeps_original_checkpoints(tmp_path):
    source = runner.source_checkpoint('local')
    original_hash = runner.sha(source)
    assert runner.main(['--train', '--smoke', '--condition', 'local', '--agents', '1',
                        '--stop-after', '1', '--output', str(tmp_path)]) == 0
    stage = tmp_path / 'local/n1'
    assert not (stage / 'summary.json').exists()
    assert torch.load(stage / 'last.pt', map_location='cpu', weights_only=False)['updates'] == 1
    assert runner.main(['--train', '--smoke', '--condition', 'local', '--agents', '1',
                        '--output', str(tmp_path)]) == 0
    summary = json.loads((stage / 'summary.json').read_text())
    assert summary['completed_updates'] == 2
    assert (stage / 'last.pt').exists() and (stage / 'best.pt').exists()
    assert runner.sha(source) == original_hash
    (stage / 'summary.json').unlink()
    assert runner.main(['--train', '--smoke', '--condition', 'local', '--agents', '1',
                        '--output', str(tmp_path)]) == 0
    assert json.loads((stage / 'summary.json').read_text()) == summary
    protocol = json.loads((stage / 'protocol.json').read_text())
    protocol['config']['n_agents'] = 99
    (stage / 'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='protocol changed'):
        runner.main(['--train', '--smoke', '--condition', 'local', '--agents', '1',
                     '--output', str(tmp_path)])


def test_policy_package_requires_new_schema(tmp_path):
    from navigation_policy import export_actor
    path = tmp_path / 'actor.pt'
    export_actor(NavigationActor(False), path)
    PolicyRuntime(path)
    old = torch.load(path, map_location='cpu', weights_only=False)
    old['format'] = 'navigation-actor-v1'
    torch.save(old, path)
    with pytest.raises(ValueError, match='Incompatible'):
        PolicyRuntime(path)


def test_state_and_observation_sizes_on_fixed_map():
    layout = runner.load_layout()
    for agents in runner.FLEETS:
        from navigation_envs import NavigationEpisode
        episode = NavigationEpisode(layout, 700 + agents, 10700 + agents, 2,
                                    layout['shape'], agents)
        try:
            assert episode.obs.shape == (agents, 361 + 21 * (agents - 1))
            assert episode.state.shape == (4 * 21 * 41 + agents * 368,)
            critic = NavigationCritic(layout['shape'])
            value, _ = critic(torch.as_tensor(episode.state))
            assert torch.isfinite(value).all()
        finally:
            episode.close()
