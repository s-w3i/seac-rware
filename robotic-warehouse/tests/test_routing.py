"""Phase 2 checks: scripted environment steps and inference only, no training."""
from collections import deque
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest
import rware
from rware.warehouse import Action, Direction, ObservationType, TaskPhase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'seac' / 'seac'))
from wrappers import RecordEpisodeStatistics


def make(**kwargs):
    env = gym.make('rware-custom-5ag-routing-v2', **kwargs)
    env.reset(seed=5)
    return env


def sync(w):
    """Refresh cached distances after arranging a test scenario."""
    w._recalc_grid()
    w._distances = np.array([w._static_distance((a.x, a.y), *w._routing_context(i))
                             for i, a in enumerate(w.agents)])


def test_observations_reset_and_baseline():
    baseline = gym.make('rware-custom-5ag-v2')
    old, _ = baseline.reset(seed=5)
    assert old[0].shape == (79,)
    assert baseline.unwrapped.sensor_range == 1
    assert all(space.n == 4 for space in baseline.action_space)
    env = make()
    obs, info = env.reset(seed=5)
    assert obs[0].shape == (199,)
    assert env.observation_space.contains(obs)
    assert np.isfinite(obs).all()
    np.testing.assert_array_equal(obs[0][-7:-3], np.zeros(4))
    assert obs[0][-8] == obs[0][-1]
    env.step([0] * 5)
    obs, _ = env.reset(seed=5)
    assert not obs[0][-7:-1].any()
    for kind in (ObservationType.DICT,):
        dictionary = make(observation_type=kind)
        obs, _ = dictionary.reset(seed=5)
        assert dictionary.observation_space.contains(obs)
        flat = gym.spaces.flatten(dictionary.observation_space[0], obs[0])
        expected, _ = env.reset(seed=5)
        np.testing.assert_array_equal(flat, expected[0])
        dictionary.close()
    env.close()
    baseline.close()


def test_distance_and_configuration():
    env = make(n_agents=1)
    w = env.unwrapped
    w._rack_positions = {(1, 0), (1, 1)}
    assert w._static_distance((0, 0), (2, 0), False, (5, 5)) == 2
    assert w._static_distance((0, 0), (2, 0), True, (5, 5)) == 6
    assert w._static_distance((0, 0), (1, 0), True, (1, 0)) == 1
    assert w._static_distance((1, 0), (2, 0), True, (5, 5)) == 1
    assert w._static_distance((0, 0), (0, 0), True, (5, 5)) == 0
    assert w._static_distance((0, 0), (1, 1), True, (5, 5)) == 60
    assert w._static_distance((0, 0), (-1, 0), False, (5, 5)) == 60
    env.close()
    with pytest.raises(ValueError):
        make(task_manager_enabled=False)
    with pytest.raises(ValueError):
        make(observation_type=ObservationType.IMAGE)


def test_progress_wait_and_service_rewards():
    env = make(n_agents=1)
    w = env.unwrapped
    a = w.agents[0]
    a.x, a.y, a.dir = 0, 0, Direction.RIGHT
    task = w.task_manager.tasks[0]
    task.rack_position = (2, 0)
    sync(w)
    obs, reward, _, _, info = env.step([1])
    assert reward[0] == pytest.approx(.09)
    assert info['reward_progress'][0] == pytest.approx(.1)
    assert obs[0][-3] == 1
    a.dir = Direction.LEFT
    _, reward, _, _, _ = env.step([1])
    assert reward[0] == pytest.approx(-.11)
    for action in (2, 3):
        _, _, _, _, info = env.step([action])
        assert info['wait_steps'][0] == 0
    _, _, _, _, info = env.step([0])
    assert info['wait_steps'][0] == 1
    # Physically invalid boundary movement is not robot blocking.
    a.dir = Direction.LEFT
    for _ in range(11):
        _, reward, _, _, info = env.step([1])
        assert info['movement_denied'][0] == 1
        assert info['robot_blocked'][0] == info['reward_stall'][0] == 0
        assert info['deadlock_events'] == 0
    env.close()


def test_conflict_stall_and_rearming():
    env = make(n_agents=2)
    w = env.unwrapped
    for a, x, direction in zip(w.agents, (0, 1), (Direction.RIGHT, Direction.LEFT)):
        a.x, a.y, a.dir = x, 0, direction
    sync(w)
    for step in range(12):
        _, reward, _, _, info = env.step([1, 1])
        np.testing.assert_array_equal(info['robot_blocked'], [1, 1])
        assert info['deadlock_events'] == int(step == 9)
        assert reward[0] == pytest.approx(-.11 - (.2 if step >= 9 else 0))
    env.step([2, 0])
    assert not w._blocked_streak.any()
    w.agents[0].dir = Direction.RIGHT
    for step in range(10):
        _, _, _, _, info = env.step([1, 1])
        assert info['deadlock_events'] == int(step == 9)
    # One robot remains stationary; the blocked robot still gets a conflict penalty.
    _, _, _, _, info = env.step([0, 0])
    assert not info['robot_blocked'].any()
    env.close()


def navigation_action(w, i=0):
    a = w.agents[i]
    task = w.task_manager.tasks[i]
    if task.phase in (TaskPhase.AUTO_PICKUP, TaskPhase.AUTO_DELIVERY, TaskPhase.AUTO_DROP):
        return 0
    start, target = (a.x, a.y), w.task_manager.target(task)
    occupied = {(other.x, other.y) for other in w.agents if other is not a}
    blocked = {(s.x, s.y) for s in w.shelfs if s is not a.carrying_shelf} if a.carrying_shelf else set()
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        point, route = queue.popleft()
        if point == target:
            if not route:
                return 0
            dx, dy = route[0][0] - a.x, route[0][1] - a.y
            desired = {(1, 0): Direction.RIGHT, (-1, 0): Direction.LEFT,
                       (0, 1): Direction.DOWN, (0, -1): Direction.UP}[dx, dy]
            return 1 if a.dir == desired else 3
        x, y = point
        for nxt in ((x-1,y), (x+1,y), (x,y-1), (x,y+1)):
            if (0 <= nxt[0] < w.grid_size[1] and 0 <= nxt[1] < w.grid_size[0]
                    and nxt not in seen and nxt not in occupied and nxt not in blocked):
                seen.add(nxt)
                queue.append((nxt, route + [nxt]))
    raise AssertionError('Scripted target is unreachable')


def test_five_robot_cycle_and_episode_totals():
    env = make(max_steps=500)
    w = env.unwrapped
    # Park four unloaded robots on racks away from robot 0's assigned rack.
    task = w.task_manager.tasks[0]
    parking = [s for s in w.shelfs if s.id != task.rack_id][:4]
    for a, shelf in zip(w.agents[1:], parking):
        a.x, a.y = shelf.x, shelf.y
    w.agents[0].x, w.agents[0].y = 0, 0
    sync(w)
    wrapped = RecordEpisodeStatistics(env)
    expected = {}
    phases = []
    for step in range(300):
        before = w.task_manager.tasks[0].phase
        obs, rewards, _, _, info = wrapped.step([navigation_action(w)] + [0]*4)
        assert w.observation_space.contains(obs)
        components = [value for key, value in info.items() if key.startswith('reward_')]
        np.testing.assert_allclose(sum(components), rewards)
        for key in ('pickups','deliveries','completed_cycles','path_length','wait_steps'):
            expected[key] = expected.get(key, 0) + np.sum(info[key])
        if before in (TaskPhase.AUTO_PICKUP, TaskPhase.AUTO_DELIVERY, TaskPhase.AUTO_DROP):
            phases.append(before)
            bonus = {TaskPhase.AUTO_PICKUP:1, TaskPhase.AUTO_DELIVERY:2, TaskPhase.AUTO_DROP:3}[before]
            assert rewards[0] == pytest.approx(bonus-.01)
            assert info['reward_progress'][0] == info['wait_steps'][0] == 0
            assert obs[0][-8] == obs[0][-1]
        if info['completed_cycles'][0]:
            assert info['cycle_time'][0] == step+1
            break
    else:
        pytest.fail('No completed cycle')
    assert phases == [TaskPhase.AUTO_PICKUP, TaskPhase.AUTO_DELIVERY, TaskPhase.AUTO_DROP]
    # End one step later and verify completed-only means and new unfinished task.
    w.max_steps = w._cur_steps + 1
    _, _, done, _, info = wrapped.step([0]*5)
    assert done
    summary = info['episode_metrics']
    for key in expected:
        assert summary[key] == expected[key] + np.sum(info[key])
    assert summary['mean_cycle_time'] == step+1
    assert summary['cycles_per_1000_steps'] == pytest.approx(1000/(step+2))
    assert summary['unfinished_cycles'] == 5
    wrapped.reset(seed=5)
    assert wrapped.metric_totals == {}
    wrapped.close()


def test_truncation_empty_episode_and_inference():
    import torch
    from model import Policy
    env = RecordEpisodeStatistics(gym.wrappers.TimeLimit(make(max_steps=None), 2))
    obs, _ = env.reset(seed=5)
    model = Policy(env.observation_space[0], env.action_space[0])
    model.eval()
    with torch.no_grad():
        value, action, logp, _ = model.act(torch.tensor(obs[0]).unsqueeze(0),
                                         torch.zeros(1,1), torch.ones(1,1))
    assert 0 <= action.item() < 4
    assert torch.isfinite(value).all() and torch.isfinite(logp).all()
    env.step([0]*5)
    _, _, terminated, truncated, info = env.step([0]*5)
    assert truncated and not terminated
    assert info['episode_metrics']['completed_cycles'] == 0
    assert 'mean_cycle_time' not in info['episode_metrics']
    assert info['episode_metrics']['wait_steps'] == 10
    env.reset(seed=5)
    assert env.metric_totals == {}
    env.close()


def test_standing_rack_and_unreachable_progress():
    env = make(n_agents=1)
    w = env.unwrapped
    a = w.agents[0]
    task = w.task_manager.tasks[0]
    a.carrying_shelf = w.task_manager.shelf(task)
    a.x, a.y, a.dir = 0, 1, Direction.RIGHT
    a.carrying_shelf.x, a.carrying_shelf.y = 0, 1
    task.phase = TaskPhase.DELIVER
    # Ensure the cell ahead is a standing rack, not the carried shelf's former location.
    rack = next(s for s in w.shelfs if s is not a.carrying_shelf)
    rack.x, rack.y = 1, 1
    sync(w)
    _, _, _, _, info = env.step([1])
    assert info['movement_denied'][0] == 1
    assert info['robot_blocked'][0] == info['reward_conflict'][0] == 0
    # An unreachable static target remains finite and produces no artificial progress.
    task.workstation_position = (-1, 0)
    a.dir = Direction.DOWN
    sync(w)
    obs, _, _, _, info = env.step([1])
    assert obs[0][-8] == 1
    assert info['reward_progress'][0] == 0
    env.close()


def test_reporting_ignores_step_events_and_weights_completions():
    from train import _squash_info
    first = {'episode_reward': np.array([1., 2.]), 'episode_length': 10,
             'episode_metrics': {'completed_cycles': 1, 'mean_cycle_time': 4.,
                                 'cycles_per_1000_steps': 100.}}
    second = {'episode_reward': np.array([3., 4.]), 'episode_length': 30,
              'episode_metrics': {'completed_cycles': 3, 'mean_cycle_time': 8.,
                                  'cycles_per_1000_steps': 100.}}
    summary = _squash_info([{'completed_cycles': np.array([99])}, first, second])
    assert summary['episode_reward'] == 5
    assert summary['mean_cycle_time'] == 7
    assert summary['completed_cycles'] == 2
    assert summary['cycles_per_1000_steps'] == 100
    assert _squash_info([{'path_length': [1]}]) == {}
