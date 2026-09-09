import gymnasium as gym
import numpy as np
import pytest

import rware
from rware.warehouse import Action, Direction, TaskPhase


def task_signature(env):
    return [
        (
            task.rack_id,
            task.rack_position,
            task.workstation_position,
            task.return_position,
            task.phase,
        )
        for task in env.unwrapped.task_manager.tasks
    ]


def test_seeded_unique_tasks_and_movement_only_action_space():
    first = gym.make("rware-custom-5ag-v2")
    second = gym.make("rware-custom-5ag-v2")
    first.reset(seed=17)
    second.reset(seed=17)

    assert task_signature(first) == task_signature(second)
    assert len({task.rack_id for task in first.unwrapped.task_manager.tasks}) == 5
    first.unwrapped.task_manager.tasks[0] = None
    second.unwrapped.task_manager.tasks[0] = None
    first.unwrapped.task_manager.assign(0)
    second.unwrapped.task_manager.assign(0)
    assert task_signature(first) == task_signature(second)
    assert all(space.n == 4 for space in first.action_space.spaces)
    with pytest.raises(ValueError, match="TOGGLE_LOAD"):
        first.step([Action.TOGGLE_LOAD.value, 0, 0, 0, 0])

    second.reset(seed=18)
    assert task_signature(first) != task_signature(second)
    first.close()
    second.close()


def test_visible_automatic_task_cycle_and_reward():
    env = gym.make("rware-custom-5ag-v2")
    observation, _ = env.reset(seed=5)
    warehouse = env.unwrapped
    agent = warehouse.agents[0]
    task = warehouse.task_manager.tasks[0]
    shelf = warehouse.task_manager.shelf(task)
    noops = [Action.NOOP.value] * warehouse.n_agents

    other_rack = next(
        candidate for candidate in warehouse.shelfs if candidate.id != task.rack_id
    )
    agent.x, agent.y = other_rack.x, other_rack.y
    warehouse._recalc_grid()
    _, _, _, _, info = env.step(noops)
    assert task.phase == TaskPhase.PICKUP
    assert agent.carrying_shelf is None
    assert set(info) == {
        "completed_cycles",
        "deliveries",
        "pickup_time",
        "delivery_time",
        "return_time",
        "path_length",
        "wait_steps",
        "conflict_attempts",
        "movement_denied",
        "deadlock_events",
    }

    agent.x, agent.y = task.rack_position
    warehouse._recalc_grid()
    observation, _, _, _, _ = env.step(noops)
    assert task.phase == TaskPhase.AUTO_PICKUP
    np.testing.assert_array_equal(observation[0][-8:-2], [0, 1, 0, 0, 0, 0])

    pickup_position = (agent.x, agent.y)
    actions = [Action.FORWARD.value, 0, 0, 0, 0]
    _, rewards, _, _, info = env.step(actions)
    assert (agent.x, agent.y) == pickup_position
    assert agent.carrying_shelf is shelf
    assert task.phase == TaskPhase.DELIVER
    assert rewards[0] == 0
    assert info["pickup_time"][0] > 0

    agent.x, agent.y = task.workstation_position
    shelf.x, shelf.y = task.workstation_position
    warehouse._recalc_grid()
    env.step(noops)
    assert task.phase == TaskPhase.AUTO_DELIVERY

    delivery_position = (agent.x, agent.y)
    _, rewards, _, _, info = env.step([Action.RIGHT.value, 0, 0, 0, 0])
    assert (agent.x, agent.y) == delivery_position
    assert task.phase == TaskPhase.RETURN
    assert rewards[0] == pytest.approx(1.0)
    assert info["deliveries"][0] == 1
    assert info["delivery_time"][0] > 0

    agent.x, agent.y = task.return_position
    shelf.x, shelf.y = task.return_position
    warehouse._recalc_grid()
    env.step(noops)
    assert task.phase == TaskPhase.AUTO_DROP

    completed_task = task
    drop_position = (agent.x, agent.y)
    _, _, _, _, info = env.step(actions)
    assert (agent.x, agent.y) == drop_position
    assert agent.carrying_shelf is None
    assert warehouse.task_manager.tasks[0] is not completed_task
    assert warehouse.task_manager.tasks[0].phase == TaskPhase.PICKUP
    assert len({task.rack_id for task in warehouse.task_manager.tasks}) == 5
    assert info["completed_cycles"][0] == 1
    assert info["return_time"][0] > 0
    env.close()


def test_movement_and_deadlock_metrics():
    env = gym.make("rware-custom-5ag-v2")
    env.reset(seed=23)
    warehouse = env.unwrapped
    warehouse.agents[0].x, warehouse.agents[0].y = 0, 0
    warehouse.agents[0].dir = Direction.RIGHT
    warehouse.agents[1].x, warehouse.agents[1].y = 2, 0
    warehouse.agents[1].dir = Direction.LEFT
    warehouse._recalc_grid()
    _, _, _, _, info = env.step(
        [Action.FORWARD.value, Action.FORWARD.value, 0, 0, 0]
    )
    np.testing.assert_array_equal(info["conflict_attempts"][:2], [1, 1])
    np.testing.assert_array_equal(info["movement_denied"][:2], [1, 1])

    env.reset(seed=23)
    warehouse = env.unwrapped
    agent = warehouse.agents[0]
    agent.x, agent.y = 0, 0
    agent.dir = Direction.UP
    warehouse._recalc_grid()
    actions = [Action.FORWARD.value, 0, 0, 0, 0]

    for step in range(10):
        _, _, _, _, info = env.step(actions)
        assert info["path_length"][0] == 0
        assert info["wait_steps"][0] == 1
        assert info["movement_denied"][0] == 1
        assert info["deadlock_events"] == int(step == 9)
    env.close()


def test_standard_rware_keeps_toggle_action():
    env = gym.make("rware-tiny-2ag-v2")
    env.reset(seed=1)
    assert all(space.n == len(Action) for space in env.action_space.spaces)
    env.step([Action.TOGGLE_LOAD.value, Action.NOOP.value])
    env.close()
