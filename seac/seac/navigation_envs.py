"""Versioned layouts, local observations, independent RNG streams and diagnostics."""
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

from navigation_policy import (HEADINGS, LOCAL_SIZE, PACKET_SIZE, ROBOT_STATE_SIZE,
                               relative, goal_features, packet_features)
from shared_envs import decision_mask, Transition
from rware.warehouse import TaskManager


def geometry_hash(layout):
    return hashlib.sha256(('\n'.join(layout.strip().splitlines()) + '\n').encode()).hexdigest()


def geometry_family(layout):
    cells = np.array([list(row) for row in layout.strip().splitlines()])
    return min('\n'.join(''.join(row) for row in variant) for rotation in range(4)
               for variant in (np.rot90(cells, rotation), np.fliplr(np.rot90(cells, rotation))))


def validate_layout(layout, n_agents=5):
    rows = layout.strip().splitlines()
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError('Layout must be a nonempty rectangle')
    if set(''.join(rows)) - set('.xg'):
        raise ValueError('Supported layout symbols are . x g')
    racks = {(x, y) for y, row in enumerate(rows) for x, c in enumerate(row) if c == 'x'}
    goals = {(x, y) for y, row in enumerate(rows) for x, c in enumerate(row) if c == 'g'}
    if len(racks) < n_agents or not goals or n_agents < 1:
        raise ValueError('Require a workstation and at least one rack per robot')
    height, width = len(rows), len(rows[0])
    for rack in racks:
        seen, queue = {rack}, deque([rack])
        while queue:
            x, y = queue.popleft()
            for point in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if 0 <= point[0] < width and 0 <= point[1] < height and point not in seen and point not in racks - {rack}:
                    seen.add(point)
                    queue.append(point)
        if not goals <= seen:
            raise ValueError(f'Loaded route unavailable for rack {rack}')
    return height, width


def load_layouts(path, split='train', mode='within_layout', training_families=(), n_agents=5):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get('version') != 1 or mode not in ('within_layout', 'cross_layout'):
        raise ValueError('Invalid layout manifest/evaluation mode')
    result, ids, families = [], set(), {}
    training_families = set(training_families) | {e['family'] for e in manifest['layouts'] if e['split'] == 'train'}
    for entry in manifest['layouts']:
        if entry['id'] in ids:
            raise ValueError('Duplicate layout ID')
        ids.add(entry['id'])
        source = (path.parent / entry['path']).resolve()
        layout = source.read_text()
        shape = validate_layout(layout, n_agents)
        digest = geometry_hash(layout)
        if digest != entry['sha256'] or not np.isfinite(entry['weight']) or entry['weight'] <= 0:
            raise ValueError(f'Invalid layout hash/weight: {source}')
        family_key = geometry_family(layout)
        if families.setdefault(family_key, entry['family']) != entry['family']:
            raise ValueError('Reflected/rotated geometries must share a layout family')
        if entry['split'] != split:
            continue
        if mode == 'cross_layout' and entry['family'] in training_families:
            raise ValueError('Cross-layout evaluation overlaps training family')
        if mode == 'cross_layout' and entry.get('engineering_fixture', False):
            raise ValueError('Engineering fixtures cannot support unseen-layout evaluation')
        result.append(dict(entry, path=str(source), layout=layout, shape=shape))
    if not result:
        raise ValueError(f'No layouts in split {split}')
    return result


def stream_seed(*parts):
    # The high bit separates generated training/channel seeds from held-out episode IDs.
    return (1 << 40) + int(np.random.SeedSequence(list(parts)).generate_state(1)[0])


class MessageChannel:
    def __init__(self, n_agents, seed, loss=.1, delay=1):
        if not 0 <= loss <= 1 or delay not in (1, 2):
            raise ValueError('Invalid communication loss/delay')
        self.n_agents, self.loss, self.delay = n_agents, loss, delay
        self.rng = np.random.default_rng(seed)
        self.pending = []
        self.latest = [dict() for _ in range(n_agents)]
        self.sent = self.delivered = self.dropped = 0

    def transmit(self, statuses, step):
        for sender, status in enumerate(statuses):
            self.sent += 1  # Broadcast packets, not recipient copies.
            for receiver, other in enumerate(statuses):
                if sender == receiver or np.max(np.abs(np.asarray(status['position']) - other['position'])) > 2:
                    continue
                if self.rng.random() < self.loss:
                    self.dropped += 1
                else:
                    self.pending.append((step + self.delay, receiver, sender, dict(status, step=step)))

    def inboxes(self, positions, headings, step):
        waiting = []
        for due, receiver, sender, packet in self.pending:
            if due <= step:
                self.latest[receiver][sender] = packet
                self.delivered += 1
            else:
                waiting.append((due, receiver, sender, packet))
        self.pending = waiting
        result = np.zeros((self.n_agents, max(0, self.n_agents - 1), PACKET_SIZE + 1), np.float32)
        for receiver, packets in enumerate(self.latest):
            slot = 0
            for sender, packet in list(packets.items()):
                features = packet_features(packet, np.asarray(positions[receiver]), headings[receiver], step)
                if features is None:
                    del packets[sender]
                else:
                    result[receiver, slot] = np.r_[1., features]
                    slot += 1
        return result


def local_observations(w, stationary, previous_denied):
    """This is the simulator sensor adapter; the actor never receives w."""
    height, width = w.grid_size
    result = []
    for i, agent in enumerate(w.agents):
        position = np.array([agent.x, agent.y])
        forward = HEADINGS[agent.dir.value]
        right = np.array([-forward[1], forward[0]])
        grid = np.zeros((7, 5, 5), np.float32)
        for row in range(5):
            for col in range(5):
                x, y = position + (col - 2) * right + (2 - row) * forward
                if not (0 <= x < width and 0 <= y < height):
                    continue
                grid[0, row, col] = 1
                grid[1, row, col] = w.grid[1, y, x] > 0
                robot = int(w.grid[0, y, x]) - 1
                if robot >= 0:
                    grid[2, row, col] = 1
                    heading = relative(HEADINGS[w.agents[robot].dir.value], agent.dir.value)
                    index = {(0, 1): 0, (0, -1): 1, (-1, 0): 2, (1, 0): 3}[tuple(heading.astype(int))]
                    grid[3 + index, row, col] = 1
        task = w.task_manager.tasks[i]
        goal = relative(np.asarray(w.task_manager.target(task)) - position, agent.dir.value)
        scalars = np.r_[float(agent.carrying_shelf is not None), np.eye(6)[task.phase.value], goal_features(goal),
                        w._previous_actions[i], w._movement_success[i], previous_denied[i],
                        min(stationary[i], 64) / 64, min(w._cycle_steps[i], 1000) / 1000]
        assert len(scalars) == 18
        result.append(np.r_[grid.ravel(), scalars])
    return np.asarray(result, np.float32)


def privileged_state(w, local, map_shape):
    maps = np.zeros((4, *map_shape), np.float32)
    h, width = w.grid_size
    maps[3, :h, :width] = 1
    for channel, positions in enumerate([w.goals, w._rack_positions, [(s.x, s.y) for s in w.shelfs]]):
        for x, y in positions:
            maps[channel, y, x] = 1
    robots = []
    for i, a in enumerate(w.agents):
        robots.append(np.r_[1., local[i], a.x / max(width-1, 1), a.y / max(h-1, 1), np.eye(4)[a.dir.value]])
    return np.r_[maps.ravel(), np.asarray(robots).ravel()].astype(np.float32)


class IndependentTaskManager(TaskManager):
    """Use the unchanged simulator assignment rules with a separate task stream."""
    def __init__(self, warehouse, seed):
        super().__init__(warehouse)
        self.rng = np.random.default_rng(seed)

    def assign(self, agent_index):
        # Assign is synchronous; restore the spawn stream even if assignment fails.
        original = self.warehouse.np_random
        self.warehouse.np_random = self.rng
        try:
            return super().assign(agent_index)
        finally:
            self.warehouse.np_random = original


class NavigationEpisode:
    def __init__(self, layout, spawn_seed, task_seed, horizon, map_shape, n_agents=5,
                 communication=False, loss=.1, channel_seed=0, delay=1):
        if 3000 <= spawn_seed <= 3049 or 3000 <= task_seed <= 3049:
            raise ValueError('Reserved final-evaluation seeds are unavailable in this phase')
        self.layout, self.horizon, self.map_shape = layout, horizon, map_shape
        self.spawn_seed, self.task_seed, self.communication = spawn_seed, task_seed, communication
        # Original environment rules/rewards, with count override only for engineering tests.
        import gymnasium as gym
        from wrappers import RecordEpisodeStatistics
        env = gym.make('rware-custom-5ag-routing-v2', layout=layout['layout'], n_agents=n_agents,
                       request_queue_size=n_agents, max_steps=None, max_inactivity_steps=None,
                       coordination_trace_enabled=True)
        self.env = RecordEpisodeStatistics(gym.wrappers.TimeLimit(env, horizon))
        self.env.unwrapped.task_manager = IndependentTaskManager(self.env.unwrapped, task_seed)
        self.env.reset(seed=spawn_seed)
        self.w = self.env.unwrapped
        self.n, self.t = n_agents, 0
        self.channel = MessageChannel(n_agents, channel_seed, loss, delay)
        self.stationary = np.zeros(n_agents, int)
        self.longest_stationary = np.zeros(n_agents, int)
        self.previous_denied = np.zeros(n_agents)
        self.gap = self.max_gap = self.gap_events = 0
        self.cycles = np.zeros(n_agents, int)
        self.durations, self.reward = [], np.zeros(n_agents)
        self.totals = {}
        self.denial_reasons = dict(boundary=0, rack=0, traffic=0)
        self.repetition = np.zeros(n_agents, int)
        self.histories = [[] for _ in range(n_agents)]
        self.aged_robot_steps = 0
        self.navigation = 0
        self.initial = dict(positions=[[int(a.x), int(a.y)] for a in self.w.agents], headings=[a.dir.value for a in self.w.agents],
                            racks=[t.rack_id for t in self.w.task_manager.tasks])
        self.refresh()

    def refresh(self):
        local = local_observations(self.w, self.stationary, self.previous_denied)
        inbox = self.channel.inboxes([[a.x, a.y] for a in self.w.agents], [a.dir.value for a in self.w.agents], self.t)
        if not self.communication:
            inbox[:] = 0
        self.obs = np.concatenate([local, inbox.reshape(self.n, -1)], -1)
        self.state = privileged_state(self.w, local, self.map_shape)

    def step(self, actions):
        mask = decision_mask(self.w)
        self.navigation += int(mask.sum())
        positions = [(a.x, a.y) for a in self.w.agents]
        phases = [t.phase.value for t in self.w.task_manager.tasks]
        statuses = [dict(position=[a.x, a.y], heading=a.dir.value,
                         goal=list(self.w.task_manager.target(self.w.task_manager.tasks[i])),
                         phase=phases[i], carrying=a.carrying_shelf is not None,
                         previous_action=int(np.argmax(self.w._previous_actions[i])),
                         cycle_age=int(self.w._cycle_steps[i])) for i, a in enumerate(self.w.agents)]
        if self.communication:
            self.channel.transmit(statuses, self.t)
        reasons = []
        for i, a in enumerate(self.w.agents):
            x, y = np.array(positions[i]) + HEADINGS[a.dir.value]
            if not (0 <= x < self.w.grid_size[1] and 0 <= y < self.w.grid_size[0]):
                reasons.append('boundary')
            elif a.carrying_shelf is not None and self.w.grid[1, y, x] and not (
                    self.w.grid[0, y, x] and self.w.agents[self.w.grid[0, y, x]-1].carrying_shelf is not None):
                reasons.append('rack')
            else:
                reasons.append('traffic')
        _, rewards, term, trunc, info = self.env.step(np.where(mask, actions, 0))
        self.t += 1
        completed = np.asarray(info['completed_cycles'])
        self.gap += 1
        self.max_gap = max(self.max_gap, self.gap)
        self.gap_events += int(self.gap == 500)
        if completed.sum():
            self.gap = 0
        self.cycles += completed
        self.reward += rewards
        self.durations.extend(np.asarray(info['cycle_time'])[completed > 0].tolist())
        changed = np.array([phases[i] != t.phase.value for i, t in enumerate(self.w.task_manager.tasks)])
        self.stationary = np.where(changed | (info['path_length'] > 0), 0, self.stationary + mask)
        self.longest_stationary = np.maximum(self.longest_stationary, self.stationary)
        self.previous_denied = np.asarray(info['movement_denied'])
        self.aged_robot_steps += int((self.w._cycle_steps >= 500).sum())
        for i in range(self.n):
            if info['movement_denied'][i]:
                self.denial_reasons[reasons[i]] += 1
            if mask[i]:
                signature = (*positions[i], statuses[i]['heading'], phases[i], *statuses[i]['goal'], int(actions[i]))
                h = (self.histories[i] + [signature])[-32:]
                self.histories[i] = h
                if len(h) == 32 and any(all(h[j] == h[j-p] for j in range(p, 32)) for p in range(1, 9)):
                    self.repetition[i] += 1
            else:
                self.histories[i] = []
        for key, value in info.items():
            if key.startswith('reward_') or key in ('completed_cycles', 'deliveries', 'pickups', 'robot_blocked',
                    'movement_denied', 'movement_attempts', 'conflict_attempts', 'deadlock_events', 'wait_steps'):
                self.totals[key] = self.totals.get(key, 0.) + float(np.asarray(value).sum())
        info['coordination']['requested_action'] = np.asarray(actions).copy()
        self.refresh()
        return np.asarray(rewards, np.float32), term, trunc, info, mask

    def metrics(self):
        ages = self.w._cycle_steps.tolist()
        return dict(**self.totals, seed=self.spawn_seed, task_seed=self.task_seed, layout_id=self.layout['id'],
                    layout_family=self.layout['family'], geometry_sha256=self.layout['sha256'],
                    steps=self.t, requested_steps=self.horizon, initial=self.initial,
                    cycles_per_1000_steps=1000 * int(self.cycles.sum()) / max(self.t, 1),
                    cycles_per_robot=self.cycles.tolist(), cycle_durations=self.durations,
                    reward_per_robot=self.reward.tolist(), unfinished_task_ages=ages,
                    max_unfinished_task_age=max(ages), unfinished_tasks=self.n,
                    zero_cycle_fraction=float(np.mean(self.cycles == 0)),
                    mean_cycle_time=float(np.mean(self.durations)) if self.durations else None,
                    p95_cycle_time=float(np.percentile(self.durations, 95)) if self.durations else None,
                    navigation_decisions=self.navigation, team_stall_events=self.totals.get('deadlock_events', 0),
                    max_fleet_completion_gap=self.max_gap, fleet_completion_gaps=self.gap_events, fleet_failure=self.max_gap >= 500,
                    longest_stationary=self.longest_stationary.tolist(), repetition_windows=self.repetition.tolist(),
                    aged_robot_time_fraction=self.aged_robot_steps / max(self.t * self.n, 1),
                    aged_endpoint_fraction=float(np.mean(np.array(ages) >= 500)), denial_reasons=self.denial_reasons,
                    packets_sent=self.channel.sent, deliveries_received=self.channel.delivered,
                    packets_dropped=self.channel.dropped)

    def close(self):
        self.env.close()


class NavigationEnvs:
    def __init__(self, layouts, seed, horizons, communication=False, loss=.1, n_agents=5):
        self.layouts, self.seed, self.horizons = layouts, seed, list(horizons)
        self.communication, self.loss, self.n = communication, loss, n_agents
        self.map_shape = tuple(map(int, np.max([r['shape'] for r in layouts], axis=0)))
        self.counters = np.zeros(len(horizons), int)
        self.layout_rng = [np.random.default_rng(stream_seed(seed, i, 0x1A)) for i in range(len(horizons))]
        self.envs = [self.new_episode(i) for i in range(len(horizons))]
        self.completed = []
        self.refresh()

    def new_episode(self, i):
        weights = np.array([r['weight'] for r in self.layouts], float)
        layout = self.layouts[self.layout_rng[i].choice(len(weights), p=weights / weights.sum())]
        episode = int(self.counters[i])
        self.counters[i] += 1
        return NavigationEpisode(layout, stream_seed(self.seed, i, episode, 1), stream_seed(self.seed, i, episode, 2),
                                 self.horizons[i], self.map_shape, self.n, self.communication, self.loss,
                                 stream_seed(self.seed, i, episode, 3))

    def refresh(self):
        self.obs = np.stack([e.obs for e in self.envs])
        self.state = np.stack([e.state for e in self.envs])

    def step(self, actions):
        final_obs, final_states, rewards, terms, truncs, infos, masks = [], [], [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            reward, term, trunc, info, mask = env.step(action)
            final_obs.append(env.obs.copy())
            final_states.append(env.state.copy())
            rewards.append(reward); terms.append(term); truncs.append(trunc); infos.append(info); masks.append(mask)
            if term or trunc:
                self.completed.append(env.metrics())
                env.close()
                self.envs[i] = self.new_episode(i)
        self.refresh()
        return Transition(self.obs, self.state, np.asarray(final_obs), np.asarray(final_states),
                          np.asarray(rewards), np.asarray(terms), np.asarray(truncs), np.asarray(masks), infos)

    def close(self):
        for env in self.envs:
            env.close()
