"""Current-rollout coordination events and bounded auxiliary action selection.

No actor, replay buffer or global RNG lives here. All indices use [time, env, robot]
order, matching the PPO actor streams. Event endpoints are inclusive.
"""
from collections import Counter

import numpy as np
import torch

DETECTOR_VERSION = 'ccpd-v0-1'
DEFAULTS = dict(ccpd_mode='off', ccpd_coef=.01, ccpd_max_sample_fraction=.10,
                ccpd_max_noop_fraction=.25, ccpd_event_horizon=32, ccpd_clear_steps=3,
                ccpd_confirmation_steps=8, ccpd_progress_cap=4., ccpd_progress_weight=1.,
                ccpd_duration_cost=.1, ccpd_conflict_cost=.5, ccpd_recurrence_cost=1.,
                ccpd_min_progress=1., ccpd_min_quality=0., ccpd_trace_events=False)
MODES = ('off', 'random', 'all_conflict', 'successful')


def enabled(config):
    return config.get('ccpd_mode', 'off') != 'off' and config.get('ccpd_coef', .01) > 0


def validate(config):
    if config['ccpd_mode'] not in MODES:
        raise ValueError(f'ccpd_mode must be one of {MODES}')
    if config['ccpd_mode'] != 'off' and (config['method'] != 'mappo' or not config['recurrent']):
        raise ValueError('CCPD v0 requires recurrent MAPPO')
    for key in ('ccpd_event_horizon', 'ccpd_clear_steps', 'ccpd_confirmation_steps'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['ccpd_clear_steps'] > min(config['ccpd_event_horizon'], config['ccpd_confirmation_steps']):
        raise ValueError('ccpd_clear_steps must fit the event horizon and confirmation window')
    for key in ('ccpd_coef', 'ccpd_progress_cap', 'ccpd_progress_weight', 'ccpd_duration_cost',
                'ccpd_conflict_cost', 'ccpd_recurrence_cost', 'ccpd_min_progress', 'ccpd_min_quality'):
        if isinstance(config[key], bool) or not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key in ('ccpd_max_sample_fraction', 'ccpd_max_noop_fraction'):
        if isinstance(config[key], bool) or not np.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    if type(config['ccpd_trace_events']) is not bool:
        raise ValueError('ccpd_trace_events must be a boolean')


def detect_events(trace, done, config):
    """Label independent, nonoverlapping events, without crossing a reset/rollout.

    trace fields are numpy arrays [T,E,N,...]; done is [T,E]. A boundary may
    complete a full confirmation window, but cannot supply missing future data.
    """
    blocked = trace['robot_blocked'].astype(bool)
    conflict = blocked | trace['conflict_attempts'].astype(bool)
    T, E, N = blocked.shape
    clear, horizon, confirm = (config[k] for k in
                              ('ccpd_clear_steps', 'ccpd_event_horizon', 'ccpd_confirmation_steps'))
    events = []
    for env in range(E):
        stops = np.flatnonzero(done[:, env]).tolist()
        if not stops or stops[-1] != T - 1:
            stops.append(T - 1)
        episode_start = 0
        for stop in stops:
            for robot in range(N):
                t = episode_start
                while t <= stop:
                    if not blocked[t, env, robot]:
                        t += 1
                        continue
                    start = t
                    limit = min(stop, start + horizon - 1)
                    resolution = None
                    for end in range(start, limit + 1):
                        first = end - clear + 1
                        if first >= start and not conflict[first:end + 1, env, robot].any() and \
                                trace['movement_success'][first:end + 1, env, robot].any():
                            resolution = end
                            break
                    core_end = limit if resolution is None else resolution
                    end = core_end if resolution is None else min(stop, resolution + confirm)
                    duration = core_end - start + 1
                    additional = int(conflict[start + 1:core_end + 1, env, robot].sum())
                    event = dict(env=env, robot=robot, start=start, core_end=core_end, end=end,
                                 duration=duration, additional_conflicts=additional,
                                 progress=None, recurrence=None, quality=None)
                    if resolution is None:
                        event.update(outcome='failed' if duration == horizon else 'censored',
                                     reason='timeout' if duration == horizon else 'boundary')
                    elif end < resolution + confirm:
                        event.update(outcome='censored', reason='boundary')
                    else:
                        post = slice(resolution + 1, end + 1)
                        progress = float(trace['progress'][post, env, robot].sum())
                        recurrence = int(conflict[post, env, robot].sum())
                        quality = (config['ccpd_progress_weight'] * min(progress, config['ccpd_progress_cap'])
                                   - config['ccpd_duration_cost'] * duration
                                   - config['ccpd_conflict_cost'] * additional
                                   - config['ccpd_recurrence_cost'] * recurrence)
                        if not trace['progress_valid'][start:end + 1, env, robot].all():
                            reason = 'invalid_progress'
                        elif progress < config['ccpd_min_progress']:
                            reason = 'insufficient_progress'
                        elif conflict[end - clear + 1:end + 1, env, robot].any():
                            reason = 'late_conflict'
                        elif quality <= config['ccpd_min_quality']:
                            reason = 'low_quality'
                        else:
                            reason = 'accepted'
                        event.update(progress=progress, recurrence=recurrence, quality=quality,
                                     outcome='successful' if reason == 'accepted' else 'failed', reason=reason)
                    events.append(event)
                    t = end + 1
            episode_start = stop + 1
    return events


def capped_sample(pool, actions, budget, noop_fraction, rng):
    """Largest feasible uniform-within-action-group sample under the NOOP cap."""
    pool = np.flatnonzero(pool)
    moving = pool[actions[pool] != 0]
    size = min(budget, len(pool))
    while size and size - int(np.floor(size * noop_fraction)) > len(moving):
        size -= 1
    # First draw uniformly; replace excess NOOPs with unused non-NOOP decisions.
    selected = rng.choice(pool, size=size, replace=False)
    selected_noop = selected[actions[selected] == 0]
    excess = len(selected_noop) - int(np.floor(size * noop_fraction))
    if excess > 0:
        remove = rng.choice(selected_noop, size=excess, replace=False)
        selected = selected[~np.isin(selected, remove)]
        unused = moving[~np.isin(moving, selected)]
        selected = np.concatenate([selected, rng.choice(unused, size=excess, replace=False)])
    return selected


def select_samples(data, config, update_number):
    """Return flat weights, scalar diagnostics, and CPU event records."""
    trace = data['coordination']
    done = (data['terminated'] | data['truncated']).cpu().numpy()
    events = detect_events(trace, done, config)
    shape = data['actions'].shape
    eligible = data['eligible'].cpu().numpy().reshape(-1)
    raw_adv = data['advantages'].detach().cpu().numpy()
    if raw_adv.ndim == 2:
        raw_adv = np.broadcast_to(raw_adv[..., None], shape)
    raw_adv = raw_adv.reshape(-1)
    actions = data['actions'].cpu().numpy().reshape(-1)
    candidate = eligible & (raw_adv > 0) & ~trace['movement_denied'].astype(bool).reshape(-1)
    scores, conflict_pool = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=bool)
    for event in events:
        if event['outcome'] == 'censored':
            continue
        idx = (slice(event['start'], event['core_end'] + 1), event['env'], event['robot'])
        conflict_pool[idx] = True
        if event['outcome'] == 'successful':
            scores[idx] = event['quality']
    scores = scores.reshape(-1)
    success_pool = candidate & (scores > 0)
    budget = int(np.floor(config['ccpd_max_sample_fraction'] * eligible.sum()))
    # Independent local generator: this never advances numpy or torch global RNG.
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], update_number, 0xCC0D]))
    chosen = capped_sample(success_pool, actions, budget, config['ccpd_max_noop_fraction'], rng)
    sample_weights = scores[chosen]
    if len(chosen):
        sample_weights = np.minimum(sample_weights / sample_weights.mean(), 2.)
    if config['ccpd_mode'] != 'successful':
        pool = candidate if config['ccpd_mode'] == 'random' else candidate & conflict_pool.reshape(-1)
        chosen = capped_sample(pool, actions, len(chosen), config['ccpd_max_noop_fraction'], rng)
        # Both pools are supersets of successful candidates and can fill its budget.
        assert len(chosen) == len(sample_weights)
        sample_weights = rng.permutation(sample_weights)
    weights = np.zeros(eligible.shape, dtype=np.float32)
    weights[chosen] = sample_weights
    counts, reasons = Counter(e['outcome'] for e in events), Counter(e['reason'] for e in events)
    quality = [e['quality'] for e in events if e['quality'] is not None]
    recurrence = [e['recurrence'] for e in events if e['recurrence'] is not None]
    durations = [e['duration'] for e in events if e['quality'] is not None]
    valid_adv = raw_adv[eligible]
    metrics = dict(ccpd_events=len(events), ccpd_successful_events=counts['successful'],
                   ccpd_failed_events=counts['failed'], ccpd_censored_events=counts['censored'],
                   ccpd_quality_mean=float(np.mean(quality)) if quality else 0.,
                   ccpd_resolution_duration_mean=float(np.mean(durations)) if durations else 0.,
                   ccpd_recurrence_mean=float(np.mean(recurrence)) if recurrence else 0.,
                   ccpd_success_candidates=int(success_pool.sum()), ccpd_sample_budget=budget,
                   ccpd_selected_samples=len(chosen),
                   ccpd_selected_fraction=len(chosen) / max(int(eligible.sum()), 1),
                   ccpd_selected_noop_fraction=float(np.mean(actions[chosen] == 0)) if len(chosen) else 0.,
                   ccpd_selected_advantage_mean=float(raw_adv[chosen].mean()) if len(chosen) else 0.,
                   raw_advantage_mean=float(valid_adv.mean()) if len(valid_adv) else 0.,
                   raw_advantage_std=float(valid_adv.std()) if len(valid_adv) else 0.,
                   raw_advantage_positive_fraction=float(np.mean(valid_adv > 0)) if len(valid_adv) else 0.)
    for reason in ('timeout', 'boundary', 'invalid_progress', 'insufficient_progress', 'late_conflict', 'low_quality'):
        metrics['ccpd_rejected_' + reason] = reasons[reason]
    return torch.as_tensor(weights, device=data['actions'].device), metrics, events


def event_records(data, events, limit=20):
    """Bounded JSON-ready audit records; no observations or hidden states retained."""
    for event in sorted(events, key=lambda e: (e['start'], e['env'], e['robot']))[:limit]:
        env, robot = event['env'], event['robot']
        steps = []
        for t in range(event['start'], event['end'] + 1):
            record = {key: np.asarray(value[t, env, robot]).tolist()
                      for key, value in data['coordination'].items()}
            record['rollout_step'] = t
            steps.append(record)
        yield dict(**event, steps=steps)
