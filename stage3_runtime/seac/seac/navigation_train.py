"""Training/evaluation adapters around the existing clipped PPO learner."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from navigation_envs import NavigationEnvs, NavigationEpisode, load_layouts, stream_seed
from navigation_policy import NavigationActor, NavigationCritic, SPEC, STATE_SCHEMA, export_actor
from shared_ppo import SharedPPO
from train_shared import DEFAULTS, atomic_save

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = {'local': (False, 'off'), 'local_ccpd': (False, 'successful'),
              'communicating': (True, 'off'), 'communicating_ccpd': (True, 'successful')}
HORIZONS = [500, 500, 2000, 2000, 10000, 10000, 10000, 10000]


class NavigationPPO(SharedPPO):
    def update(self, data):
        metrics = super().update(data)
        # Bring carried memory to the updated weights using the same bounded,
        # recorded burn-in that PPO will replay at the next rollout boundary.
        # Otherwise collection uses old-weight memory while replay uses new memory.
        history = self.actor_history
        if history is not None:
            with torch.no_grad():
                hidden = history['states'][0]
                for obs, reset in zip(history['inputs'], history['resets']):
                    _, hidden = self.actor(obs, hidden, reset)
                self.actor_state = hidden.reshape_as(self.actor_state)
        return metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for folder in ('seac/seac', 'robotic-warehouse/rware', 'scripts')
            for p in sorted((ROOT / folder).glob('*.py'))}


def configuration(manifest, condition, seed, budget=20_000_000, device='cpu'):
    communication, mode = CONDITIONS[condition]
    return dict(DEFAULTS, method='mappo', recurrent=True, seed=seed, num_env_steps=budget, device=device,
                ccpd_mode=mode, ccpd_coef=.01, ccpd_trace_events=True, num_envs=8,
                manifest=str(Path(manifest).resolve()), condition=condition, communication=communication,
                horizons=HORIZONS, time_limit=None, packet_loss=.1, n_agents=5,
                eval_seeds=list(range(1000, 1020)), long_eval_seeds=list(range(1000, 1008)),
                eval_interval_env_steps=2_000_000,
                save_interval_env_steps=2_000_000, observation_spec=SPEC, state_schema=STATE_SCHEMA)


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    if (saved.get('format') != 'navigation-training-v2' or saved.get('observation_spec') != SPEC
            or saved.get('state_schema') != STATE_SCHEMA):
        raise ValueError('Incompatible navigation checkpoint/schema')
    actor = NavigationActor(saved['config']['communication']).to(device)
    actor.load_state_dict(saved['actor'])
    actor.eval()
    return actor, saved


@torch.no_grad()
def evaluate(actor, layout, seeds, steps, replicate=0, deterministic=False, profile='reference', loss=.1,
             n_agents=5, delay=1, allow_stage4_holdout=False, stop_after_first_cycle=False):
    if stop_after_first_cycle and n_agents != 1:
        raise ValueError('First-cycle completion is an isolated, one-robot diagnostic')
    device = next(actor.parameters()).device
    previous_rngs = actor.rngs
    rows = []
    try:
        for seed in seeds:
            if 3000 <= seed <= 3049:
                raise ValueError('Reserved evaluation seed')
            spawn = seed + (100 if profile == 'changed_starts' else 0)
            task = seed + (10050 if profile == 'changed_tasks' else 10000)
            channel_seed = stream_seed(seed, replicate, 0xC4)
            episode = NavigationEpisode(layout, spawn, task, steps, layout['shape'], n_agents,
                                        actor.communication, loss, channel_seed, delay,
                                        allow_stage4_holdout=allow_stage4_holdout)
            actor.seed_streams(stream_seed(seed, replicate, 0xA4), n_agents)
            state = actor.initial_state((n_agents,), device)
            elapsed = 0.
            try:
                for _ in range(steps):
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    action, _, state = actor.act(torch.as_tensor(episode.obs, device=device), state, deterministic=deterministic)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    elapsed += time.perf_counter() - start
                    _, term, trunc, _, _ = episode.step(action.cpu().numpy())
                    if term or trunc or (stop_after_first_cycle and episode.cycles.sum()):
                        break
                row = episode.metrics()
                row.update(episode_id=seed, action_replicate=replicate, deterministic=deterministic,
                           profile=profile, packet_loss=loss if actor.communication else None,
                           channel_seed=channel_seed, action_seed=stream_seed(seed, replicate, 0xA4),
                           inference_ms_per_fleet_step=1000 * elapsed / episode.t, device=str(device))
                rows.append(row)
            finally:
                episode.close()
    finally:
        actor.rngs = previous_rngs
    return rows


def validate_policy(actor, layout, short=False):
    if short:
        return evaluate(actor, layout, [1000], 32)
    return evaluate(actor, layout, range(1000, 1020), 500) + evaluate(actor, layout, range(1000, 1008), 10000)


def train(config, output, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
    if config['deterministic']:
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    device = torch.device(config['device'])
    layouts = load_layouts(config['manifest'], n_agents=config['n_agents'])
    envs = NavigationEnvs(layouts, config['seed'], config['horizons'], config['communication'],
                          config['packet_loss'], config['n_agents'])
    learner = NavigationPPO(envs.obs.shape[-1], 4, envs.state.shape[-1], config, device,
                        actor=NavigationActor(config['communication']), critic=NavigationCritic(envs.map_shape))
    learner.actor.seed_streams(stream_seed(config['seed'], 0xAA), len(envs.envs) * envs.n)
    sources = source_hashes()
    import rware
    if Path(rware.__file__).resolve() != ROOT / 'robotic-warehouse/rware/__init__.py':
        raise RuntimeError('Incorrect simulator import')
    provenance = dict(config=config, sources=sources, manifest_sha256=sha(config['manifest']),
                      layout_hashes=[r['sha256'] for r in layouts], simulator=rware.__file__,
                      torch=torch.__version__, numpy=np.__version__, training_families=sorted({r['family'] for r in layouts}),
                      actor_parameters=sum(p.numel() for p in learner.actor.parameters()),
                      critic_parameters=sum(p.numel() for p in learner.critic.parameters()),
                      physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'), map_shape=envs.map_shape)
    write_json(output / 'provenance.json', provenance)
    initial_actor = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    initial_critic = {k: v.clone() for k, v in learner.critic.state_dict().items()}
    steps, best, next_eval = 0, None, config['eval_interval_env_steps']
    started = time.perf_counter()

    def checkpoint():
        return dict(format='navigation-training-v2', observation_spec=SPEC, state_schema=STATE_SCHEMA,
                    config=config, map_shape=envs.map_shape, actor=learner.actor.state_dict(),
                    critic=learner.critic.state_dict(), actor_optimizer=learner.actor_optimizer.state_dict(),
                    critic_optimizer=learner.critic_optimizer.state_dict(), env_steps=steps,
                    updates=learner.update_number, best_validation=best, provenance=provenance)

    try:
        with (output / 'metrics.jsonl').open('x') as metrics_file, (output / 'episodes.jsonl').open('x') as episode_file, \
                (output / 'validation.jsonl').open('x') as validation_file, (output / 'events.jsonl').open('x') as event_file:
            while steps < config['num_env_steps']:
                data, infos, collection_seconds = learner.collect(envs)
                before = learner.policy_diagnostics(data)
                if before['max_log_prob_error'] > 2e-5:
                    raise RuntimeError('Collected/reconstructed PPO log probabilities differ')
                metrics = learner.update(data)
                steps += config['num_envs'] * config['rollout_steps']
                metrics.update(env_steps=steps, collection_seconds=collection_seconds, pre_update_log_prob_error=before['max_log_prob_error'],
                               wall_seconds=time.perf_counter() - started)
                if any(not math.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))):
                    raise FloatingPointError('Nonfinite training diagnostics')
                metrics_file.write(json.dumps(metrics) + '\n'); metrics_file.flush()
                for record in learner.ccpd_records:
                    event_file.write(json.dumps(dict(update=learner.update_number, **record)) + '\n')
                for row in envs.completed:
                    episode_file.write(json.dumps(dict(env_steps=steps, **row)) + '\n')
                envs.completed.clear(); episode_file.flush()
                print(f"{config['condition']} seed={config['seed']} steps={steps} KL={metrics['approx_kl']:.5f}", flush=True)
                final = steps >= config['num_env_steps']
                if steps >= next_eval or final:
                    validation = validate_policy(copy.deepcopy(learner.actor).eval(), layouts[0], short=smoke)
                    for row in validation:
                        validation_file.write(json.dumps(dict(env_steps=steps, **row)) + '\n')
                    validation_file.flush()
                    long = [r for r in validation if r['steps'] == (32 if smoke else 10000)]
                    rank = [-sum(r['fleet_completion_gaps'] for r in long), float(np.mean([r['cycles_per_1000_steps'] for r in long])), -steps]
                    if best is None or rank > best:
                        best = rank
                        atomic_save(checkpoint(), output / 'best.pt')
                    atomic_save(checkpoint(), output / 'last.pt')
                    next_eval = (steps // config['eval_interval_env_steps'] + 1) * config['eval_interval_env_steps']
            export_actor(learner.actor, output / 'actor.pt')
            (output / 'observation_spec.json').write_text(json.dumps(SPEC, indent=2))
            (output / 'navigation_policy.py').write_bytes((ROOT / 'seac/seac/navigation_policy.py').read_bytes())
            if source_hashes() != sources:
                raise ValueError('Sources changed during training')
            summary = dict(complete=True, env_steps=steps, updates=learner.update_number,
                           actor_changed=any(not torch.equal(v, initial_actor[k]) for k, v in learner.actor.state_dict().items()),
                           critic_changed=any(not torch.equal(v, initial_critic[k]) for k, v in learner.critic.state_dict().items()),
                           wall_seconds=time.perf_counter() - started, best_validation=best)
            write_json(output / 'summary.json', summary)
            for handle in (metrics_file, episode_file, validation_file, event_file):
                handle.flush()
            write_json(output / 'complete.json', dict(config=config, sources=sources,
                       artifacts={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
            return summary
    finally:
        envs.close()


def verify_training(path, config):
    path = Path(path)
    marker = path / 'complete.json'
    if not marker.exists():
        raise ValueError(f'Incomplete training; inspect {path}')
    saved = json.loads(marker.read_text())
    if saved['config'] != config or saved['sources'] != source_hashes():
        raise ValueError(f'Incompatible completed training: {path}')
    if any(not (path / name).is_file() or sha(path / name) != digest for name, digest in saved['artifacts'].items()):
        raise ValueError(f'Modified training artifact: {path}')
    actor, checkpoint = load_checkpoint(path / 'last.pt')
    expected = math.ceil(config['num_env_steps'] / (config['num_envs'] * config['rollout_steps'])) * config['num_envs'] * config['rollout_steps']
    summary = json.loads((path / 'summary.json').read_text())
    if checkpoint['env_steps'] != expected or summary['env_steps'] != expected or not summary['complete']:
        raise ValueError(f'Incomplete training budget: {path}')
    if any(not torch.isfinite(p).all() for p in actor.parameters()):
        raise ValueError('Nonfinite actor checkpoint')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), args.output, args.smoke)
