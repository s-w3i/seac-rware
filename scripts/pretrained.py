#!/usr/bin/env python3
"""Verify, evaluate, or record the released seed-0 actors without training."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'stage3_runtime/scripts')]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    manifest = json.loads((ROOT / 'pretrained/manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    for name, digest in manifest['runtime_sources'].items():
        if sha(ROOT / 'stage3_runtime' / name) != digest:
            raise ValueError(f'Runtime source mismatch: {name}')
    return manifest

def report(manifest):
    print('Seed 0; 30 validation episodes/checkpoint; not independent holdout evidence.')
    for model in manifest['models']:
        path = ROOT / 'evidence' / model['method'] / f"n{model['agents']}" / 'validation.jsonl.gz'
        with gzip.open(path, 'rt') as f:
            validations = [json.loads(line) for line in f]
        for kind, update in [('best', model['best_update']), ('last', 683)]:
            row = next(r for r in validations if r['update'] == update)
            episodes = row['rows']
            throughput = sum(e['completed_cycles'] for e in episodes) * 1000 / sum(e['steps'] for e in episodes)
            assert abs(throughput - model[kind]['throughput']) < 1e-8
            print(f"{model['method']:20s} n={model['agents']} {kind:4s} update={update:3d} "
                  f"cycles/1000={throughput:.2f} progress_failures={sum(e['progress_failure'] for e in episodes)}/{len(episodes)}")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--report', action='store_true')
    p.add_argument('--method', choices=['local', 'communicating_ccpd'])
    p.add_argument('--agents', type=int, choices=[40, 50], default=50)
    p.add_argument('--kind', choices=['best', 'last'], default='best')
    p.add_argument('--seed', type=int, default=200001, help='Demonstration seed; reserved holdout is rejected')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--replicate', type=int, default=0)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--validation', action='store_true', help='Re-run all 30 original validation episodes on CPU; can take hours')
    p.add_argument('--video', type=Path, help='MP4 output; needs a display or xvfb-run')
    p.add_argument('--output', type=Path, help='Write per-episode JSON metrics')
    args = p.parse_args()
    if args.steps < 1 or args.replicate < 0 or args.seed < 0:
        p.error('steps must be positive; seed and replicate must be nonnegative')
    if args.video and args.validation:
        p.error('--video and --validation are separate operations')
    manifest = verify()
    if args.report:
        report(manifest)
    if args.verify:
        print('All released artifact and runtime checksums verified.')
    if not args.method:
        if args.video or args.validation:
            p.error('--method is required for inference')
        if not (args.verify or args.report):
            p.error('Choose --verify, --report, or --method')
        return
    import torch
    import numpy as np
    from run_stage3 import load_layout, NavigationEpisode, stream_seed
    from navigation_policy import PolicyRuntime
    from navigation_train import evaluate
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    package = ROOT / 'pretrained' / args.method / f'n{args.agents}' / f'actor_{args.kind}.pt'
    actor = PolicyRuntime(package).actor
    layout = load_layout()
    if args.validation:
        rows = []
        for replicate in range(3):
            for seed in range(100000, 100010):
                rows.extend(evaluate(actor, layout, [seed], 5000, replicate=replicate, n_agents=args.agents))
                print(f'validation {len(rows)}/30', flush=True)
    elif not args.video:
        rows = evaluate(actor, layout, [args.seed], args.steps, replicate=args.replicate,
                        deterministic=args.deterministic, n_agents=args.agents)
    else:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        actor.seed_streams(stream_seed(args.seed, args.replicate, 0xA4), args.agents)
        ep = NavigationEpisode(layout, args.seed, args.seed+10000, args.steps, layout['shape'], args.agents,
                               actor.communication, .1, stream_seed(args.seed, args.replicate, 0xC4), 1)
        ep.w.render_mode = 'rgb_array'
        state = actor.initial_state((args.agents,), 'cpu')
        args.video.parent.mkdir(parents=True, exist_ok=True)
        def frame():
            img = Image.fromarray(ep.w.render()).convert('RGB')
            canvas = Image.new('RGB', (img.width + img.width % 2, img.height + 60 + img.height % 2), 'white')
            canvas.paste(img, (0, 60))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), f'{args.method} | {args.agents} agents | seed-0 {args.kind} | step {ep.t}/{args.steps}', fill='black')
            draw.text((12, 30), f'Scenario {args.seed} | cycles {int(ep.cycles.sum())} | 20 steps/second', fill='black')
            return np.asarray(canvas)
        try:
            with imageio.get_writer(args.video, fps=20, codec='libx264', macro_block_size=1) as writer:
                writer.append_data(frame())
                with torch.no_grad():
                    for step in range(args.steps):
                        action, _, state = actor.act(torch.as_tensor(ep.obs), state, deterministic=args.deterministic)
                        ep.step(action.numpy())
                        writer.append_data(frame())
                        if (step+1) % 250 == 0:
                            print(f'recorded {step+1}/{args.steps}', flush=True)
            row = ep.metrics()
            row.update(episode_id=args.seed, action_replicate=args.replicate, deterministic=args.deterministic,
                       action_seed=stream_seed(args.seed,args.replicate,0xA4), channel_seed=stream_seed(args.seed,args.replicate,0xC4))
            rows = [row]
        finally:
            ep.close()
    result = dict(method=args.method, agents=args.agents, training_seed=0, checkpoint_kind=args.kind,
                  checkpoint_sha256=sha(package), map_sha256=layout['sha256'], device='cpu',
                  evaluation='selection_validation' if args.validation else 'demonstration', episodes=rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(episodes=len(rows), cycles_per_1000_steps=sum(r['cycles_per_1000_steps'] for r in rows)/len(rows),
                          progress_failure_episodes=sum(r['progress_failure'] for r in rows))))

if __name__ == '__main__':
    main()
