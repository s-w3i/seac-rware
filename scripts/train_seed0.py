#!/usr/bin/env python3
"""New seed-0 adaptation run from released Stage-1 checkpoints (never auto-evaluates holdout)."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stage3_runtime/scripts'))
import run_stage4 as runner
# This wrapper declares the reduced study explicitly without modifying the historical runtime.
runner.SEEDS = (0,)
runner.DEFAULT_OUTPUT = ROOT / 'results/stage4_reproduction'
runner.source_path = lambda method, seed: ROOT / 'pretrained/stage1' / method / f'seed_{seed}/last.pt'
_original_child_command = runner.child_command
def child_command(args, operation, **options):
    command = _original_child_command(args, operation, **options)
    command[2] = str(Path(__file__).resolve())
    return command
runner.child_command = child_command

if __name__ == '__main__':
    if '--campaign' in sys.argv or '--evaluate' in sys.argv or '--lock-evaluation' in sys.argv:
        raise SystemExit('Use --train for one method, or --prepare/--profile/--report. Holdout requires a separately locked protocol.')
    runner.main()
