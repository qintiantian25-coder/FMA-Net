import argparse
import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.gate_config import DEFAULT_GATE_CONFIG_PATH, pick_value, read_gate_section


def run(cmd):
    print('> ' + ' '.join(cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description='Run base and gated evaluation back-to-back for quick comparison.')
    parser.add_argument('--gate_config_path', type=str, default=DEFAULT_GATE_CONFIG_PATH)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--base_ckpt', type=str, default=None)
    parser.add_argument('--gate_ckpt', type=str, default=None)
    parser.add_argument('--split', type=str, default=None, choices=['val', 'test'])
    parser.add_argument('--fill_strength', type=float, default=None)
    parser.add_argument('--blind_threshold', type=float, default=None)
    args = parser.parse_args()

    common_cfg = read_gate_section(args.gate_config_path, 'common')
    compare_cfg = read_gate_section(args.gate_config_path, 'compare')

    config_path = pick_value(args.config_path, common_cfg, 'config_path', str, './experiment.cfg')
    base_ckpt = pick_value(args.base_ckpt, compare_cfg, 'base_ckpt', str, None)
    gate_ckpt = pick_value(args.gate_ckpt, compare_cfg, 'gate_ckpt', str, None)
    split = pick_value(args.split, compare_cfg, 'split', str, 'val')
    fill_strength = pick_value(args.fill_strength, compare_cfg, 'fill_strength', float, 1.0)
    blind_threshold = pick_value(args.blind_threshold, compare_cfg, 'blind_threshold', float, None)

    if base_ckpt is None:
        parser.error('Missing --base_ckpt. Set it via CLI or [compare].base_ckpt in gate config.')
    if gate_ckpt is None:
        parser.error('Missing --gate_ckpt. Set it via CLI or [compare].gate_ckpt in gate config.')
    if split not in ('val', 'test'):
        parser.error("split must be 'val' or 'test'.")

    base_cmd = [
        sys.executable,
        'scripts/eval_stage2_or_gate.py',
        '--gate_config_path', args.gate_config_path,
        '--config_path', config_path,
        '--base_ckpt', base_ckpt,
        '--split', split,
    ]
    if blind_threshold is not None:
        base_cmd.extend(['--blind_threshold', str(blind_threshold)])

    gate_cmd = [
        sys.executable,
        'scripts/eval_stage2_or_gate.py',
        '--gate_config_path', args.gate_config_path,
        '--config_path', config_path,
        '--base_ckpt', base_ckpt,
        '--gate_ckpt', gate_ckpt,
        '--split', split,
        '--fill_strength', str(fill_strength),
    ]
    if blind_threshold is not None:
        gate_cmd.extend(['--blind_threshold', str(blind_threshold)])

    rc1 = run(base_cmd)
    rc2 = run(gate_cmd)

    if rc1 != 0 or rc2 != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

