"""Diagnose finetune checkpoint shape mismatches: print what's ON DISK vs what the configs REBUILD.

Run after a (smoke) train pass, before finetune:
    python scripts/diagnose_finetune_ckpts.py --variant single --task 4

For each of reward / kernel, it loads the saved finetuning checkpoint, prints its real layer shapes,
and prints the dims the finetune RewardConfig would rebuild the net with — so any mismatch is obvious.
"""
import argparse
import os
import pickle
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def leaf_shapes(d, prefix=''):
    """Walk a (possibly nested) flax state-dict and yield 'path: shape' for array leaves."""
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            out += leaf_shapes(v, f'{prefix}/{k}')
    else:
        shape = getattr(d, 'shape', None)
        if shape is not None:
            out.append(f'{prefix}: {tuple(shape)}')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', default='single')
    p.add_argument('--task', type=int, default=4)
    args = p.parse_args()

    import subprocess
    try:
        head = subprocess.check_output(['git', 'log', '--oneline', '-1']).decode().strip()
    except Exception:
        head = '(unknown)'
    print('git HEAD :', head)

    data = args.variant            # kernel/reward dir spelling
    print(f'\nvariant={args.variant}  task={args.task}\n' + '=' * 70)

    # ---- reward ----
    reward_path = f'Finetuning/Rewards/cube/{data}/Models/Cube_{args.variant.capitalize()}_Task{args.task}_Reward_0.pkl'
    print(f'\n[reward] looking for: {reward_path}')
    print(' exists:', os.path.exists(reward_path))
    # also list everything actually in that Models dir (catch stale files)
    rdir = f'Finetuning/Rewards/cube/{data}/Models'
    if os.path.isdir(rdir):
        print(' files in dir:', sorted(os.listdir(rdir)))
    if os.path.exists(reward_path):
        with open(reward_path, 'rb') as f:
            d = pickle.load(f)
        print(' SAVED reward leaf shapes:')
        for s in leaf_shapes(d):
            print('   ', s)

    print('\n[reward] the finetune RewardConfig in run_cube_pipeline.py currently rebuilds SimpleReward with'
          ' hidden_dim_reward=512, num_hidden_layers_reward=4 (grep RWConfig in run_cube_pipeline.py to confirm)')

    # ---- kernel ----
    kdir = f'Finetuning/Kernels/cube/{data}/Models/0'
    print(f'\n[kernel] dir: {kdir}  exists: {os.path.isdir(kdir)}')
    if os.path.isdir(kdir):
        files = sorted(os.listdir(kdir))
        print(' files:', files)
        if files:
            with open(os.path.join(kdir, files[0]), 'rb') as f:
                d = pickle.load(f)
            print(' SAVED kernel[0] leaf shapes:')
            for s in leaf_shapes(d):
                print('   ', s)


if __name__ == '__main__':
    main()
