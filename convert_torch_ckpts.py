#!/usr/bin/env python3
r"""Convert the teammate's PyTorch (torch.save) ODP checkpoints -> flax pickles the JAX loaders read.

WHY: the teammate's reward/critic/kernel/planner checkpoints are torch.save zip archives (state_dicts with
(out,in) Linear layout); the JAX loaders do pickle.load + flax.serialization.from_state_dict and CANNOT read
them. This remaps each net into the flax param tree (transpose Linear `weight`->`kernel`, LayerNorm
`weight`->`scale`, split nn.MultiheadAttention `in_proj_weight` into flax q/k/v, copy the planner's Fourier
`freqs`) and writes the flax pickles in the exact format/paths the loaders expect.

RUN ON THE SERVER (needs: torch + jax + flax + numpy, and BOTH repos present):
    cd ~/ODP-jax && python convert_torch_ckpts.py \
        --torch-repo ~/Online-Diffusion-Planning \
        --src checkpoints --dst Finetuning

VALIDATION: every net is forward-parity checked. The TORCH ground-truth forward is computed in a SUBPROCESS
run with cwd=torch-repo (so its `Pretrain.*` imports resolve without colliding with this repo's `Pretrain.*`),
on a FIXED seeded input; this process then runs the converted FLAX net on the SAME input and asserts
max|flax-torch| <= TOL. A net that fails parity is NOT written (it prints the diff so we can fix the mapping).

Reads  (torch.save):  <src>/{Rewards,Critics,Kernels,Planners}/cube/<spec>/Models|Stats/...
Writes (flax pickle): <dst>/{Rewards,Critics,Kernels,Planners}/cube/<spec>/Models|Stats/...
"""
import argparse
import glob
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

TOL = 2e-3   # float32 forward: a correct mapping is ~1e-5-1e-4; a real bug is O(0.1+)

# cube/task4 dims (from run_cube_pipeline.py): reward/kernel use specific='single', critic/planner='single-play'.
TASK_ID = 4
REWARD_HID, REWARD_LAYERS = 512, 4
CRITIC_HID, CRITIC_LAYERS = 512, 4
KERNEL_MODES, KERNEL_LAYERS, KERNEL_HID, KERNEL_NOISE = 10, 4, 514, 5e-4
PLAN_EMB, PLAN_DMODEL, PLAN_HEADS, PLAN_DEPTH = 128, 256, 4, 2
PLAN_EMB_TYPE = 'fourier'


# ----------------------------------------------------------------------------- torch-side ref (subprocess)
_TORCH_REF_SRC = r'''
import sys, os, pickle
import numpy as np
import torch, torch.nn as nn
# The torch repo defines SimpleReward/Critic MULTIPLE times in one file; `import` would grab the LAST
# (a wrong fixed-arch Critic). So reward/critic are rebuilt GENERICALLY here to match the checkpoint keys
# net.0..15 ([Linear,LN,SiLU]x(1+layers) + final Linear). MoG and DiT1d are single-def -> import them.
from Pretrain.Transition_Kernel.Kernel_Net import MoGTransitionKernel as TKernel
from Pretrain.Planners.Backbone.Dit import DiT1d as TDiT

def _mlp(in_dim, hidden, n_blocks):
    layers = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
    for _ in range(n_blocks - 1):
        layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()]
    layers += [nn.Linear(hidden, 1)]
    return nn.Sequential(*layers)

class RewardNet(nn.Module):
    def __init__(self, obs, act, hidden, layers):
        super().__init__(); self.net = _mlp(obs + act, hidden, 1 + layers)
    def forward(self, o, a):
        return self.net(torch.cat([o, a], -1)).squeeze(-1)

class CriticNet(nn.Module):
    def __init__(self, obs, hidden, layers):
        super().__init__(); self.net = _mlp(obs, hidden, 1 + layers)
    def forward(self, o):
        return self.net(o).squeeze(-1)

spec = pickle.load(open(sys.argv[1], 'rb'))
out = {}
d = spec['dims']
torch.manual_seed(0)
with torch.no_grad():
    for name, info in spec['nets'].items():
        kind = info['kind']; sd = torch.load(info['path'], map_location='cpu')
        if kind == 'planner':
            sd = sd['ema']
        if kind == 'reward':
            net = RewardNet(d['obs'], d['act'], info['hid'], info['layers'])
            net.load_state_dict(sd); net.eval()
            o = net(torch.tensor(spec['ins']['obs']), torch.tensor(spec['ins']['act']))
        elif kind == 'critic':
            net = CriticNet(d['obs'], info['hid'], info['layers'])
            net.load_state_dict(sd); net.eval()
            o = net(torch.tensor(spec['ins']['obs']))
        elif kind == 'kernel':
            net = TKernel(d['obs'], d['act'], num_modes=info['modes'],
                          num_hidden_layers=info['layers'], hidden_dim=info['hid'],
                          noise_floor=info['noise'])
            net.load_state_dict(sd); net.eval()
            mu, log_std, w = net(torch.tensor(spec['ins']['s']), torch.tensor(spec['ins']['a']))
            o = torch.cat([mu.reshape(mu.shape[0], -1), log_std.reshape(log_std.shape[0], -1), w], -1)
        elif kind == 'planner':
            net = TDiT(d['obs'] + d['act'], info['emb'], info['dmodel'], info['heads'],
                       info['depth'], 0.0, info['emb_type'])
            net.load_state_dict(sd); net.eval()
            o = net(torch.tensor(spec['ins']['x']), torch.tensor(spec['ins']['noise']))
        out[name] = np.asarray(o.detach().cpu().numpy(), dtype=np.float32)
np.savez(sys.argv[2], **out)
'''


def torch_refs(torch_repo, nets, ins, dims):
    """Run the torch nets in a subprocess (cwd=torch_repo) -> dict name->output array."""
    with tempfile.TemporaryDirectory() as td:
        srcf = os.path.join(td, '_torch_ref.py')
        specf = os.path.join(td, 'spec.pkl')
        outf = os.path.join(td, 'out.npz')
        open(srcf, 'w').write(_TORCH_REF_SRC)
        pickle.dump({'nets': nets, 'ins': ins, 'dims': dims}, open(specf, 'wb'))
        r = subprocess.run([sys.executable, srcf, specf, outf], cwd=torch_repo,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"torch ref subprocess failed:\nSTDOUT{r.stdout}\nSTDERR{r.stderr}")
        z = np.load(outf)
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- flax param-tree mapping
def _ordered(d):
    """Iterate a flax params dict in insertion (module-creation) order."""
    return list(d.items())


def _torch_layers(sd, prefix):
    """Return (linears, lns) as ordered lists of (weight, bias) numpy arrays for keys f'{prefix}.<i>.*',
    sorted by the integer <i>. Linear has 2-D weight, LayerNorm has 1-D weight."""
    idx = {}
    for k, v in sd.items():
        if not k.startswith(prefix + '.'):
            continue
        rest = k[len(prefix) + 1:]
        parts = rest.split('.')
        if not parts[0].isdigit():
            continue
        i = int(parts[0])
        idx.setdefault(i, {})[parts[1]] = np.asarray(v, dtype=np.float32)
    lin, ln = [], []
    for i in sorted(idx):
        w = idx[i].get('weight')
        if w is None:
            continue
        if w.ndim == 2:
            lin.append((w, idx[i].get('bias')))
        else:
            ln.append((w, idx[i].get('bias')))
    return lin, ln


def _fill_mlp(params, denses_torch, lns_torch):
    """Fill an MLP-style flax params tree: k-th flax Dense<->k-th torch Linear (kernel=weight.T),
    k-th flax LayerNorm<->k-th torch LN (scale=weight). Robust to flax naming (matches by creation order)."""
    dense_paths = [k for k, v in _ordered(params) if isinstance(v, dict) and 'kernel' in v]
    ln_paths = [k for k, v in _ordered(params) if isinstance(v, dict) and 'scale' in v]
    assert len(dense_paths) == len(denses_torch), \
        f"Dense count mismatch flax {len(dense_paths)} vs torch {len(denses_torch)}: {dense_paths}"
    assert len(ln_paths) == len(lns_torch), \
        f"LayerNorm count mismatch flax {len(ln_paths)} vs torch {len(lns_torch)}: {ln_paths}"
    for p, (w, b) in zip(dense_paths, denses_torch):
        assert params[p]['kernel'].shape == w.T.shape, f"{p} kernel {params[p]['kernel'].shape} vs {w.T.shape}"
        params[p]['kernel'] = w.T
        params[p]['bias'] = b
    for p, (w, b) in zip(ln_paths, lns_torch):
        params[p]['scale'] = w
        params[p]['bias'] = b
    return params


def convert_mlp(sd, template, prefix='net'):
    import copy
    params = copy.deepcopy(template)
    lin, ln = _torch_layers(sd, prefix)
    return _fill_mlp(params, lin, ln)


def convert_kernel(sd, template):
    """MoG: backbone (MLP) under flax 'backbone_*' + a 'head' Dense. torch: backbone.* + head.*."""
    import copy
    params = copy.deepcopy(template)
    # backbone leaves (flax names backbone_<i>) in creation order
    bb = {k: v for k, v in params.items() if k.startswith('backbone')}
    bb_dense = [k for k, v in _ordered(bb) if 'kernel' in v]
    bb_ln = [k for k, v in _ordered(bb) if 'scale' in v]
    lin, lns = _torch_layers(sd, 'backbone')
    assert len(bb_dense) == len(lin), f"kernel backbone Dense {len(bb_dense)} vs torch {len(lin)}"
    assert len(bb_ln) == len(lns), f"kernel backbone LN {len(bb_ln)} vs torch {len(lns)}"
    for p, (w, b) in zip(bb_dense, lin):
        assert params[p]['kernel'].shape == w.T.shape, f"{p} {params[p]['kernel'].shape} vs {w.T.shape}"
        params[p]['kernel'] = w.T; params[p]['bias'] = b
    for p, (w, b) in zip(bb_ln, lns):
        params[p]['scale'] = w; params[p]['bias'] = b
    hw = np.asarray(sd['head.weight'], dtype=np.float32)
    assert params['head']['kernel'].shape == hw.T.shape, f"head {params['head']['kernel'].shape} vs {hw.T.shape}"
    params['head']['kernel'] = hw.T
    params['head']['bias'] = np.asarray(sd['head.bias'], dtype=np.float32)
    return params


def _attn(sd, prefix, n_heads, d_model):
    """nn.MultiheadAttention -> flax MultiHeadDotProductAttention {query,key,value,out}."""
    hd = d_model // n_heads
    inw = np.asarray(sd[f'{prefix}.in_proj_weight'], dtype=np.float32)   # (3d, d)
    inb = np.asarray(sd[f'{prefix}.in_proj_bias'], dtype=np.float32)     # (3d,)
    Wq, Wk, Wv = inw[:d_model], inw[d_model:2 * d_model], inw[2 * d_model:]
    bq, bk, bv = inb[:d_model], inb[d_model:2 * d_model], inb[2 * d_model:]
    ow = np.asarray(sd[f'{prefix}.out_proj.weight'], dtype=np.float32)   # (d, d)
    ob = np.asarray(sd[f'{prefix}.out_proj.bias'], dtype=np.float32)     # (d,)
    def qkv(W, b):
        return {'kernel': W.T.reshape(d_model, n_heads, hd), 'bias': b.reshape(n_heads, hd)}
    return {
        'query': qkv(Wq, bq), 'key': qkv(Wk, bk), 'value': qkv(Wv, bv),
        'out': {'kernel': ow.T.reshape(n_heads, hd, d_model), 'bias': ob},
    }


def convert_planner(sd, template, n_heads, d_model, depth):
    import copy
    p = copy.deepcopy(template)
    def lin(name):  # torch Linear weight (out,in) -> (kernel=W.T, bias)
        return np.asarray(sd[f'{name}.weight'], dtype=np.float32).T, np.asarray(sd[f'{name}.bias'], dtype=np.float32)

    k, b = lin('x_proj'); p['x_proj']['kernel'], p['x_proj']['bias'] = k, b
    k, b = lin('map_emb.0'); p['map_emb_0']['kernel'], p['map_emb_0']['bias'] = k, b
    k, b = lin('map_emb.2'); p['map_emb_2']['kernel'], p['map_emb_2']['bias'] = k, b
    # map_noise (FourierEmbedding): freqs buffer + 2 Denses (mlp.0, mlp.2)
    p['map_noise']['freqs'] = np.asarray(sd['map_noise.freqs'], dtype=np.float32)
    k, b = lin('map_noise.mlp.0'); p['map_noise']['Dense_0']['kernel'], p['map_noise']['Dense_0']['bias'] = k, b
    k, b = lin('map_noise.mlp.2'); p['map_noise']['Dense_1']['kernel'], p['map_noise']['Dense_1']['bias'] = k, b
    for i in range(depth):
        blk = p[f'blocks_{i}']
        # adaLN_modulation.1 -> Dense_0 ; mlp.0 -> Dense_1 ; mlp.3 -> Dense_2
        k, b = lin(f'blocks.{i}.adaLN_modulation.1'); blk['Dense_0']['kernel'], blk['Dense_0']['bias'] = k, b
        k, b = lin(f'blocks.{i}.mlp.0'); blk['Dense_1']['kernel'], blk['Dense_1']['bias'] = k, b
        k, b = lin(f'blocks.{i}.mlp.3'); blk['Dense_2']['kernel'], blk['Dense_2']['bias'] = k, b
        blk['MultiHeadDotProductAttention_0'] = _attn(sd, f'blocks.{i}.attn', n_heads, d_model)
    k, b = lin('final_layer.adaLN_modulation.1')
    p['final_layer']['Dense_0']['kernel'], p['final_layer']['Dense_0']['bias'] = k, b
    k, b = lin('final_layer.linear')
    p['final_layer']['Dense_1']['kernel'], p['final_layer']['Dense_1']['bias'] = k, b
    return p


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--torch-repo', default=os.path.expanduser('~/Online-Diffusion-Planning'))
    ap.add_argument('--src', default='checkpoints')
    ap.add_argument('--dst', default='Finetuning')
    args = ap.parse_args()

    repo = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, repo); sys.path.insert(0, os.path.join(repo, 'Pretrain'))
    import jax, jax.numpy as jnp, flax
    from Pretrain.Rewards.nets import SimpleReward
    from Pretrain.Critic.nets import Critic
    from Pretrain.Transition_Kernel.Kernel_Net import MoGTransitionKernel
    from Pretrain.Planners.Backbone.Dit import DiT1d
    from Finetuning.utils import get_env
    import torch

    _, OBS, ACT = get_env('cube', 'single')
    DIMS = {'obs': int(OBS), 'act': int(ACT)}
    print(f"[dims] obs={OBS} act={ACT}")

    rng = np.random.default_rng(0)
    INS = {
        'obs': rng.standard_normal((4, OBS)).astype(np.float32),
        'act': rng.standard_normal((4, ACT)).astype(np.float32),
        's': rng.standard_normal((4, OBS)).astype(np.float32),
        'a': rng.standard_normal((4, ACT)).astype(np.float32),
        'x': rng.standard_normal((2, 8, OBS + ACT)).astype(np.float32),       # (b, horizon, in_dim)
        'noise': rng.standard_normal((2,)).astype(np.float32),
    }

    S = args.src
    # Source files are located by NAME anywhere under --src (recursive) so the staging layout doesn't matter
    # (whatever folder you copied under ODP-jax). Destinations are the pipeline's EXISTING checkpoint dirs.
    def find_src(fname):
        hits = sorted(glob.glob(os.path.join(S, '**', fname), recursive=True))
        if len(hits) == 0:
            raise FileNotFoundError(
                f"'{fname}' not found under --src '{S}'. Point --src at the folder that holds the torch checkpoints.")
        if len(hits) > 1:
            raise RuntimeError(f"'{fname}' is ambiguous under '{S}' ({hits}). Use a clean staging dir.")
        return hits[0]

    # (name, kind, model_filename, dst_model_path, stats_filename, dst_stats_path, info)
    jobs = []
    def J(name, kind, mp, dp, sp, sdp, **info):
        jobs.append((name, kind, mp, dp, sp, sdp, info))
    J('reward', 'reward', 'Cube_Single_Task4_Reward_0.pkl',
      f'{args.dst}/Rewards/cube/single/Models/Cube_Single_Task4_Reward_0.pkl',
      'Cube_Single_Task4_Reward_stats_0.pkl',
      f'{args.dst}/Rewards/cube/single/Stats/Cube_Single_Task4_Reward_stats_0.pkl',
      hid=REWARD_HID, layers=REWARD_LAYERS)
    J('critic', 'critic', 'Cube_SinglePlay_task4_Critic_0.pkl',
      f'{args.dst}/Critics/cube/single-play/Models/Cube_SinglePlay_task4_Critic_0.pkl',
      'Cube_SinglePlay_task4_Critic_stats_0.pkl',
      f'{args.dst}/Critics/cube/single-play/Stats/Cube_SinglePlay_task4_Critic_stats_0.pkl',
      hid=CRITIC_HID, layers=CRITIC_LAYERS)
    for i in range(10):
        J(f'kernel{i}', 'kernel', f'Cube_Single_Kernel_{i}.pkl',
          f'{args.dst}/Kernels/cube/single/Models/0/Cube_Single_Kernel_{i}.pkl',
          None, None,
          modes=KERNEL_MODES, layers=KERNEL_LAYERS, hid=KERNEL_HID, noise=KERNEL_NOISE)
    J('kernel_stats', 'copystats', None, None,
      'Cube_Single_Kernel_stats_0.pkl',
      f'{args.dst}/Kernels/cube/single/Stats/Cube_Single_Kernel_stats_0.pkl')
    J('planner', 'planner', 'Cube_SinglePlay_task4_Planner_0.pt',
      f'{args.dst}/Planners/cube/single-play/Cube_SinglePlay_task4_Planner_0.pt',
      'Cube_SinglePlay_task4_Planner_stats.pkl',
      # planner STATS are read by Planner_Processor from PRETRAIN_DIR/Planners/.../Stats (NOT Finetuning),
      # so they MUST land under Pretrain/ to override the JAX-pretrain normalization for the converted planner.
      'Pretrain/Planners/cube/single-play/Stats/Cube_SinglePlay_task4_Planner_stats.pkl',
      emb=PLAN_EMB, dmodel=PLAN_DMODEL, heads=PLAN_HEADS, depth=PLAN_DEPTH, emb_type=PLAN_EMB_TYPE)

    # 1) torch ground-truth (subprocess, cwd=torch_repo)
    ref_nets = {}
    for name, kind, mp, dp, sp, sdp, info in jobs:
        if kind in ('reward', 'critic', 'kernel', 'planner'):
            ref_nets[name] = {'kind': kind, 'path': find_src(mp), **info}
    print(f"[torch] computing reference forwards for {len(ref_nets)} nets via subprocess ...")
    refs = torch_refs(args.torch_repo, ref_nets, INS, DIMS)

    def init_params(module, *call_args):
        # to_state_dict -> a MUTABLE nested dict (flax .init returns an immutable FrozenDict; the convert_*
        # fillers assign into this tree). Leaves are overwritten by the torch weights, so its init dtype is moot.
        return flax.serialization.to_state_dict(module.init(jax.random.PRNGKey(0), *call_args)['params'])

    def to_np(tree):
        return jax.tree_util.tree_map(lambda a: np.asarray(a, dtype=np.float32), tree)

    def parity(name, flax_out, ref):
        flax_out = np.asarray(flax_out, dtype=np.float32)
        d = float(np.max(np.abs(flax_out - ref)))
        ok = d <= TOL
        print(f"  [{name:11s}] parity max|diff|={d:.2e}  {'OK' if ok else 'FAIL <<<<<<'}")
        return ok

    def save_pickle(path, obj):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def copy_stats(src, dst):
        # stats are plain pickles referencing a stats class; loadable here (this repo's Pretrain is on path).
        try:
            with open(src, 'rb') as f:
                obj = pickle.load(f)
        except Exception as e:
            print(f"  [stats] WARN could not unpickle {src}: {type(e).__name__}: {e}; copying bytes verbatim")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, 'rb') as a, open(dst, 'wb') as b:
                b.write(a.read())
            return
        save_pickle(dst, obj)

    n_ok = 0
    for name, kind, mp, dp, sp, sdp, info in jobs:
        if kind == 'copystats':
            copy_stats(find_src(sp), sdp); print(f"  [{name}] stats copied"); continue
        sd = torch.load(find_src(mp), map_location='cpu')
        if kind == 'planner':
            sd = sd['ema']
        sd = {k: np.asarray(v.detach().cpu().numpy() if hasattr(v, 'detach') else v) for k, v in sd.items()}

        if kind == 'reward':
            mod = SimpleReward(OBS, ACT, info['hid'], info['layers'])
            tmpl = init_params(mod, jnp.zeros((1, OBS)), jnp.zeros((1, ACT)))
            params = convert_mlp(sd, to_np(tmpl), 'net')
            out = mod.apply({'params': params}, jnp.asarray(INS['obs']), jnp.asarray(INS['act']))
        elif kind == 'critic':
            mod = Critic(OBS, info['hid'], info['layers'])
            tmpl = init_params(mod, jnp.zeros((1, OBS)))
            params = convert_mlp(sd, to_np(tmpl), 'net')
            out = mod.apply({'params': params}, jnp.asarray(INS['obs']))
        elif kind == 'kernel':
            mod = MoGTransitionKernel(OBS, ACT, num_modes=info['modes'], num_hidden_layers=info['layers'],
                                      hidden_dim=info['hid'], noise_floor=info['noise'])
            tmpl = init_params(mod, jnp.zeros((1, OBS)), jnp.zeros((1, ACT)))
            params = convert_kernel(sd, to_np(tmpl))
            mu, log_std, w = mod.apply({'params': params}, jnp.asarray(INS['s']), jnp.asarray(INS['a']))
            out = np.concatenate([np.asarray(mu).reshape(mu.shape[0], -1),
                                  np.asarray(log_std).reshape(log_std.shape[0], -1), np.asarray(w)], -1)
        elif kind == 'planner':
            mod = DiT1d(in_dim=OBS + ACT, emb_dim=info['emb'], d_model=info['dmodel'],
                        n_heads=info['heads'], depth=info['depth'], timestep_emb_type=info['emb_type'])
            tmpl = init_params(mod, jnp.zeros((1, 8, OBS + ACT)), jnp.zeros((1,)))
            params = convert_planner(sd, to_np(tmpl), info['heads'], info['dmodel'], info['depth'])
            out = mod.apply({'params': params}, jnp.asarray(INS['x']), jnp.asarray(INS['noise']))
        else:
            continue

        if not parity(name, out, refs[name]):
            print(f"  [{name}] NOT written (parity failed) — fix mapping before using.")
            continue

        # save in the loader's format
        if kind == 'planner':
            data = {'dataset_name': 'cube', 'specific_dataset': 'single-play', 'task_id': TASK_ID,
                    'step': 0, 'ema': flax.serialization.to_state_dict(params)}
            save_pickle(dp, data)
        else:
            save_pickle(dp, flax.serialization.to_state_dict(params))
        if sp and sdp:
            copy_stats(find_src(sp), sdp)
        n_ok += 1
        print(f"  [{name}] -> {dp}")

    n_models = sum(1 for j in jobs if j[1] in ('reward', 'critic', 'kernel', 'planner'))
    print(f"\nDONE: {n_ok}/{n_models} nets converted + parity-verified. "
          f"{'ALL GOOD.' if n_ok == n_models else 'Some FAILED — see above.'}")


if __name__ == '__main__':
    main()
