"""
diag_reward_targets.py -- pinpoint WHY two machines get different initial reward loss.

The initial training loss is (almost) deterministic: at init the net outputs ~0, so
   loss_0  ~=  mean( smooth_l1(0, target) )
i.e. it is a pure function of the TARGET LABELS, not of hardware/seed.
So if two people see 6.0 vs 0.66, the labels differ -> data or config differs.

Run on BOTH machines and diff the output:
    cd <repo>/Pretrain && python diag_reward_targets.py
"""
import os, sys, hashlib
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- the exact knobs train_reward_script.py passes (EDIT to match your script) ----
DATASET_NAME     = 'cube'
SPECIFIC_DATASET = 'single'
TASK_ID          = 4
TRAJ_LENGTH      = None      # <-- read this off YOUR train_reward_script.py
SIGMA            = 4.0       # <-- ditto
TARGET_REWARD    = 500.0     # <-- ditto
ALPHA            = None

print("=" * 78)
print("PROVENANCE")
print("=" * 78)
import scipy, torch
print(f"python           : {sys.version.split()[0]}")
print(f"numpy / scipy    : {np.__version__} / {scipy.__version__}")
print(f"torch            : {torch.__version__}")
try:
    import ogbench
    print(f"ogbench          : {getattr(ogbench, '__version__', 'unknown')}  @ {os.path.dirname(ogbench.__file__)}")
except Exception as e:
    print(f"ogbench          : import failed ({e})")

# raw .npz files actually consumed (dataset provenance -- catches a re-downloaded/updated dataset)
for d in (os.path.expanduser(os.environ.get("OGBENCH_DATASET_DIR", "~/.ogbench")),):
    for root, _, files in os.walk(d):
        for fn in sorted(files):
            if fn.startswith("cube-single") and fn.endswith(".npz"):
                p = os.path.join(root, fn)
                h = hashlib.md5(open(p, "rb").read(1 << 20)).hexdigest()[:12]  # first 1MB is enough to fingerprint
                print(f"dataset file     : {fn:44s} {os.path.getsize(p):>12,d} B  md5[1MB]={h}")

print()
print("=" * 78)
print(f"CONFIG USED HERE : traj_length={TRAJ_LENGTH}  sigma={SIGMA}  target_reward={TARGET_REWARD}  alpha={ALPHA}")
print("=" * 78)

# ---- build the labels through the repo's OWN code path ----
from Rewards.Reward_Backbone import Train_Dataset, RewardDataset

trajs, reward_name, obs_dim, act_dim = Train_Dataset(
    DATASET_NAME, SPECIFIC_DATASET, TASK_ID, None, TRAJ_LENGTH)

lens = np.array([len(t['actions']) for t in trajs])
raw = np.concatenate([np.asarray(t['rewards']).ravel() for t in trajs])
print(f"trajectories     : {len(trajs)}")
print(f"traj length      : mean={lens.mean():.1f}  min={lens.min()}  max={lens.max()}  total_steps={lens.sum():,d}")
print(f"RAW labels       : nonzero={np.count_nonzero(raw):,d} / {raw.size:,d} "
      f"({100.0*np.count_nonzero(raw)/raw.size:.4f}%)  unique={np.unique(raw)[:6]}")

ds = RewardDataset(trajs, reward_name, sigma=SIGMA, alpha=ALPHA, target_reward=TARGET_REWARD)
r = np.array([float(t[2]) for t in ds.transitions])

nz = r[r != 0]
print()
print("=" * 78)
print("FINAL TARGETS fed to smooth_l1 (after boost_signal + gaussian_filter1d)")
print("=" * 78)
print(f"samples          : {r.size:,d}")
print(f"nonzero          : {nz.size:,d}  ({100.0*nz.size/r.size:.4f}%)   <-- density, set by truncate & traj_length")
print(f"mean |target|    : {np.abs(r).mean():.6f}")
print(f"max target       : {r.max():.4f}    <-- peak height, set by target_reward & sigma")
print(f"mean of nonzero  : {nz.mean() if nz.size else 0:.4f}")

# predicted initial loss: net outputs ~0 at init, smooth_l1(0, t) with beta=1
pred = np.where(np.abs(r) < 1.0, 0.5 * r ** 2, np.abs(r) - 0.5).mean()
print()
print(f">>> PREDICTED initial loss (smooth_l1 vs 0) = {pred:.4f}")
print(">>> Compare with the FIRST 'Step ..., loss ...' line your training prints.")
print(">>> If two machines differ here, the LABELS differ -> data or config, not hardware.")
