'''SDE diffusion-planner pretraining trainer — JAX/Flax (FQL-style) port of the original PyTorch module.'''
import os
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
from typing import Optional
from typing import Dict
import copy
import json
import pickle

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import optax
import wandb

from .utils import cosine_alpha_sigma, cosine_beta, EMA, cycle
from Dataset import get_env, determine_stride
from .Dit import DiT1d
from .UNet import TemporalUnet
from Dataset import get_PlannerName, PlannerDataset, PlannerDataset_Rollout
from .utils import LossTracker, get_pretrained_planner, getName

from flax_utils import TrainState


class SDETrainer:
    def __init__(
        self,
        dataset_name,
        specific_dataset,
        task_id,
        horizon,
        backbone_name,
        num_steps = 1000000,
        batch_size = 128,
        lr=2e-4,
        device=None,
        update_ema_every = 2,
        step_start_ema = 1000,
        gradient_accumulate_every=2,
        ema_decay=0.9999,
        save_freq= 200000,
        log_freq = 10,
        s: float = 0.008,                  # cosine offset
        weight_type: str = 'sigma2',         # {"one", "sigma2", "beta"}
        eps: float = 1e-5,               # clamp for t, ᾱ stability
        stride: Optional[int] = 1,
        *,
        seed: int = 0,                   # API-CHANGE: JAX has no global RNG; seed the trainer's key.
    ):
        self.device = device
        # JAX has no global RNG: hold the trainer's PRNG key and thread it through the loop.
        self.rng = jax.random.PRNGKey(seed)
        self.dataset_name = dataset_name
        self.specific_dataset = specific_dataset
        self.task_id = task_id
        _, self.state_dim, self.action_dim = get_env(self.dataset_name, self.specific_dataset)
        if(determine_stride(self.dataset_name, self.specific_dataset)):
            self.Dimension = self.state_dim
            self.stride = stride
        else:
            self.Dimension = self.state_dim + self.action_dim
            self.stride = 1
        self.backbone_name = backbone_name
        self.horizon = horizon
        self.s = s
        self.weight_type = weight_type
        self.eps = eps
        self.update_ema_every = update_ema_every
        self.gradient_accumulate_every = gradient_accumulate_every
        self.lr = lr
        self.step_start_ema = step_start_ema
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_freq = log_freq
        self.save_freq = save_freq
        # Build the (frozen) linen model_def and initialize its params into a TrainState.
        self.backbone_selection()
        self.model_name = get_PlannerName(self.dataset_name, self.specific_dataset, self.task_id)
        # torch deep-copied the model for the EMA target; here we keep an EMA param pytree.
        self.reset_parameters()
        self.ema = EMA(ema_decay)
        # torch: AdamW(lr, weight_decay=1e-5) + CosineAnnealingLR over num_steps. optax folds the
        # cosine schedule into the optimizer (alpha=0 reproduces CosineAnnealingLR to 0). Grad clip
        # (clip_grad_norm_(1.0)) is chained before AdamW (CONVERSION_GUIDE §5/§6).
        #self.optim = torch.optim.Adam(self.model.parameters(), self.lr)
        schedule = optax.cosine_decay_schedule(self.lr, self.num_steps, alpha=0.0)
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule, weight_decay=1e-5),
        )
        self.train_state = TrainState.create(self.model_def, self.params, tx=tx)
        # Keep `self.schedule` so save_hyperparameters can report the schedule horizon.
        self.schedule = schedule
        #self.logdir = f"./{self.dataset_name}_{self.specific_dataset}_checkpoints/"
        self.logdir = os.path.join(
            REPO_ROOT,
            "Pretrain",
            f"{self.dataset_name}_{self.specific_dataset}"
             + (f"_task{self.task_id}" if self.task_id is not None else "")
             + "_checkpoints",
        )
        self.loss_tracker = LossTracker(save_dir="./logs/")

    def save_hyperparameters(self, filepath: Optional[str] = None):
        if filepath is None:
            os.makedirs(f"./Pretrain/Planners/args/{self.dataset_name}/{self.specific_dataset}/", exist_ok=True)
            filepath = f"./Pretrain/Planners/args/{self.dataset_name}/{self.specific_dataset}/hyperparameters.json"

        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif obj is None:
                return None
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
                return str(obj)
            return obj

        # Get optimizer and scheduler info (optax-side, matching the torch AdamW + CosineAnnealingLR).
        optimizer_type = 'AdamW'
        optimizer_params = {
             'type': optimizer_type,
             'lr': self.lr,
             'weight_decay': 1e-5,
         }

        scheduler_type = 'CosineAnnealingLR'
        scheduler_params = {
             'type': scheduler_type,
             'T_max': self.num_steps,
        }

       # Get model architecture info
        model_info = {
             'backbone_name': self.backbone_name,
             'state_dim': int(self.state_dim),
             'action_dim': int(self.action_dim),
             'horizon': self.horizon,
         }

        # Add backbone-specific parameters if available
        if hasattr(self.model, 'in_dim'):
              model_info['model_in_dim'] = int(self.model.in_dim)
        if hasattr(self.model, 'emb_dim'):
              model_info['model_emb_dim'] = int(self.model.emb_dim)
        if hasattr(self.model, 'd_model'):
              model_info['model_d_model'] = int(self.model.d_model)
        if hasattr(self.model, 'n_heads'):
              model_info['model_n_heads'] = int(self.model.n_heads)
        if hasattr(self.model, 'depth'):
              model_info['model_depth'] = int(self.model.depth)

        # Compile all hyperparameters
        hyperparams = {
           'env_details': {
                'dataset_name': self.dataset_name,
                'specific_dataset': self.specific_dataset,
                'state_dim': int(self.state_dim),
                'action_dim': int(self.action_dim),
            },
           'model_architecture': model_info,
           'training_hyperparameters': {
                'horizon': self.horizon,
                'num_steps': self.num_steps,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'gradient_accumulate_every': self.gradient_accumulate_every,
                'optimizer': optimizer_params,
                'scheduler': scheduler_params,
            },
           'ema_hyperparameters': {
                'ema_decay': self.ema.beta,
                'update_ema_every': self.update_ema_every,
                'step_start_ema': self.step_start_ema,
            },
           'training_config': {
                 'save_freq': self.save_freq,
                 'log_freq': self.log_freq,
                 'logdir': self.logdir,
                 'model_name': self.model_name,
             },
           'sde_hyperparameters': {
                's': self.s,
                'weight_type':self.weight_type,
                'eps': self.eps,
            }
        }

        # Handle numpy arrays and other non-JSON-serializable types
        hyperparams = convert_to_json_serializable(hyperparams)

        # Save with pretty printing (indent=4 makes it human-readable)
        with open(filepath, 'w') as f:
            json.dump(hyperparams, f, indent=4, sort_keys=False)

        print(f"Pretraining hyperparameters saved to {filepath}", flush=True)

    def backbone_selection(self):
         # In JAX the model is a (frozen) linen module definition; params live in the TrainState.
         if(self.backbone_name == 'transformer'):
              self.model_def = DiT1d(
                   in_dim = self.Dimension, emb_dim = 128,
                   d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
         elif(self.backbone_name == 'unet'):
              self.model_def = TemporalUnet(self.horizon, self.Dimension)
         # Keep `self.model` as a public alias of the model definition (save_hyperparameters reads its
         # attributes). The torch `self.model` was an nn.Module; here it is the linen model_def.
         self.model = self.model_def
         self.rng, init_rng, dropout_rng = jax.random.split(self.rng, 3)
         self.params = self.model_def.init(
             {'params': init_rng, 'dropout': dropout_rng}, *self._example_inputs()
         )['params']

    def _example_inputs(self):
        '''Example inputs for linen parameter init, matching how the model is called in `Loss`.'''
        x = jnp.zeros((1, self.horizon, self.Dimension), dtype=jnp.float32)
        t = jnp.zeros((1,), dtype=jnp.float32)
        if self.backbone_name == 'transformer':
            return (x, t)
        # TemporalUnet.__call__(x, conditions, time): torch called the model as model(x, t); the unet
        # also consumes conditioning. Pass `None` conditions (unused by the unet body) to match arity.
        return (x, None, t)

    def _apply_model(self, params, x, t, *, rng=None):
        '''Call the backbone via the TrainState apply (threads a dropout key when needed).'''
        rngs = {} if rng is None else {'dropout': rng}
        if self.backbone_name == 'transformer':
            return self.train_state(x, t, params=params, rngs=rngs)
        return self.train_state(x, None, t, params=params, rngs=rngs)

    def reset_parameters(self):
        # torch copied the online state_dict into the EMA model; here the EMA params start as a copy
        # of the online params pytree.
        self.ema_params = jax.tree_util.tree_map(lambda p: p, self.params)

    def step_ema(self):
        if self.step < self.step_start_ema:
            # torch reset the EMA module to the *current* online model here; copy the current online
            # params (train_state may have advanced past the init params).
            self.ema_params = jax.tree_util.tree_map(lambda p: p, self.train_state.params)
            return
        # torch mutated the EMA module in place; the JAX EMA helper takes and returns param pytrees.
        self.ema_params = self.ema.update_model_average(self.ema_params, self.train_state.params)


    def save(self, epoch):
        # TODO(checkpoint-bridge): the torch trainer saved a dict with the EMA module `state_dict()`
        # to a `.pt` file (`torch.save({'ema': self.ema_model.state_dict(), ...}, path)`) so that
        # get_pretrained_planner could reload it. Here we serialize the EMA param pytree with
        # flax.serialization (pickled), keeping the ODP `{model_name}_{epoch}.pt` / `{model_name}_0.pt`
        # path layout byte-for-byte. Downstream torch ingest must be migrated to read this format
        # (see CONVERSION_GUIDE §10).
        data = {
              'dataset_name': self.dataset_name,
              'specific_dataset': self.specific_dataset,
              'task_id': self.task_id,                     # NEW
              'step': self.step,
              'ema': flax.serialization.to_state_dict(self.ema_params),
        }
        if epoch == self.num_steps:
            file_name = f"{self.model_name}_0.pt"
            save_dir = os.path.join(
                REPO_ROOT, "Finetuning", "Planners",
                self.dataset_name, self.specific_dataset,
            )
        else:
            file_name = f"{self.model_name}_{epoch}.pt"
            save_dir = self.logdir
        os.makedirs(save_dir, exist_ok=True)
        savepath = os.path.join(save_dir, file_name)
        with open(savepath, 'wb') as f:
            pickle.dump(data, f)
        print(f'Saved model to {savepath}', flush=True)

    @staticmethod
    def _batches(dataset, batch_size, shuffle=True, drop_last=True):
        '''fql-style numpy batch generator over a dataset's __len__/__getitem__ (replaces DataLoader).

        Yields `(traj, cond)` stacked numpy arrays. Host-side numpy RNG for shuffling (CONVERSION_GUIDE
        §13: data shuffling stays numpy; only model stochasticity needs jax keys).
        '''
        n = len(dataset)
        order = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            idxs = order[start:start + batch_size]
            if drop_last and len(idxs) < batch_size:
                break
            trajs = np.stack([np.asarray(dataset[i][0], dtype=np.float32) for i in idxs], axis=0)
            conds = np.stack([np.asarray(dataset[i][1], dtype=np.float32) for i in idxs], axis=0)
            yield trajs, conds

    def train(self):
        print(self.device)
        dataset = PlannerDataset(self.dataset_name, self.specific_dataset, self.task_id, self.horizon,
                                 self.state_dim, self.action_dim, self.stride)
        # NOTE: cycle(gen) hangs after one epoch because `_batches` is a ONE-SHOT generator (once exhausted,
        # `while True: for x in dead_gen` spins forever yielding nothing). Build an infinite loader that
        # RE-CREATES (reshuffles) `_batches` each epoch instead.
        def _infinite_batches():
            while True:
                for batch in self._batches(dataset, self.batch_size, shuffle=True):
                    yield batch
        dataloader = _infinite_batches()
        print(f"Training planner for {self.dataset_name}-{self.specific_dataset} Dataset")
        print(f"Backbone:{self.backbone_name}, Horizon: {self.horizon}, Epochs: {self.num_steps}, Batch Size: {self.batch_size}, Learning Rate; {self.lr}")

        # Save hyperparameters at the start of training
        self.save_hyperparameters()

        # The EMA params are a frozen target (torch set requires_grad_(False) on the EMA module);
        # in JAX they are simply never passed as grad params.
        self.step = 0
        total_loss = 0

        # JIT the per-step update over `gradient_accumulate_every` micro-batches (fql pattern).
        @jax.jit
        def _update(train_state, trajs, conds, rng):
            def loss_fn(params):
                acc = 0.0
                cur = rng
                for i in range(self.gradient_accumulate_every):
                    cur, sub = jax.random.split(cur)
                    loss = self._loss(params, trajs[i], conds[i], rng=sub)
                    loss = loss / self.gradient_accumulate_every
                    acc = acc + loss
                return acc, {'loss': acc}

            grads, info = jax.grad(loss_fn, has_aux=True)(train_state.params)
            new_train_state = train_state.apply_gradients(grads=grads)
            return new_train_state, info

        while(self.step < self.num_steps):
            # Gather `gradient_accumulate_every` micro-batches, stacking into leading-axis arrays.
            trajs = []
            conds = []
            for i in range(self.gradient_accumulate_every):
                traj, cond = next(dataloader)
                trajs.append(jnp.asarray(traj))
                conds.append(jnp.asarray(cond))
            trajs = jnp.stack(trajs, axis=0)
            conds = jnp.stack(conds, axis=0)

            self.rng, step_rng = jax.random.split(self.rng)
            self.train_state, info = _update(self.train_state, trajs, conds, step_rng)
            loss = info['loss']
            total_loss += float(loss)
            cur_lr = float(self.schedule(self.train_state.step - 1))
            self.loss_tracker.log_loss(self.step, float(loss), cur_lr)

            if ((self.step % self.update_ema_every) == 0):
                self.step_ema()

            if ((self.step % self.log_freq) == 0):
                print(f"step {self.step} loss {total_loss/self.log_freq}")
                if wandb.run is not None:
                    wandb.log({'pretrain/loss': float(loss), 'pretrain/lr': cur_lr,
                               'pretrain/avg_loss': total_loss / self.log_freq}, step=self.step)
                total_loss = 0

            if ((self.step % self.save_freq == 0) and (self.step!=0)):
                self.save(self.step)
                self.loss_tracker.save_logs(f"{self.model_name}_logs.pkl")
                self.loss_tracker.plot_loss_curve(
                      save_path=f"./plots/{self.model_name}_loss_curve.png",
                      title=f"{self.model_name} Training Loss",
                      show_lr=True,
                      smooth_window=50)

            self.step += 1
        # Final save and plot
        self.save(self.step)
        self.loss_tracker.save_logs(f"{self.model_name}_final_logs.pkl")
        self.loss_tracker.plot_loss_curve(
             save_path=f"./plots/{self.model_name}_final_loss_curve.png",
             title=f"{self.model_name} Training Loss",
             show_lr=True,
             smooth_window=50)

    def selector(self, specific_dataset, times = 1000, *, rng=None):
         dataset = PlannerDataset_Rollout(self.dataset_name, specific_dataset, self.specific_dataset,
                                          self.horizon, self.state_dim, self.action_dim)
         # fql-style numpy batching: iterate batches of 10 (torch used DataLoader(dataset, 10)).
         # torch `N = len(dataloader)` is the number of size-10 batches (default DataLoader drops nothing,
         # but shuffles); keep the per-batch-averaged-then-summed semantics identical.
         N = (len(dataset) + 10 - 1) // 10
         min_Loss = float('inf')
         checkpoint = self.save_freq
         best_checkpoint = 0
         validation_tracker = LossTracker(save_dir="./logs/")
         print(f"Loss of {self.model_name} on {specific_dataset} dataset. Running {times} times for each checkpoints")
         if rng is None:
             rng = self.rng
         while(checkpoint <= self.num_steps):
            self.backbone_selection()
            # TODO(checkpoint-bridge): get_pretrained_planner ingests a torch `.pt` ema state_dict; the
            # state_dict -> flax param-tree remap is not yet ported (see CONVERSION_GUIDE §10). Once it
            # is, the returned params replace `self.train_state.params` below.
            state_dict = get_pretrained_planner(self.dataset_name, self.specific_dataset, checkpoint, self.task_id)
            eval_params = flax.serialization.from_state_dict(self.params, state_dict)
            avg_loss = 0
            for i in range(times):
                total_loss = 0
                for traj, cond in self._batches(dataset, 10, shuffle=True, drop_last=False):
                    rng, sub = jax.random.split(rng)
                    loss = self._loss(eval_params, jnp.asarray(traj), jnp.asarray(cond), rng=sub)
                    total_loss += float(loss)
                Loss = total_loss/N
                avg_loss += Loss
            final_loss = avg_loss/times
            if(final_loss < min_Loss):
                 min_Loss = final_loss
                 best_checkpoint = checkpoint
            print(f"Checkpoint: {checkpoint} Loss: {final_loss}")
            validation_tracker.log_loss(checkpoint, final_loss)
            checkpoint += self.save_freq
         print(f"Best Checkpoint: {best_checkpoint}, Loss: {min_Loss}")
         #self.loss_tracker.save_logs(f"{self.model_name}_{specific_dataset}_validation_loss_curve.pkl")

         validation_tracker.plot_loss_curve(
             save_path=f"./plots/{self.model_name}_{specific_dataset}_validation_loss_curve.png",
             title=f"{self.model_name} {specific_dataset} Validation Loss",
             show_lr=False,
             smooth_window=5)

         return best_checkpoint, min_Loss

    def _loss(self, params, x0, conditions, *, rng=None):                       # (B,H,D)
        # Internal param-dependent loss for jax.grad. `Loss` (public) wraps this with the stored params.
        B, H, D = x0.shape
        mask = jnp.zeros((B, H, D), dtype = jnp.float32)
        y = jnp.zeros((B, H, D), dtype = jnp.float32)
        mask = mask.at[:, 0, :self.state_dim].set(1)
        y = y.at[:, 0, :self.state_dim].set(conditions)

        rng, t_rng, eps_rng = jax.random.split(rng, 3)

        # 1) sample time t ~ U(eps, 1 - eps), per sample (shape: (B,))
        t = jax.random.uniform(t_rng, (B,)) * (1.0 - 2*self.eps) + self.eps

        # 2) α(t), σ(t) from cosine schedule (return 1D tensors, then expand to (B,1,1))
        alpha, sigma = cosine_alpha_sigma(t, self.s)     # (B,), (B,)
        alpha_b = alpha.reshape(B, 1, 1)                 # -> (B,1,1) for broadcasting
        sigma_b = sigma.reshape(B, 1, 1)                 # -> (B,1,1)

        # 3) perturbation
        eps = jax.random.normal(eps_rng, x0.shape).astype(x0.dtype)            # (B,H,D)
        x_t = alpha_b * x0 + sigma_b * eps               # (B,H,D)


        #x_t = apply_conditioning(x_t, conditions, self.state_dim)
        xt_clamped = mask * y + (1 - mask) * x_t
        # 4) analytic Gaussian score target for VP
        target = -(xt_clamped - alpha_b * x0) / ( sigma_b**2 + 1e-8)   # (B,H,D)  (Song et al.) :contentReference[oaicite:2]{index=2}

        # 5) model prediction (must match (B,H,D)); pass per-sample t
        rng, model_rng = jax.random.split(rng)
        pred = self._apply_model(params, xt_clamped, t, rng=model_rng)        # (B,H,D)


        # 6) loss weighting λ(t)
        if self.weight_type == "one":
            lam = jnp.ones(B)                            # classic VP choice
        elif self.weight_type == "sigma2":
            lam = sigma**2                               # common balancing heuristic (more VE-like)
        elif self.weight_type == "beta":
            beta = cosine_beta(t, self.s)                # g(t)^2 = β(t) for VP-SDE
            lam = beta
        else:
            raise ValueError(f"Unsupported weight_type {self.weight_type}")

        # 7) weighted MSE; λ(t) is per-sample => apply after summing over (H,D)
        diff = (pred - target) * (1 - mask)
        mse = (diff**2).sum(axis = (1,2))
        loss = (lam * mse).mean()
        loss = loss/((H*D) - self.state_dim)
        return loss

    def Loss(self, x0, conditions, *, rng=None):                       # (B,H,D)
        # Public loss using the stored (current) params; wraps the param-dependent `_loss`.
        if rng is None:
            self.rng, rng = jax.random.split(self.rng)
        return self._loss(self.train_state.params, x0, conditions, rng=rng)
