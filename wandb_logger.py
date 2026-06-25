"""Lightweight, dependency-optional Weights & Biases logging helper for the JAX/Flax ODP port.

Design goals (so adding `wlog(...)` calls anywhere is always safe):
  * **No-op unless a run is active.** If `wandb` is not installed, or no run has been started via
    `wandb_init(...)`, every call here is a cheap no-op. Default training behavior is therefore
    completely unchanged when you don't opt into wandb.
  * **One global run.** Training functions just call `wlog({...}, step=...)`; the entry script / runner
    owns the run lifecycle with `wandb_init(...)` / `wandb_finish()`.
  * **Disable switch.** Set the environment variable `ODP_WANDB=0` (or `WANDB_DISABLED=true`) to force
    every helper into no-op mode even if you call `wandb_init`.

Typical usage in a runner / entry script:

    from wandb_logger import wandb_init, wandb_finish
    wandb_init(project='odp-cube-double', name='pretrain', config={...})
    ...                                   # run a training stage; its inner wlog(...) calls now log
    wandb_finish()

Typical usage inside a training loop (already wired into the 5 stages):

    from wandb_logger import wlog
    wlog({'pretrain/loss': loss, 'pretrain/lr': lr}, step=step)
"""

import os

try:  # wandb is optional — import lazily and degrade gracefully.
    import wandb as _wandb
except Exception:  # pragma: no cover - import guard
    _wandb = None

# Module-global handle to the active run (None when inactive).
_RUN = None


def _enabled():
    """True only when wandb is importable, a run is active, and not disabled via env."""
    if _wandb is None or _RUN is None:
        return False
    if os.environ.get('ODP_WANDB', '1') == '0':
        return False
    if os.environ.get('WANDB_DISABLED', '').lower() in ('1', 'true', 'yes'):
        return False
    return True


def wandb_available():
    """Whether the `wandb` package could be imported at all."""
    return _wandb is not None


def wandb_init(project='odp', name=None, group=None, config=None, mode=None, **kwargs):
    """Start a wandb run (idempotent-ish: finishes any prior run first).

    Returns the run object, or None if wandb is unavailable / disabled. Safe to call even when wandb is
    not installed — it simply returns None and all later `wlog` calls stay no-ops.
    """
    global _RUN
    if _wandb is None:
        print('[wandb_logger] wandb not installed; logging disabled (pip install wandb to enable).')
        return None
    if os.environ.get('ODP_WANDB', '1') == '0' or os.environ.get('WANDB_DISABLED', '').lower() in ('1', 'true', 'yes'):
        print('[wandb_logger] wandb disabled via environment variable; logging is a no-op.')
        return None
    # Close a previous run if one is open (e.g. between pipeline stages).
    if _RUN is not None:
        wandb_finish()
    run_mode = mode or os.environ.get('WANDB_MODE', 'online')
    _RUN = _wandb.init(project=project, name=name, group=group, config=config, mode=run_mode,
                       reinit=True, **kwargs)
    return _RUN


def wlog(metrics, step=None, commit=None):
    """Log a dict of scalar metrics. No-op unless a run is active.

    Args:
        metrics: dict of {str: number}. Non-finite / non-scalar values are passed through to wandb as-is.
        step: optional global step (monotonic per run recommended).
        commit: optional wandb commit flag.
    """
    if not _enabled():
        return
    try:
        if step is not None:
            _RUN.log(dict(metrics), step=int(step), commit=commit)
        else:
            _RUN.log(dict(metrics), commit=commit)
    except Exception as e:  # pragma: no cover - never let logging crash training
        print(f'[wandb_logger] wlog failed (ignored): {e}')


def wandb_config_update(config):
    """Merge extra keys into the active run's config. No-op unless a run is active."""
    if not _enabled():
        return
    try:
        _RUN.config.update(dict(config), allow_val_change=True)
    except Exception as e:  # pragma: no cover
        print(f'[wandb_logger] config update failed (ignored): {e}')


def wandb_finish():
    """Finish the active run (no-op if none). Call between pipeline stages and at the end."""
    global _RUN
    if _wandb is None or _RUN is None:
        _RUN = None
        return
    try:
        _wandb.finish()
    except Exception as e:  # pragma: no cover
        print(f'[wandb_logger] finish failed (ignored): {e}')
    finally:
        _RUN = None
