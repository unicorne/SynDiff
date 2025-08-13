"""
fine_tune.py — Hyperparameter search wrapper for your Syndiff training script

What this does
- Runs your existing training loop (single-GPU DDP) with different hyperparameters
- Tracks metrics and configs in Weights & Biases (wandb)
- Uses Optuna to search a flexible set of hyperparameters
- Reads validation L1/PSNR from the numpy files your script already saves and reports them to wandb & Optuna

How to use
1) Adjust the USER EDIT section below (paths, project name, trials, search space).
2) Make sure this file lives next to your training script and that you can `import train`.
   If your file is named differently, set TRAIN_MODULE accordingly.
3) Run:  `python fine_tune.py`

Notes
- This runs with `world_size=1` and leverages your script's DDP setup internally.
- Defaults are respected: we only override the params we explicitly set or sample.
- Each trial writes to a unique experiment subfolder under OUTPUT_ROOT.
"""

import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

# =========================
# ===== USER EDITS ========
# =========================

# The module name of your training script (change if your file isn't named train.py)
TRAIN_MODULE = "train"  # e.g., "syndiff_train" if your file is syndiff_train.py

# Data / output paths
INPUT_PATH = "/home/students/studweilc1/SynDiff/my_data_group"
OUTPUT_ROOT = "/home/students/studweilc1/SynDiff/my_results_finetuned"

# W&B settings
WANDB_PROJECT = "syndiff-tuning"
WANDB_ENTITY = None  # set to your team/entity if needed, or leave None
WANDB_GROUP = "default"
WANDB_MODE = "online"  # set to "disabled" to run without logging

# Study settings
N_TRIALS = 20
SEED = 1024
MAX_EPOCHS = 1  # you can lower for quick searches

# Metric to optimize: choose "psnr" (maximize) or "l1" (minimize)
OPTIMIZE_FOR = "psnr"

# Fixed overrides you always want (leave None to use your training default)
FIXED_OVERRIDES = {
    "batch_size": 1,
    "num_timesteps": 4,
    "use_ema": True,
    "save_content": False,
    "no_lr_decay": False,
    "num_epoch": MAX_EPOCHS,  # we'll set this automatically anyway
    "contrast1": "T1_mapping_fl2d",
    "contrast2": "DIXON_T1_mapping_fl2d",
    "progressive": "none",
    "progressive_input": "residual",
    "progressive_combine": "sum",
    "ch_mult": [1, 1, 2, 2, 4, 4],
    "num_channels": 2,          # diffusion NCSNpp sees concat(x_t+1, source) -> 2ch
    "image_size": 256,
    "attn_resolutions": (16,)        
}
# Hyperparameter search space — edit freely
# Only parameters listed here are sampled; everything else keeps defaults or FIXED_OVERRIDES
SEARCH_SPACE = {
    "lr_g": ("log_uniform", 1e-5, 5e-4),
    "lr_d": ("log_uniform", 1e-5, 5e-4),
    "beta1": ("uniform", 0.4, 0.9),
    "beta2": ("uniform", 0.85, 0.999),
    "lambda_l1_loss": ("log_uniform", 1e-2, 1.0),
    #"nz": ("int", 32, 256),
    #"ngf": ("int", 32, 128),
    "num_timesteps": ("int", 2, 8),
    "use_geometric": ("categorical", [False, True]),
    "beta_min": ("log_uniform", 1e-3, 0.5),
    "beta_max": ("log_uniform", 1.0, 40.0),
}

# =========================
# ===== END USER EDITS ====
# =========================

import importlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, asdict, field
from types import SimpleNamespace
from typing import Any, Dict, Tuple
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

import numpy as np

# Optional deps

import optuna
import wandb

api_key = os.getenv("WANDB_API_KEY")
if api_key:
    wandb.login(key=api_key, relogin=True)



# -------------------------
# Defaults mirroring train.py
# -------------------------
@dataclass
class TrainDefaults:
    seed: int = 1024
    resume: bool = False
    image_size: int = 32
    num_channels: int = 2
    centered: bool = True
    use_geometric: bool = False
    beta_min: float = 0.1
    beta_max: float = 20.0

    num_channels_dae: int = 64
    n_mlp: int = 3
    ch_mult: Any = field(default_factory=lambda: [1,1,2,2,4,4])
    num_res_blocks: int = 2
    attn_resolutions: Any = (16,)
    dropout: float = 0.0
    resamp_with_conv: bool = True
    conditional: bool = True
    fir: bool = True
    fir_kernel: list = field(default_factory=lambda: [1,3,3,1])
    skip_rescale: bool = True
    resblock_type: str = "biggan"
    progressive: str = "none"
    progressive_input: str = "residual"
    progressive_combine: str = "sum"

    embedding_type: str = "positional"
    fourier_scale: float = 16.0
    not_use_tanh: bool = False

    exp: str = "tune"
    input_path: str = INPUT_PATH
    output_path: str = OUTPUT_ROOT
    nz: int = 100
    num_timesteps: int = 4

    z_emb_dim: int = 256
    t_emb_dim: int = 256
    batch_size: int = 1
    num_epoch: int = MAX_EPOCHS
    ngf: int = 64

    lr_g: float = 1.5e-4
    lr_d: float = 0.5e-4
    beta1: float = 0.5
    beta2: float = 0.9
    no_lr_decay: bool = False

    use_ema: bool = False
    ema_decay: float = 0.9999

    r1_gamma: float = 0.05
    lazy_reg: Any = None

    save_content: bool = False
    save_content_every: int = 10
    save_ckpt_every: int = 10
    lambda_l1_loss: float = 0.5

    # DDP single GPU
    num_proc_node: int = 1
    num_process_per_node: int = 1
    node_rank: int = 0
    local_rank: int = 0
    master_address: str = "127.0.0.1"
    contrast1: str = "T1_mapping_fl2d"
    contrast2: str = "DIXON_T1_mapping_fl2d"
    port_num: str = "6021"


# -------------------------
# Utility helpers
# -------------------------

def as_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**d)


def make_trial_exp_dir(root: str, trial_number: int) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(root, f"tune_trial{trial_number:04d}_{stamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def load_train_module():
    try:
        return importlib.import_module(TRAIN_MODULE)
    except Exception as e:
        raise SystemExit(
            f"Could not import training module '{TRAIN_MODULE}'.\n"
            f"Error: {e}\n"
            f"Tip: rename TRAIN_MODULE at the top of fine_tune.py to match your file name."
        )


def summarize_metrics(exp_dir: str) -> Dict[str, float]:
    """Load numpy metrics saved by the training loop and compute summaries."""
    metrics = {}
    l1_path = os.path.join(exp_dir, "val_l1_loss.npy")
    psnr_path = os.path.join(exp_dir, "val_psnr_values.npy")

    if os.path.exists(l1_path):
        l1 = np.load(l1_path)
        # shape [2, epochs+1? or epochs, len(val_loader)] — we'll mean across class & iter and take last epoch
        # robustly handle any leading zeros from prealloc
        last_epoch = l1.shape[1] - 1
        metrics["l1_last"] = float(np.nanmean(l1[:, last_epoch, :]))
        metrics["l1_best"] = float(np.nanmin(np.nanmean(l1, axis=2)))  # best across epochs, per class mean
        metrics["l1_mean"] = float(np.nanmean(l1))

    if os.path.exists(psnr_path):
        ps = np.load(psnr_path)
        last_epoch = ps.shape[1] - 1
        metrics["psnr_last"] = float(np.nanmean(ps[:, last_epoch, :]))
        # best across epochs
        metrics["psnr_best"] = float(np.nanmax(np.nanmean(ps, axis=2)))
        metrics["psnr_mean"] = float(np.nanmean(ps))

    return metrics


def apply_overrides(defaults: TrainDefaults, fixed: Dict[str, Any]) -> Dict[str, Any]:
    base = asdict(defaults)
    for k, v in (fixed or {}).items():
        if v is not None:
            base[k] = v
    return base


def sample_from_space(trial: "optuna.trial.Trial") -> Dict[str, Any]:

    sampled: Dict[str, Any] = {}

    # 1) Sample use_geometric first (so we can branch the sensitive params)
    ug_kind, ug_choices = SEARCH_SPACE["use_geometric"]
    assert ug_kind == "categorical"
    sampled["use_geometric"] = trial.suggest_categorical("use_geometric", ug_choices)

    # 2) Sample the rest from your SEARCH_SPACE, except the conditional ones we’ll handle later
    skip = {"use_geometric", "num_timesteps", "beta_min", "beta_max"}
    for name, spec in SEARCH_SPACE.items():
        if name in skip:
            continue
        kind = spec[0]
        if kind == "log_uniform":
            low, high = spec[1], spec[2]
            sampled[name] = trial.suggest_float(name, low, high, log=True)
        elif kind == "uniform":
            low, high = spec[1], spec[2]
            sampled[name] = trial.suggest_float(name, low, high)
        elif kind == "int":
            low, high = spec[1], spec[2]
            sampled[name] = trial.suggest_int(name, low, high)
        elif kind == "categorical":
            choices = spec[1]
            sampled[name] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(f"Unknown space kind: {kind}")

    # 3) Conditionally sample the diffusion-schedule params with safer ranges
    if sampled["use_geometric"]:
        # geometric schedule is touchy → keep timesteps >=5 and narrower beta ranges
        sampled["num_timesteps"] = trial.suggest_int("num_timesteps", 5, 8)
        sampled["beta_min"] = trial.suggest_float("beta_min", 1e-3, 1e-1, log=True)
        sampled["beta_max"] = trial.suggest_float("beta_max", 5e-1, 1e1, log=True)
    else:
        # vp schedule is stabler → still avoid extremes
        sampled["num_timesteps"] = trial.suggest_int("num_timesteps", 4, 8)
        sampled["beta_min"] = trial.suggest_float("beta_min", 1e-3, 2e-1, log=True)
        sampled["beta_max"] = trial.suggest_float("beta_max", 1.0, 2.0e1, log=True)

    # 4) Sanity constraints → prune unstable configs early (prevents NaN blowups)
    if sampled["beta_max"] <= sampled["beta_min"]:
        raise optuna.TrialPruned("beta_max must be > beta_min")

    beta_ratio = sampled["beta_max"] / sampled["beta_min"]
    trial.set_user_attr("beta_ratio", beta_ratio)
    if beta_ratio > 1e3:
        raise optuna.TrialPruned("beta_max/beta_min ratio too large")

    # Keep D/G lrs in a reasonable band (helps GAN stability)
    lr_ratio = sampled["lr_d"] / sampled["lr_g"]
    trial.set_user_attr("lr_ratio", lr_ratio)
    if not (0.25 <= lr_ratio <= 4.0):
        raise optuna.TrialPruned("lr_d/lr_g out of [0.25, 4.0]")

    return sampled



def namespace_for_trial(trial_cfg: Dict[str, Any]) -> SimpleNamespace:
    # The training code expects args like argparse.Namespace
    ns = as_namespace(trial_cfg)
    # The training main sets world_size and size; we follow the flow
    ns.world_size = ns.num_proc_node * ns.num_process_per_node
    return ns


# -------------------------
# Trial runner
# -------------------------

def run_trial(trial: "optuna.trial.Trial") -> float:
    train_mod = load_train_module()

    # base defaults -> apply fixed overrides -> sample search params
    defaults = TrainDefaults(seed=SEED, input_path=INPUT_PATH, output_path=OUTPUT_ROOT, num_epoch=MAX_EPOCHS)
    cfg = apply_overrides(defaults, FIXED_OVERRIDES)

    sample = sample_from_space(trial)
    cfg.update(sample)

    # Unique experiment directory per trial
    exp_dir = make_trial_exp_dir(OUTPUT_ROOT, trial.number)
    cfg["exp"] = os.path.basename(exp_dir)

    # Ensure content saving so we can read metrics each epoch
    cfg.setdefault("save_content", False)  # not strictly required

    # Ports: avoid collisions if you parallelize trials (we run sequential by default)
    cfg["port_num"] = str(int(cfg.get("port_num", "6021")) + (trial.number % 1000))

    # W&B run
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=WANDB_GROUP,
        config={**cfg, "trial_number": trial.number},
        reinit=True,
        mode=WANDB_MODE,
        name=f"trial-{trial.number:04d}",
    )

    # Prepare namespace for training
    args = namespace_for_trial(cfg)

    # Run training via the provided DDP bootstrap
    try:
        train_mod.init_processes(0, 1, train_mod.train_syndiff, args)
    except Exception as e:
        # Mark the trial as failed in wandb for visibility
        wandb.alert(title="Trial failed", text=str(e))
        wandb.finish(exit_code=1)
        raise

    # Summarize metrics
    metrics = summarize_metrics(exp_dir)

    # Log files (optional): sample images if present
    for fname in [
        "sample1_discrete_epoch_{}".format(args.num_epoch) + ".png",
        "sample2_discrete_epoch_{}".format(args.num_epoch) + ".png",
        "sample1_translated_epoch_{}".format(args.num_epoch) + ".png",
        "sample2_translated_epoch_{}".format(args.num_epoch) + ".png",
    ]:
        fpath = os.path.join(exp_dir, fname)
        if os.path.exists(fpath):
            wandb.log({f"images/{fname}": wandb.Image(fpath)})

    # Log scalar summaries
    for k, v in metrics.items():
        wandb.summary[k] = v
    wandb.config.update({"exp_dir": exp_dir}, allow_val_change=True)

    # Choose objective value
    if OPTIMIZE_FOR.lower() == "psnr":
        objective = metrics.get("psnr_best") or metrics.get("psnr_last") or -1.0
        direction_value = float(objective)
    elif OPTIMIZE_FOR.lower() == "l1":
        # Optuna maximizes by default? We'll set study direction below; here just return
        objective = metrics.get("l1_best") or metrics.get("l1_last") or 1e9
        direction_value = float(objective)
    else:
        # fallback: psnr_mean
        direction_value = float(metrics.get("psnr_mean", -1.0))

    wandb.finish()
    return direction_value


# -------------------------
# Main
# -------------------------

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Save a copy of this script for reproducibility
    shutil.copyfile(__file__, os.path.join(OUTPUT_ROOT, os.path.basename(__file__)))

    # Configure Optuna study direction
    if OPTIMIZE_FOR.lower() == "psnr":
        direction = "maximize"
    elif OPTIMIZE_FOR.lower() == "l1":
        direction = "minimize"
    else:
        direction = "maximize"

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction=direction, sampler=sampler)

    def _objective(trial: "optuna.trial.Trial") -> float:
        return run_trial(trial)

    study.optimize(_objective, n_trials=N_TRIALS)

    # Print and persist best result
    print("\n==== BEST TRIAL ====")
    print(f"Trial #{study.best_trial.number}")
    print("Value:", study.best_value)
    print("Params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Save study for later inspection
    study_path = os.path.join(OUTPUT_ROOT, "optuna_study.json")
    with open(study_path, "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_trial": {
                "number": study.best_trial.number,
                "params": study.best_trial.params,
            },
            "direction": direction,
            "n_trials": len(study.trials),
        }, f, indent=2)
    print(f"Saved study summary to {study_path}")


if __name__ == "__main__":
    main()
