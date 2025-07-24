#!/usr/bin/env python
"""
generate_synthetic_diffusion.py
———————————————
Create a synthetic HDF5 dataset with a trained diffusion model.

Example
-------
python generate_synthetic_diffusion.py \
  --input_hdf5 data/data_T1_mapping_fl2d.h5 \
  --output_hdf5 data/synthetic_diffusion/synth_T1_fl2d.h5 \
  --ckpt_path my_results/exp_syndiff2/content.pth \
  --filters_json configs/filters_t1.json \
  --transforms_json configs/transforms.json \
  --num_timesteps 4 \
  --batch_size 4 \
  --device cuda
"""


import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)
import os
import json
import argparse
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────
# Your project‑specific imports
# ─────────────────────────────────────────────
from pytorch_dataset import HDF5ContrastDataset        # adjust if package path differs
from backbones.ncsnpp_generator_adagn import NCSNpp                  # ditto

# ═════════════════════════════════════════════
# ⬇ Diffusion‑specific helper functions
# ═════════════════════════════════════════════
def get_time_schedule(num_timesteps: int, device: torch.device):
    """Continuous‑time index ϵ ∈ (0,1] used to build β‑schedule."""
    eps_small = 1e-3
    t = torch.linspace(0, 1, num_timesteps + 1, dtype=torch.float64,
                       device=device) * (1. - eps_small) + eps_small
    return t                                               # shape: [num_timesteps+1]

def get_sigma_schedule(args, device):
    """Re‑implements the schedule used during training."""
    n_timestep = args.num_timesteps
    beta_min, beta_max = args.beta_min, args.beta_max
    t = get_time_schedule(n_timestep, device)
    var = 1. - torch.exp(-0.5 * (beta_max - beta_min) * t ** 2 - beta_min * t)
    alpha_bars = 1. - var
    betas = 1. - alpha_bars[1:] / alpha_bars[:-1]
    betas = torch.cat((torch.tensor([1e-8], device=device), betas)).float()
    sigmas = torch.sqrt(betas)
    a_s = torch.sqrt(1. - betas)
    return sigmas, a_s, betas

class PosteriorCoefficients:
    """Pre‑computes p(x_{t−1}|x_t,x₀) coefficients."""
    def __init__(self, args, device):
        _, _, betas = get_sigma_schedule(args, device)
        betas = betas[1:]                                    # drop t=0 dummy
        alphas     = 1. - betas
        cum_alphas = torch.cumprod(alphas, dim=0)
        cum_prev   = torch.cat((torch.ones(1, device=device), cum_alphas[:-1]), dim=0)
        var        = betas * (1. - cum_prev) / (1. - cum_alphas)
        self.c1    = betas * torch.sqrt(cum_prev) / (1. - cum_alphas)
        self.c2    = (1. - cum_prev) * torch.sqrt(alphas) / (1. - cum_alphas)
        self.log_var = torch.log(var.clamp(min=1e-20))

def _gather(coeff, timesteps, shape):
    out = coeff.gather(0, timesteps)
    return out.view(shape[0], *([1] * (len(shape) - 1)))

@torch.no_grad()
def sample_posterior(coeffs, x0, xt, t):
    mean = _gather(coeffs.c1, t, xt.shape) * x0 + _gather(coeffs.c2, t, xt.shape) * xt
    log_var = _gather(coeffs.log_var, t, xt.shape)
    noise = torch.randn_like(xt)
    mask = (1. - (t == 0).float()).view(-1, 1, 1, 1)
    return mean + mask * torch.exp(0.5 * log_var) * noise

@torch.no_grad()
def ddpm_sample(generator, coeffs, init_pair, args, num_steps, device):
    """Iterative DDPM‑style reverse process (batch‑wise)."""
    x = init_pair[:, [0]]          # current sample  (starts from noise)
    src = init_pair[:, [1]]        # conditioning image (fixed)

    for t_idx in reversed(range(num_steps)):
        t = torch.full((x.size(0),), t_idx, dtype=torch.int64, device=device)
        z = torch.randn(x.size(0), args.nz, device=device)
        x0_pred = generator(torch.cat((x, src), dim=1), t, z)
        x = sample_posterior(coeffs, x0_pred[:, [0]], x, t)
    return x                       # shape: [B,1,H,W] in [-1,1]

def predict_batch(generator, imgs, args, num_steps, device):
    """Batched variant of your original `predict_image`."""
    # (a) normalise each slice → [-1,1]
    imgs_flat = imgs.view(imgs.size(0), -1)
    min_v = imgs_flat.min(dim=1)[0].view(-1, 1, 1, 1)
    max_v = imgs_flat.max(dim=1)[0].view(-1, 1, 1, 1)
    imgs_norm = (imgs - min_v) / (max_v - min_v + 1e-8) * 2. - 1.

    # (b) create noise / pair
    noise = torch.randn_like(imgs_norm)
    x_init = torch.cat((noise, imgs_norm), dim=1).to(device)

    # (c) time schedule and coeffs
    T = get_time_schedule(num_steps, device)
    coeffs = PosteriorCoefficients(args, device)

    # (d) reverse diffusion
    synth = ddpm_sample(generator, coeffs, x_init, args, num_steps, device)
    return synth.cpu()             # keep on CPU for HDF5 I/O

# ═════════════════════════════════════════════
# ⬇ I/O helpers
# ═════════════════════════════════════════════
def save_to_hdf5(out_path, images, masks, meta_df):
    """Replicates the layout used by your CycleGAN dump."""
    with h5py.File(out_path, "w") as h5f:
        h5f.create_dataset("images", data=images, compression="gzip")
        if masks is not None:
            h5f.create_dataset("masks",  data=masks,  compression="gzip")
    with pd.HDFStore(out_path, mode="a") as store:
        store.put("metadata", meta_df, format="fixed", data_columns=True)

def load_diffusion_model(ckpt_path, device):
    """Restores generator + training args exactly as in your notebook."""
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    args.num_channels = 2                               # undo in‑training mutation
    # strip DistributedDataParallel prefix
    sd = {k.replace("module.", ""): v
          for k, v in ckpt["gen_diffusive_2_dict"].items()}
    net = NCSNpp(args).to(device)
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net, args

def load_net_T1_DIXON(content_path, device = "cpu"):

    ckpt        = torch.load(content_path,
                            map_location=device,
                            weights_only=False)
    train_args  = ckpt["args"]

    # 🔧 undo the in‑training mutation
    train_args.num_channels = 2          # ← IMPORTANT
    # state‑dict from DDP → strip the 'module.' prefix
    raw_sd   = ckpt["gen_diffusive_2_dict"]
    clean_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}

    # rebuild the generator exactly as at training time
    gen_T1_DIXON = NCSNpp(train_args)
    gen_T1_DIXON.load_state_dict(clean_sd, strict=True)
    gen_T1_DIXON.to(device).eval()
    return gen_T1_DIXON, train_args

def load_net_DIXON_T1(content_path, device = "cpu"):
    ckpt        = torch.load(content_path,
                            map_location=device,
                            weights_only=False)
    train_args  = ckpt["args"]

    # 🔧 undo the in‑training mutation
    train_args.num_channels = 2          # ← IMPORTANT
    # state‑dict from DDP → strip the 'module.' prefix
    raw_sd   = ckpt["gen_diffusive_1_dict"]
    clean_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}

    # rebuild the generator exactly as at training time
    gen_DIOXN_T1 = NCSNpp(train_args)
    gen_DIOXN_T1.load_state_dict(clean_sd, strict=True)
    gen_DIOXN_T1.to(device).eval()
    return gen_DIOXN_T1, train_args

# ═════════════════════════════════════════════
# ⬇ Main routine
# ═════════════════════════════════════════════
def generate_synthetic_dataset(
    input_hdf5_path: str,
    output_hdf5_path: str,
    ckpt_path: str,
    filters: dict,
    transforms: dict,
    num_timesteps: int = 4,
    batch_size: int = 1,
    device_str: str = "cpu",
):
    device = torch.device(device_str)
    print(f"[INFO] Loading diffusion generator from {ckpt_path} on {device}…")
    gen, train_args = load_net_T1_DIXON(ckpt_path, device)

    ds = HDF5ContrastDataset(
        hdf5_path=input_hdf5_path,
        filter=filters,
        transform=transforms,
        stage="eval",
    )
    print(f"[INFO] Dataset: {len(ds)} samples • batch={batch_size}")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    synth_imgs, mask_list, meta_list = [], [], []

    for batch in dl:
        imgs  = batch["image"]            # [B,1,H,W], torch.float32
        synth = predict_batch(gen, imgs, train_args,
                              num_steps=num_timesteps,
                              device=device)              # [B,1,H,W]
        # save image for testing as png
        #synth_png = synth.squeeze().numpy()
        synth = (synth+1)/2
        # save as png
        # synth_png = (synth_png - synth_png.min()) / (synth_png.max() - synth_png.min())
        # synth_png = (synth_png * 255).astype(np.uint8)
        #plt.imsave(f"synth_{batch['metadata']['index'][0]}.png", synth_png, cmap='gray')


        synth_imgs.append(synth)                          # keep on CPU
        if "mask" in batch:
            mask_list.append(batch["mask"])               # already CPU
        # drop heavy feature blobs
        meta = {k: v for k, v in batch["metadata"].items()
                if k not in ("dino_features", "fid_features")}
        meta_list.append(pd.DataFrame(meta))

    # stack & final shape conversion → (N,H,W,1)
    imgs_np  = torch.cat(synth_imgs).numpy().transpose(0, 2, 3, 1)
    masks_np = (torch.cat(mask_list).numpy().transpose(0, 2, 3, 1)
                if mask_list else None)
    meta_df  = pd.concat(meta_list, ignore_index=True)
    meta_df["index"] = np.arange(len(meta_df))

    print(f"[INFO] Writing → {output_hdf5_path}")
    os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)
    save_to_hdf5(output_hdf5_path, imgs_np, masks_np, meta_df)
    print("[DONE] Synthetic dataset written successfully.")

# ═════════════════════════════════════════════
# ⬇ CLI
# ═════════════════════════════════════════════
def _json_or_dict(s):
    if os.path.isfile(s):
        with open(s) as f:
            return json.load(f)
    # allow passing inline JSON on CLI
    return json.loads(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--input_hdf5",  required=True)
    #parser.add_argument("--output_hdf5", required=True)
    #parser.add_argument("--ckpt_path",   required=True)
    #parser.add_argument("--filters_json",   type=_json_or_dict, required=True,
    #                    help="Either path to a JSON file or an inline JSON string.")
    #parser.add_argument("--transforms_json", type=_json_or_dict, required=True)
    #parser.add_argument("--num_timesteps",   type=int, default=4)
    #parser.add_argument("--batch_size",      type=int, default=1)
    #parser.add_argument("--device",          default="cpu",
    #                    help="cuda | cpu | cuda:0 …")
    #args = parser.parse_args()

    input_hdf5 = "/home/students/studweilc1/SynthRegGAN/data/data_T1_mapping_fl2d.h5"
    output_hdf5 = "synthetic_data/synth_T1_val.h5"
    ckpt_path = "my_results/exp_syndiff2/content.pth"

    cfg_filters_t1 = {
    "contrast__in": ["T1_mapping_fl2d"],
    "non_zero": True,
    "image_dim": 0,
    "image_type": "s",
    "split":"val",
    #"patient_id": "P_01_A",  # IGNORE
    #"z_dim__in": [14,16,18],              # IGNORE
    }

    cfg_transform = {
    "fit": [
        {"PerImageMinMaxNormalize": {}}
    ],
    "eval": [
        {"PerImageMinMaxNormalize": {}}

    ]
    }
    num_timesteps = 4
    batch_size = 4
    device = "cpu"

    generate_synthetic_dataset(
        input_hdf5_path=input_hdf5,
        output_hdf5_path=output_hdf5,
        ckpt_path=ckpt_path,
        filters=cfg_filters_t1,
        transforms=cfg_transform,
        num_timesteps=num_timesteps,
        batch_size=batch_size,
        device_str=device,
    )
