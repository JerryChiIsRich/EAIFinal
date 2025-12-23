import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from dataset_edge2shoes import Edge2ShoesDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="dataset root, contains train/ and val/")
    p.add_argument("--out_dir", type=str, default="outputs/teacher_controlnet")
    p.add_argument("--concat_format", action="store_true", help="pix2pix concat images in split folder")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--sd_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--controlnet_id", type=str, default="lllyasviel/control_v11p_sd15_canny")
    p.add_argument("--prompt", type=str, default="a photo of shoes")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision)
    device = accelerator.device

    # Load SD components
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.sd_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.sd_id, subfolder="unet")

    # Load ControlNet (start from canny-pretrained)
    controlnet = ControlNetModel.from_pretrained(args.controlnet_id)

    # Noise scheduler for training
    noise_scheduler = DDIMScheduler.from_pretrained(args.sd_id, subfolder="scheduler")

    # Freeze everything except ControlNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    ds = Edge2ShoesDataset(
        root=args.data_root,
        split="train",
        size=args.image_size,
        concat_format=args.concat_format,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    controlnet, optimizer, dl = accelerator.prepare(controlnet, optimizer, dl)
    vae.to(device); text_encoder.to(device); unet.to(device)

    # Precompute constant prompt embedding
    with torch.no_grad():
        tok = tokenizer([args.prompt], padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
        prompt_emb = text_encoder(tok.input_ids.to(device))[0]  # [1,77,768]

    global_step = 0
    pbar = tqdm(total=args.max_steps, disable=not accelerator.is_local_main_process)

    while global_step < args.max_steps:
        for batch in dl:
            if global_step >= args.max_steps:
                break

            edge = batch["edge"].to(device)       # [B,3,H,W] in [0,1]
            target = batch["target"].to(device)   # [B,3,H,W] in [0,1]

            # VAE expects [-1,1]
            target_ = target * 2.0 - 1.0

            with torch.no_grad():
                latents = vae.encode(target_).latent_dist.sample()
                latents = latents * 0.18215

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = prompt_emb.repeat(bsz, 1, 1)

            with accelerator.accumulate(controlnet):
                down_res, mid_res = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=edge,
                    return_dict=False
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                ).sample

                # standard diffusion epsilon loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_local_main_process:
                pbar.set_description(f"teacher step {global_step} loss {loss.item():.4f}")
                pbar.update(1)

            global_step += 1

    pbar.close()

    if accelerator.is_local_main_process:
        controlnet_to_save = accelerator.unwrap_model(controlnet)
        controlnet_to_save.save_pretrained(args.out_dir)
        print(f"[OK] Saved teacher ControlNet to: {args.out_dir}")


if __name__ == "__main__":
    main()
