import argparse
import os
import lpips
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from peft import LoraConfig, get_peft_model

from dataset_edge2shoes import Edge2ShoesDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--concat_format", action="store_true")
    p.add_argument("--teacher_controlnet_dir", type=str, required=True, help="path saved from teacher training")
    p.add_argument("--out_dir", type=str, default="outputs/student_lora_lcm")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--sd_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--prompt", type=str, default="a photo of shoes")

    # LoRA knobs
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    return p.parse_args()


def predict_x0_from_eps(xt, eps, scheduler, t):
    """
    scheduler = DDPMScheduler; xt = noisy latent
    For epsilon prediction:
      x0 = (xt - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)
    """
    # alpha_bar_t shape [B]
    alpha_bar = scheduler.alphas_cumprod.to(xt.device)[t].view(-1, 1, 1, 1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)
    x0 = (xt - sqrt_one_minus * eps) / (sqrt_alpha_bar + 1e-8)
    return x0


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision)
    device = accelerator.device

    # Load SD components
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.sd_id, subfolder="vae")

    # Teacher UNet (frozen)
    teacher_unet = UNet2DConditionModel.from_pretrained(args.sd_id, subfolder="unet")
    teacher_unet.requires_grad_(False)

    # Teacher ControlNet (your finetuned)
    teacher_controlnet = ControlNetModel.from_pretrained(args.teacher_controlnet_dir)
    teacher_controlnet.requires_grad_(False)

    # Student UNet + LoRA
    base_student_unet = UNet2DConditionModel.from_pretrained(args.sd_id, subfolder="unet")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # attention projections
    )
    student_unet = get_peft_model(base_student_unet, lora_cfg)
    student_unet.train()
    # 允許 out conv 微調，避免 collapse
    for name, param in student_unet.named_parameters():
        if any(k in name for k in ["conv_out", "mid_block", "up_blocks.3"]):
            param.requires_grad = True

    # Freeze VAE/TextEncoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    optimizer = torch.optim.AdamW(student_unet.parameters(), lr=args.lr)

    # Noise scheduler for training-time noising
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_id, subfolder="scheduler")

    ds = Edge2ShoesDataset(
        root=args.data_root, split="train", size=args.image_size, concat_format=args.concat_format
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    student_unet, optimizer, dl = accelerator.prepare(student_unet, optimizer, dl)
    vae.to(device); text_encoder.to(device)
    teacher_unet.to(device); teacher_controlnet.to(device)
    lpips_loss = lpips.LPIPS(net="vgg").to(device)
    lpips_loss.eval()
    for p in lpips_loss.parameters():
        p.requires_grad = False

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

            edge = batch["edge"].to(device)       # [B,3,H,W] [0,1]
            target = batch["target"].to(device)   # [B,3,H,W] [0,1]
            bsz = target.size(0)

            # Encode target to latent x0
            target_ = target * 2.0 - 1.0
            with torch.no_grad():
                x0 = vae.encode(target_).latent_dist.sample() * 0.18215

            # Sample timestep and noise, create xt
            noise = torch.randn_like(x0)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            xt = noise_scheduler.add_noise(x0, noise, t)

            encoder_hidden_states = prompt_emb.repeat(bsz, 1, 1)

            # Teacher predicts eps -> x0_teacher
            with torch.no_grad():
                down_res_t, mid_res_t = teacher_controlnet(
                    xt, t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=edge,
                    return_dict=False
                )
                eps_teacher = teacher_unet(
                    xt, t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_res_t,
                    mid_block_additional_residual=mid_res_t,
                ).sample
                x0_teacher = predict_x0_from_eps(xt, eps_teacher, noise_scheduler, t)
                
            with accelerator.accumulate(student_unet):
                # Student predicts eps -> x0_student
                # (ControlNet stays teacher/frozen; you can later also distill a ControlNet LoRA if needed)
                with torch.no_grad():
                    down_res_s, mid_res_s = teacher_controlnet(
                        xt, t,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=edge,
                        return_dict=False
                    )

                eps_student = student_unet(
                    xt, t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_res_s,
                    mid_block_additional_residual=mid_res_s,
                ).sample
                x0_student = predict_x0_from_eps(xt, eps_student, noise_scheduler, t)

                # LCM-style distillation loss (teacher-student consistency on x0)
                lambda_teacher = 1.0
                lambda_gt = 0.5

                loss_teacher = F.mse_loss(x0_student.float(), x0_teacher.float())
                loss_gt = F.mse_loss(x0_student.float(), x0.float())

                loss = loss_teacher + 0.5 * loss_gt

                # 2. perceptual loss（補紋理）— 不用每一步算
                if global_step % 4 == 0:
                    # decode student（保留 gradient）
                    x0_student_img = vae.decode(x0_student / 0.18215).sample

                    # decode teacher（完全不需要 gradient）
                    with torch.no_grad():
                        x0_teacher_img = vae.decode(x0_teacher / 0.18215).sample

                    # LPIPS 用較低解析度，省顯存
                    x0_student_img = F.interpolate(
                        x0_student_img, size=(128, 128),
                        mode="bilinear", align_corners=False
                    )
                    x0_teacher_img = F.interpolate(
                        x0_teacher_img, size=(128, 128),
                        mode="bilinear", align_corners=False
                    )

                    loss_lpips = lpips_loss(
                        x0_student_img,
                        x0_teacher_img.detach()
                    ).mean()

                    loss = loss + 0.1 * loss_lpips

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_local_main_process:
                pbar.set_description(f"distill step {global_step} loss {loss.item():.4f}")
                pbar.update(1)

            global_step += 1

    pbar.close()

    if accelerator.is_local_main_process:
        # Save LoRA adapter
        student_to_save = accelerator.unwrap_model(student_unet)
        student_to_save.save_pretrained(args.out_dir)
        print(f"[OK] Saved student LoRA to: {args.out_dir}")


if __name__ == "__main__":
    main()
