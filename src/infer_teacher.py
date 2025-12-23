import argparse
import torch
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--edge", type=str, required=True)
    p.add_argument("--teacher_controlnet_dir", type=str, required=True)
    p.add_argument("--sd_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--out", type=str, default="teacher.png")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--prompt", type=str, default="a photo of shoes")
    p.add_argument("--size", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    controlnet = ControlNetModel.from_pretrained(args.teacher_controlnet_dir, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.sd_id, controlnet=controlnet, torch_dtype=torch.float16
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    edge = Image.open(args.edge).convert("RGB").resize((args.size, args.size))
    img = pipe(
        args.prompt,
        image=edge,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
    ).images[0]
    img.save(args.out)
    print("[OK] saved:", args.out)


if __name__ == "__main__":
    main()
