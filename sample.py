from functools import partial
import os
import argparse
import yaml
from glob import glob

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
import numpy as np
import base64

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def list_numbered_pngs(root_dir: str, start: int, end: int):
    """
    Return list of (idx, filepath) for files like 69000.png ... 70000.png that exist.
    """
    pairs = []
    for k in range(start, end + 1):
        fp = os.path.join(root_dir, f"{k}.png")
        if os.path.isfile(fp):
            pairs.append((k, fp))
    return pairs


def pil_load_rgb(path: str):
    img = Image.open(path).convert("RGB")
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')

    # dataset 범위/경로
    parser.add_argument('--data_dir', type=str, required=True, help="Folder containing 69000.png ...")
    parser.add_argument('--start', type=int, default=69000)
    parser.add_argument('--end', type=int, default=70000)

    args = parser.parse_args()

    logger = get_logger()

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml("configs/model_config.yaml")
    diffusion_config = load_yaml("configs/diffusion_config.yaml")
    task_config = load_yaml("configs/super_resolution_config.yaml")

    # Load model
    model = create_model(**model_config).to(device)
    model.eval()

    # Prepare Operator and noise (Super-Resolution이어야 함)
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    base_sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    # Output dirs
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'], f"{args.start}_{args.end}")
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Transform (원 코드 동일: [0,1] -> [-1,1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Inpainting mask (필요시)
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(**measure_config['mask_opt'])

    # File list (69000~70000)
    pairs = list_numbered_pngs(args.data_dir, args.start, args.end)
    if len(pairs) == 0:
        raise FileNotFoundError(f"No png files found in range {args.start}..{args.end} under {args.data_dir}")

    logger.info(f"Found {len(pairs)} images to process.")

    for j, (idx, path) in enumerate(pairs):
        logger.info(f"[{j+1}/{len(pairs)}] Inference for image {idx}: {path}")
        fname = f"{idx}.png"

        # load + preprocess
        ref_pil = pil_load_rgb(path)
        ref_img = transform(ref_pil).unsqueeze(0).to(device)  # (1,3,H,W)

        # Inpainting special case (아마 SR이면 안 들어감)
        if measure_config['operator']['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            local_measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(base_sample_fn, measurement_cond_fn=local_measurement_cond_fn)

            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        else:
            y = ref_img
            # y = operator.forward(ref_img)   # SR이면 low-res measurement 생성

            y_n = noiser(y)
            sample_fn = base_sample_fn

        # Sampling (원 코드처럼 record=True/ save_root 사용)
        x_start = torch.randn((1, 3, 256, 256), device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

        # Save images
        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

    logger.info(f"Done. Results saved to: {out_path}")


if __name__ == '__main__':
    main()
