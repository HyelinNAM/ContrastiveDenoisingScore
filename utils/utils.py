import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler,  UNet2DConditionModel

from pipeline_cds import CDSPipeline

def load_model(args):
    if not args.v5:
        sd_version = "CompVis/stable-diffusion-v1-4"
    else:
        sd_version = "runwayml/stable-diffusion-v1-5"

    weight_dtype = torch.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    stable = CDSPipeline.from_pretrained(sd_version, torch_dtype=weight_dtype)
    return stable
