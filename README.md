# [CVPR 2024] Contrastive Denoising Score (CDS)

Official pytorch implementation of CDS.<br>

**[Contrastive Denoising Score for Text-guided Latent Diffusion Image Editing](https://arxiv.org/abs/2311.18608)**
<br>
[Hyelin Nam](https://scholar.google.com/citations?user=k6nzq08AAAAJ&hl=en),
[Gihyun Kwon](https://scholar.google.co.kr/citations?user=yexbg8gAAAAJ&hl=en),
[Geon Yeong Park](https://scholar.google.co.kr/citations?user=HGF4a14AAAAJ&hl=en), 
[Jong Chul Ye](https://scholar.google.co.kr/citations?user=HNMjoNEAAAAJ&hl=en)

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://hyelinnam.github.io/CDS/)
[![arXiv](https://img.shields.io/badge/arXiv-2311.18608-b31b1b.svg)](https://arxiv.org/abs/2311.18608)


## Abstract
With the remarkable advent of text-to-image diffusion models, image editing methods have become more diverse and continue to evolve. A promising recent approach in this realm is Delta Denoising Score (DDS) - an image editing technique based on Score Distillation Sampling (SDS) framework that leverages the rich generative prior of text-to-image diffusion models. However, relying solely on the difference between scoring functions is insufficient for preserving specific structural elements from the original image, a crucial aspect of image editing. To address this, here we present an embarrassingly simple yet very powerful modification of DDS, called Contrastive Denoising Score (CDS), for latent diffusion models (LDM). Inspired by the similarities and differences between DDS and the contrastive learning for unpaired image-to-image translation(CUT), we introduce a straightforward approach using CUT loss within the DDS framework. Rather than employing auxiliary networks as in the original CUT approach, we leverage the intermediate features of LDM, specifically those from the self-attention layers, which possesses rich spatial information. Our approach enables zero-shot image-to-image translation and neural radiance field (NeRF) editing, achieving structural correspondence between the input and output while maintaining content controllability. Qualitative results and comparisons demonstrates the effectiveness of our proposed method.

<p align="center">
<img src="assets/cover.png" width="100%"/>  
</p>

## Overview
<p>
<img src="assets/key_observation.png" width="100%"/>  
<br>
Aside from the actual image generation processes, both DDS and CUT approaches attempt to align the features from the reference and reconstruction branches. With the aim of regulating the excessive structural changes, we are interested in the key idea from CUT, which is shown effective in maintaining input structure.

<img src="assets/method_overview.png" width="100%"/>  
<br>
The original CUT algorithm requires training an encoder to extract spatial information from the input image, which is inefficient. 
On the other hand, we calculate CUT loss utilizing <b> the latent representation of self-attention layers</b>.
Those features have been shown to contain rich spatial information, disentangled from semantic information, which is exactly CUT loss requires.
</p>


## Setup
### Requirements
```shell
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install diffusers==0.16.1 
pip install transformers==4.32.1 
```

### Usage
```bash
python run.py --img_path sample/cat1.png --prompt "a cat" --trg_prompt "a pig" --w_cut 3.0 --patch_size 1 2 --n_patches 256
```
