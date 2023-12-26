# PatchNCE loss from https://github.com/taesungp/contrastive-unpaired-translation
# https://github.com/YSerin/ZeCon/blob/main/optimization/losses.py
from typing import Tuple, Union, Optional, List

from torch.nn import functional as F
import torch
import numpy as np
import torch.nn as nn

class CutLoss:
    def __init__(self, n_patches=256, patch_size=1):
        self.n_patches = n_patches
        self.patch_size = patch_size
    
    def get_attn_cut_loss(self, ref_noise, trg_noise):
        loss = 0

        bs, res2, c = ref_noise.shape
        res = int(np.sqrt(res2))

        ref_noise_reshape = ref_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2) 
        trg_noise_reshape = trg_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2)

        for ps in self.patch_size:
            if ps > 1:
                pooling = nn.AvgPool2d(kernel_size=(ps, ps))
                ref_noise_pooled = pooling(ref_noise_reshape)
                trg_noise_pooled = pooling(trg_noise_reshape)
            else:
                ref_noise_pooled = ref_noise_reshape
                trg_noise_pooled = trg_noise_reshape

            ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1)
            trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

            ref_noise_pooled = ref_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2)
            patch_ids = np.random.permutation(ref_noise_pooled.shape[1]) 
            patch_ids = patch_ids[:int(min(self.n_patches, ref_noise_pooled.shape[1]))]
            patch_ids = torch.tensor(patch_ids, dtype=torch.long, device=ref_noise.device)

            ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            trg_noise_pooled = trg_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2) 
            trg_sample = trg_noise_pooled[:1 , patch_ids, :].flatten(0, 1) 
            
            loss += self.PatchNCELoss(ref_sample, trg_sample).mean() 
        return loss

    def PatchNCELoss(self, ref_noise, trg_noise, batch_size=1, nce_T = 0.07):
        batch_size = batch_size
        nce_T = nce_T
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        mask_dtype = torch.bool

        num_patches = ref_noise.shape[0]
        dim = ref_noise.shape[1]
        ref_noise = ref_noise.detach()
        
        l_pos = torch.bmm(
            ref_noise.view(num_patches, 1, -1), trg_noise.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1) 

        # reshape features to batch size
        ref_noise = ref_noise.view(batch_size, -1, dim)
        trg_noise = trg_noise.view(batch_size, -1, dim) 
        npatches = ref_noise.shape[1]
        l_neg_curbatch = torch.bmm(ref_noise, trg_noise.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=ref_noise.device, dtype=mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0) 
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / nce_T

        loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=ref_noise.device))

        return loss

class DDSLoss:

    def noise_input(self, z, eps=None, timestep: Optional[int]= None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low = self.t_min,
                high = min(self.t_max, 1000) -1,
                size=(b,),
                device=z.device,
                dtype=torch.long
            )

        if eps is None:
            eps = torch.randn_like(z)

        z_t = self.scheduler.add_noise(z, eps, timestep)
        return z_t, eps, timestep
    
    def get_epsilon_prediction(self, z_t, timestep, embedd, guidance_scale=7.5, cross_attention_kwargs=None):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = embedd.permute(1, 0, 2, 3).reshape(-1, *embedd.shape[2:])

        e_t = self.unet(latent_input, timestep, embedd, cross_attention_kwargs=cross_attention_kwargs,).sample
        e_t_uncond, e_t = e_t.chunk(2)
        e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        assert torch.isfinite(e_t).all()

        return e_t

    def __init__(self, t_min, t_max, unet, scheduler, device):
        self.t_min = t_min
        self.t_max = t_max
        self.unet = unet
        self.scheduler = scheduler
        self.device = device
