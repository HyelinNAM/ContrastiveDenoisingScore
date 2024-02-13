from typing import Any, Optional, Union, Tuple, List, Callable, Dict
import os
from PIL import Image
from torchvision import transforms as tfms
import torch.optim as optim

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.utils import logging

from utils.attention import *
from utils.loss import *

logger = logging.get_logger(__name__)

class CDSPipeline(StableDiffusionPipeline):

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None, img_path=''):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        
        if latents is None:
            if not os.path.exists(img_path) :
                raise ValueError(f"No image to covert!")
            else:
                vae_magic = 0.18215
                img = Image.open(img_path).convert('RGB').resize((512, 512))
                img = tfms.ToTensor()(img).unsqueeze(0).to(device, dtype)
                with torch.no_grad():
                     latents = self.vae.encode(img.to(device=device)*2 -1)
                latents = latents['latent_dist'].mean * vae_magic
        else:
            latents = latents.to(device)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        img_path: str = '', 
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 7.5,
        # Target prompt for editing
        trg_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        trg_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,

        # Additional args for CDS
        n_patches: int = 256, 
        patch_size : [int, List[int]] = [1, 2],
        w_dds: float = 1.0,
        w_cut: float = 3.0,
        save_path: str = None,
    ):

        # Modify unet to save self-attention map
        self.unet = prep_unet(self.unet)

        sa_attn = {}

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input & target prompt
        prompt_embeds, trg_prompt_embeds = self._encode_prompt(
            prompt,
            trg_prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            trg_prompt_embeds=trg_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            img_path
        )

        # Update latents
        # timestep ~ U(0.05, 0.95) to avoid very high/low noise level
        self.num_train_timesteps = 1000
        self.min_step =   int(self.num_train_timesteps * 0.05) # 50
        self.max_step = int(self.num_train_timesteps * 0.95) # 950

        # Define loss class
        dds_loss = DDSLoss(
            t_min=self.min_step, 
            t_max =self.max_step,
            unet = self.unet,
            scheduler = self.scheduler,
            device=device, 
        )
        cut_loss = CutLoss(n_patches, patch_size)

        # Edit image!
        z_src = latents
        z_trg = latents.clone()
        z_trg.requires_grad = True

        optimizer = optim.SGD([z_trg], lr=0.1)

        num_warmup_steps = num_inference_steps - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(num_inference_steps):
                optimizer.zero_grad()

                z_t_src, eps, timestep = dds_loss.noise_input(z_src, eps=None, timestep=None)
                z_t_trg, _, _ = dds_loss.noise_input(z_trg, eps, timestep)

                # get score for dds & reference attention maps
                eps_pred = dds_loss.get_epsilon_prediction(
                    torch.cat((z_t_src, z_t_trg)),
                    torch.cat((timestep, timestep)),
                    torch.cat((prompt_embeds, trg_prompt_embeds))
                )

                eps_pred_src, eps_pred_trg = eps_pred.chunk(2)
                grad = eps_pred_trg - eps_pred_src
            
                sa_attn[timestep.item()] = {}

                for name, module in self.unet.named_modules(): 
                    module_name = type(module).__name__
                    
                    if module_name == "Attention":
                        if "attn1" in name and "up" in name:
                            hidden_state = module.hs
                            sa_attn[timestep.item()][name] = hidden_state.detach().cpu()

                with torch.enable_grad():
                    loss = z_trg * grad.clone()
                    # reduction 'mean'
                    loss = loss.sum() / (z_trg.shape[2] * z_trg.shape[3])

                    (2000 * loss * w_dds).backward()

                # calculate cut loss
                with torch.enable_grad():
                    z_t_trg, _, _ = dds_loss.noise_input(z_trg, eps, timestep)
                    eps_pred_trg = dds_loss.get_epsilon_prediction(
                        z_t_trg,
                        timestep,
                        trg_prompt_embeds,
                        )
                        
                    cutloss = 0
                    for name, module in self.unet.named_modules(): 
                        module_name = type(module).__name__
                        if module_name == "Attention":
                            # sa_cut
                            if "attn1" in name and "up" in name:
                                curr = module.hs
                                ref = sa_attn[timestep.item()][name].detach().to(device)
                                cutloss += cut_loss.get_attn_cut_loss(ref, curr)

                    (cutloss * w_cut).backward()

                optimizer.step()

                # call the callback, if provided
                if i == num_inference_steps - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                if (i+1)  % 50 == 0:
                    # src img
                    img_src = self.decode_latents(z_src).squeeze()
                    # trg img
                    img_trg = self.decode_latents(z_trg).squeeze()
                    
                    img = np.concatenate((img_src, img_trg), axis=1)
                    img = Image.fromarray((img * 255).astype(np.uint8))

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    img.save(os.path.join(save_path, f'{str(i).zfill(3)}.png'))

        result = self.decode_latents(z_trg).squeeze()
        result = Image.fromarray((result * 255).astype(np.uint8))

        return result
                
    def _encode_prompt(
        self,
        prompt,
        trg_prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        trg_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

        if trg_prompt_embeds is None:
            trg_text_inputs = self.tokenizer(
                trg_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            trg_text_input_ids = trg_text_inputs.input_ids
            trg_untruncated_ids = self.tokenizer(trg_prompt, padding="longest", return_tensors="pt").input_ids

            if trg_untruncated_ids.shape[-1] >= trg_text_input_ids.shape[-1] and not torch.equal(
                trg_text_input_ids, trg_untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    trg_untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
            trg_attention_mask = trg_text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
            trg_attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        trg_prompt_embeds = self.text_encoder(
            trg_text_input_ids.to(device),
            attention_mask=trg_attention_mask,
        )
        trg_prompt_embeds = trg_prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        trg_prompt_embeds = trg_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.stack([negative_prompt_embeds, prompt_embeds], axis=1)
            trg_prompt_embeds =  torch.stack([negative_prompt_embeds, trg_prompt_embeds], axis=1)

        return prompt_embeds, trg_prompt_embeds