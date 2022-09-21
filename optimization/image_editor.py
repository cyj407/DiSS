import os
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from utils.metrics_accumulator import MetricsAccumulator
from utils.video import save_video

from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from optimization.losses import range_loss, d_clip_loss
import lpips
import numpy as np
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils.visualization import show_tensor_image, show_editied_masked_image


class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        os.makedirs(self.args.output_path, exist_ok=True)

        self.ranked_results_path = Path(os.path.join(self.args.output_path, RANKED_RESULTS_DIR))
        os.makedirs(self.ranked_results_path, exist_ok=True)

        if self.args.export_assets:
            self.assets_path = Path(os.path.join(self.args.output_path, ASSETS_DIR_NAME))
            os.makedirs(self.assets_path, exist_ok=True)
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)


        self.model_config = {
            'image_size': 256, 'num_channels': 128, 'num_res_blocks': 3, 
            'num_heads': 4, 'num_heads_upsample': -1, 'num_head_channels': -1, 
            'attention_resolutions': '32,16,8', 'channel_mult': '', 'dropout': 0.0,
            'class_cond': False, 'use_checkpoint': False, 
            'use_scale_shift_norm': True, 'resblock_updown': False, 
            'use_fp16': False, 'use_new_attention_order': False, 
            'learn_sigma': False, 'diffusion_steps': 4000, 
            'noise_schedule': 'linear', 'timestep_respacing': '250',
            'use_kl': False, 'predict_xstart': False, 'rescale_timesteps': False,
            'rescale_learned_sigmas': False
        }

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(
            torch.load(
                "./checkpoints/demo_cat.pt"
                if self.args.model_output_size == 256
                else "checkpoints/512x512_diffusion.pt",
                map_location="cpu",
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 4000)).long()

        return unscaled_timestep


    def edit_image_by_prompt(self, args=None):
        
        # input
        self.input_pil = Image.open(self.args.input).convert("RGB")
        self.init_input_pil = self.input_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_input = (
            TF.to_tensor(self.input_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )

        # sketch
        if self.args.init_sketch == "":
            self.init_sketch_gray = torch.full((self.args.batch_size, 1, self.model_config["image_size"], self.model_config["image_size"]), 0.0)
            self.init_sketch = self.init_sketch_gray.to(self.device)
        else:
            self.init_sketch_gray = Image.open(self.args.init_sketch).convert("L")
            self.init_sketch_gray = self.init_sketch_gray.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.init_sketch_pil = Image.open(self.args.init_sketch).convert("RGB")
            self.init_sketch_pil = self.init_sketch_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.init_sketch = (
                TF.to_tensor(self.init_sketch_gray).to(self.device).unsqueeze(0).mul(2).sub(1)
            )

        # stroke
        self.init_stroke_pil = Image.open(self.args.init_stroke).convert("RGB")
        self.init_stroke_pil = self.init_stroke_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_stroke = (
            TF.to_tensor(self.init_stroke_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )

        if self.args.export_assets:
            img_path = self.assets_path / Path(self.args.output_file)
            self.init_stroke_pil.save(img_path)

        self.mask = torch.ones_like(self.init_stroke, device=self.device)
        self.mask_pil = None
        if self.args.mask is not None:
            self.mask_pil = Image.open(self.args.mask).convert("RGB")
            if self.mask_pil.size != self.image_size:
                self.mask_pil = self.mask_pil.resize(self.image_size, Image.NEAREST)  # type: ignore
            image_mask_pil_binarized = ((np.array(self.mask_pil) > 0.5) * 255).astype(np.uint8)
            self.mask = TF.to_tensor(Image.fromarray(image_mask_pil_binarized))
            self.mask = self.mask[0, ...].unsqueeze(0).unsqueeze(0).to(self.device)

        def cond_fn(x, sketch, stroke, t, y=None):
            if self.args.prompt == "":
                return torch.zeros_like(x)

            with torch.enable_grad():
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, sketch, stroke, t, clip_denoised=False, model_kwargs={"y": y}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                # x_in = out["pred_xstart"]

                loss = torch.tensor(0)

                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    # self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                if self.args.background_preservation_loss:
                    if self.mask is not None:
                        masked_background = x_in * (1 - self.mask)
                    else:
                        masked_background = x_in

                    if self.args.lpips_sim_lambda:
                        loss = (
                            loss
                            + self.lpips_model(masked_background, self.init_stroke).sum()
                            * self.args.lpips_sim_lambda
                        )
                    if self.args.l2_sim_lambda:
                        loss = (
                            loss
                            + mse_loss(masked_background, self.init_stroke) * self.args.l2_sim_lambda
                        )

                return -torch.autograd.grad(loss, x)[0]

        @torch.no_grad()
        def postprocess_fn(out, t):
            if self.mask is not None:
                background_stage_t = self.diffusion.q_sample(self.init_stroke, t[0])
                background_stage_t = torch.tile(
                    background_stage_t, dims=(self.args.batch_size, 1, 1, 1)
                )
                out["sample"] = out["sample"] * self.mask + background_stage_t * (1 - self.mask)

            return out


        # Create a classifier-free guidance sampling function
        def model_fn(x_t, sketch, stroke, ts, \
        sketch_guidance_scale=self.args.sketch_guidance_scale, 
        stroke_guidance_scale=self.args.stroke_guidance_scale):
            unconditional_sketch = torch.full(self.init_sketch.shape, 0.0)
            unconditional_sketch = unconditional_sketch.to(self.device)
        
            unconditional_stroke = torch.full(self.init_stroke.shape, 0.0)
            unconditional_stroke = unconditional_stroke.to(self.device)

            uncond_model_out = self.model(x_t, unconditional_sketch, unconditional_stroke, ts)
            sketch_model_out = self.model(x_t, sketch, unconditional_stroke, ts)
            stroke_model_out = self.model(x_t, unconditional_sketch, stroke, ts)

            out = uncond_model_out \
                + sketch_guidance_scale * (sketch_model_out-uncond_model_out) \
                + stroke_guidance_scale * (stroke_model_out-uncond_model_out) 
            return out


        save_image_interval = self.diffusion.num_timesteps // 5
        for iteration_number in range(self.args.iterations_num):
            print(f"Start iterations {iteration_number}")
            print(self.args)

            if self.args.partial_edit:

                samples = self.diffusion.p_sample_loop_progressive(
                    model_fn,
                    ( self.args.batch_size, 3, self.model_config["image_size"], self.model_config["image_size"]),
                    sketch=self.init_sketch,
                    stroke=self.init_stroke,
                    clip_denoised=False,
                    model_kwargs={}
                    if self.args.model_output_size == 256
                    else {
                        "y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
                    },
                    cond_fn=None,
                    device=self.device, # show attribute error if comments it
                    progress=True,
                    skip_timesteps=self.args.skip_timesteps,
                    init_image=None,
                    # postprocess_fn=None if self.args.local_clip_guided_diffusion else postprocess_fn,
                    randomize_class=True,
                    realism_scale=self.args.realism_scale,
                    # resizers=self.resizers,
                    range_t=self.args.range_t,
                    partial_edit=self.args.partial_edit,
                )

            else:

                samples = self.diffusion.p_sample_loop_progressive(
                    model_fn,
                    ( self.args.batch_size, 3, self.model_config["image_size"], self.model_config["image_size"]),
                    sketch=self.init_sketch,
                    stroke=self.init_stroke, # partial editing
                    clip_denoised=False,
                    model_kwargs={}
                    if self.args.model_output_size == 256
                    else {
                        "y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
                    },
                    cond_fn=None,
                    device=self.device, # show attribute error if comments it
                    progress=True,
                    skip_timesteps=self.args.skip_timesteps,
                    init_image=self.init_input,
                    randomize_class=True,
                    realism_scale=self.args.realism_scale,
                    range_t=self.args.range_t,
                    partial_edit=self.args.partial_edit,
                )

            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
            for j, sample in enumerate(samples):
                should_save_image = j % save_image_interval == 0 or j == total_steps
                if should_save_image or self.args.save_video:

                    for b in range(self.args.batch_size):
                        pred_image = sample["pred_xstart"][b]
                        visualization_path = Path(
                            os.path.join(self.args.output_path, self.args.output_file)
                        )

                        pred_image = pred_image.add(1).div(2).clamp(0, 1)
                        pred_image_pil = TF.to_pil_image(pred_image)

                        if self.args.export_assets:
                            pred_path = self.assets_path / visualization_path.name
                            pred_image_pil.save(pred_path)

                        intermediate_samples[b].append(pred_image_pil)
                        if should_save_image:
                            show_editied_masked_image(
                                title=self.args.prompt,
                                source_image=self.init_input_pil,
                                edited_image=pred_image_pil,
                                sketch_image=self.init_sketch_pil,
                                stroke_image=self.init_stroke_pil,
                                mask=self.mask_pil, #if self.args.init_sketch != "" else None,
                                path=visualization_path,
                            )

        return pred_image_pil