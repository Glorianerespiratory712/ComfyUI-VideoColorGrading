import logging
import numpy as np
import torch
from PIL import Image

import comfy.utils
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.sd
import comfy.clip_model
import comfy.clip_vision
import comfy.latent_formats
import folder_paths
from comfy_api.latest import IO
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

from .vcg_utils import (
    convert_diffusers_unet_to_ldm, get_referencenet_config, get_ldiffuser_config,
    make_identity_lut, preprocess_color_transfer, DDIMScheduler, ReferenceNetAttention,
)

logger = logging.getLogger(__name__)

VCG_PIPELINE = IO.Custom("VCG_PIPELINE")
VCG_LUT = IO.Custom("VCG_LUT")


CLIP_VIT_B32_CONFIG = {
    "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
    "intermediate_size": 3072, "patch_size": 32, "image_size": 224,
    "num_channels": 3, "hidden_act": "quick_gelu", "model_type": "clip_vision_model",
}

def load_clip_vision_vitb32(clip_sd):
    """Load CLIP ViT-B/32 using inline config. Bypasses __init__ because ClipVisionModel
    only accepts a json file path, and we want to avoid writing a temp file."""
    config = CLIP_VIT_B32_CONFIG
    clip = comfy.clip_vision.ClipVisionModel.__new__(comfy.clip_vision.ClipVisionModel)
    clip.image_size = config["image_size"]
    clip.image_mean = [0.48145466, 0.4578275, 0.40821073]
    clip.image_std = [0.26862954, 0.26130258, 0.27577711]
    clip.model_type = config["model_type"]
    clip.config = config.copy()
    clip.return_all_hidden_states = False
    clip.load_device = comfy.model_management.text_encoder_device()
    offload_device = comfy.model_management.text_encoder_offload_device()
    clip.dtype = comfy.model_management.text_encoder_dtype(clip.load_device)
    model_class = comfy.clip_vision.IMAGE_ENCODERS[config["model_type"]]
    clip.model = model_class(config, clip.dtype, offload_device, comfy.ops.manual_cast)
    clip.model.eval()
    clip.patcher = comfy.model_patcher.CoreModelPatcher(clip.model, load_device=clip.load_device, offload_device=offload_device)
    clip.load_sd(clip_sd)
    return clip


def pil_resize_batch(frames_tensor, target_size):
    """Resize (B,H,W,3) float [0,1] tensor batch using PIL to match original pipeline."""
    out = []
    for i in range(frames_tensor.shape[0]):
        frame_np = (frames_tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        resized = np.array(Image.fromarray(frame_np).resize((target_size, target_size)))
        out.append(torch.from_numpy(resized).float() / 255.0)
    return torch.stack(out)


# Load VCG Model
class VCGLoadModel(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VCGLoadModel", display_name="Load VCG Model", category="loaders/vcg",
            description="Load VideoColorGrading model (CLIP, VAE, ReferenceNet, L-Diffuser)",
            inputs=[IO.Combo.Input("model_name", options=folder_paths.get_filename_list("checkpoints"), tooltip="VCG model file")],
            outputs=[VCG_PIPELINE.Output(display_name="vcg_pipeline")],
        )

    @classmethod
    def execute(cls, model_name):
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("checkpoints", model_name))

        def extract(prefix):
            p = prefix + "."
            return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}

        clip = load_clip_vision_vitb32(extract("clip_vision"))
        vae = comfy.sd.VAE(sd=extract("vae"))

        unet_dtype = comfy.model_management.unet_dtype()
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        ref_config = get_referencenet_config()
        ref_model = UNetModel(**ref_config, device="cpu", dtype=unet_dtype, operations=comfy.ops.disable_weight_init)
        ref_model.load_state_dict(convert_diffusers_unet_to_ldm(extract("referencenet"), ref_config), strict=False)
        ref_patcher = comfy.model_patcher.ModelPatcher(ref_model, load_device=load_device, offload_device=offload_device)

        diff_config = get_ldiffuser_config()
        diff_model = UNetModel(**diff_config, device="cpu", dtype=unet_dtype, operations=comfy.ops.disable_weight_init)
        diff_model.load_state_dict(convert_diffusers_unet_to_ldm(extract("l_diffuser"), diff_config), strict=False)
        diff_patcher = comfy.model_patcher.ModelPatcher(diff_model, load_device=load_device, offload_device=offload_device)

        return IO.NodeOutput({"clip_vision": clip, "vae": vae, "referencenet": ref_patcher, "l_diffuser": diff_patcher})


class VCGGenerateLUT(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VCGGenerateLUT", display_name="Generate Color LUT (VCG)", category="video/color-grading",
            description="Generate a 3D color LUT from a reference image and source video frames",
            inputs=[
                VCG_PIPELINE.Input("vcg_pipeline"),
                IO.Image.Input("reference_image", tooltip="Reference image with target color style"),
                IO.Image.Input("source_frames", tooltip="Source video frames"),
                IO.Int.Input("steps", default=25, min=1, max=100, tooltip="DDIM denoising steps"),
                IO.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff),
            ],
            outputs=[
                IO.Image.Output("preprocessed_frames", display_name="preprocessed_frames", tooltip="Color-transferred frames (apply LUT to these)"),
                VCG_LUT.Output(display_name="lut"),
            ],
        )

    @classmethod
    def execute(cls, vcg_pipeline, reference_image, source_frames, steps, seed):
        size = 512
        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        clip_vision, vae = vcg_pipeline["clip_vision"], vcg_pipeline["vae"]
        referencenet, l_diffuser = vcg_pipeline["referencenet"], vcg_pipeline["l_diffuser"]

        id_lut_hwc = make_identity_lut(16)
        id_lut_chw = torch.from_numpy(id_lut_hwc.transpose(2, 0, 1)).unsqueeze(0).float()

        # Color transfer preprocessing
        ref_np = (reference_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        source_np = (source_frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        preprocessed_frames = torch.from_numpy(preprocess_color_transfer(source_np, ref_np)).float() / 255.0

        ref_img_resized = pil_resize_batch(reference_image[0:1], size)
        source_resized = pil_resize_batch(preprocessed_frames, size)

        # CLIP frame matching - load model once, encode frames without repeated load_model_gpu
        comfy.model_management.load_model_gpu(clip_vision.patcher)
        cos_sim = torch.nn.CosineSimilarity()

        def _clip_embed(img):
            px = comfy.clip_model.clip_preprocess(img.to(clip_vision.load_device), size=clip_vision.image_size, mean=clip_vision.image_mean, std=clip_vision.image_std).float()
            return clip_vision.model(pixel_values=px, intermediate_output=-2)[2].to(comfy.model_management.intermediate_device())

        clip_ref = _clip_embed(ref_img_resized).unsqueeze(1).to(dtype=dtype)

        max_sim, best_idx = -1.0, 0
        clip_src = None
        for idx in range(0, source_resized.shape[0], 24):
            src_embeds = _clip_embed(source_resized[idx:idx+1])
            sim_val = cos_sim(clip_ref.squeeze(1), src_embeds).item()
            if sim_val > max_sim:
                max_sim, best_idx = sim_val, idx
                clip_src = src_embeds.unsqueeze(1).to(dtype=dtype)

        if clip_src is None:
            clip_src = _clip_embed(source_resized[0:1]).unsqueeze(1).to(dtype=dtype)

        clip_diff = clip_ref - clip_src
        encoder_hidden_states = torch.cat([clip_ref, clip_src], dim=0)
        # VAE encode
        latent_format = comfy.latent_formats.SD15()
        ref_latents = latent_format.process_in(vae.encode(ref_img_resized))
        src_latents = latent_format.process_in(vae.encode(source_resized[best_idx:best_idx+1]))

        # Attention banks
        ref_unet, diff_unet = referencenet.model, l_diffuser.model
        writer = ReferenceNetAttention(ref_unet, mode='write', fusion_blocks="full")
        reader = ReferenceNetAttention(diff_unet, mode='read', fusion_blocks="full")

        try:
            # ReferenceNet forward
            comfy.model_management.load_model_gpu(referencenet)
            ref_input = torch.cat([ref_latents, src_latents], dim=0).to(device=device, dtype=dtype)
            context = encoder_hidden_states.to(device=device, dtype=dtype)
            ref_unet(ref_input, timesteps=torch.zeros(2, device=device, dtype=torch.long), context=context)
            reader.update(writer)

            # DDIM denoising loop
            comfy.model_management.load_model_gpu(l_diffuser)
            scheduler = DDIMScheduler()
            scheduler.set_timesteps(steps, device=device)

            generator = torch.Generator(device='cpu').manual_seed(seed)
            latents = torch.randn((1, 3, size // 8, size // 8), generator=generator, device='cpu', dtype=dtype).to(device)
            id_lut_device = id_lut_chw.to(device=device, dtype=dtype)
            clip_diff_device = clip_diff.to(device=device, dtype=dtype)
            pbar = comfy.utils.ProgressBar(len(scheduler.timesteps))

            for t in scheduler.timesteps:
                latent_input = scheduler.scale_model_input(latents, t)
                if id_lut_device.shape[2:] != latent_input.shape[2:]:
                    id_lut_resized = torch.nn.functional.interpolate(id_lut_device, size=latent_input.shape[2:], mode="bilinear", align_corners=False)
                else:
                    id_lut_resized = id_lut_device
                noisy_latents = torch.cat([latent_input, id_lut_resized], dim=1)
                pred = diff_unet(noisy_latents, timesteps=t.unsqueeze(0), context=clip_diff_device)
                latents = scheduler.step(pred, t, latents)
                pbar.update(1)
        finally:
            writer.restore()
            reader.restore()

        # Post-process LUT
        lut_residual = latents[0].detach().cpu().float().numpy()
        if lut_residual.shape[1] != 64 or lut_residual.shape[2] != 64:
            lut_residual = torch.nn.functional.interpolate(
                torch.from_numpy(lut_residual).unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
            )[0].numpy()

        lut_values = np.clip(lut_residual.transpose(1, 2, 0) + id_lut_hwc, 0.0, 1.0)
        return IO.NodeOutput(preprocessed_frames, {"values": lut_values})


class VCGApplyLUT(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VCGApplyLUT", display_name="Apply 3D LUT (VCG)", category="video/color-grading",
            description="Apply a generated 3D color LUT to images",
            inputs=[
                IO.Image.Input("images"),
                VCG_LUT.Input("lut"),
                IO.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.01, optional=True, tooltip="LUT strength: 0=no effect, 1=full effect, >1=exaggerated"),
            ],
            outputs=[IO.Image.Output()],
        )


    @classmethod
    def execute(cls, images, lut, strength=1.0):
        device = comfy.model_management.intermediate_device()
        dtype = images.dtype
        lut_3d = torch.from_numpy(lut["values"].reshape(16, 16, 16, 3).astype(np.float32))
        if strength != 1.0:
            identity = torch.from_numpy(make_identity_lut(16).reshape(16, 16, 16, 3).astype(np.float32))
            lut_3d = identity + strength * (lut_3d - identity)
        lut_vol = lut_3d.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)
        B = images.shape[0]
        output = torch.empty_like(images)
        pbar = comfy.utils.ProgressBar(B)
        for i in range(B):
            grid = (images[i:i+1].to(device).clamp(0, 1) * 2.0 - 1.0).unsqueeze(3)
            result = torch.nn.functional.grid_sample(lut_vol, grid, mode='bilinear', padding_mode='border', align_corners=True)
            output[i] = result[0, :, :, :, 0].permute(1, 2, 0).cpu()
            pbar.update(1)
        return IO.NodeOutput(output)
