import math
import torch
import numpy as np
from PIL import Image
from comfy.ldm.modules.attention import BasicTransformerBlock


# Color transfer (Monge-Kantorovitch linear)

def color_transfer_stats(src_frames, ref_image):
    r = src_frames.reshape(-1, src_frames.shape[-1]).T.astype(np.float64)
    z = ref_image.reshape(-1, ref_image.shape[-1]).T.astype(np.float64)
    mu_r = r.mean(axis=1)[..., np.newaxis]
    mu_z = z.mean(axis=1)[..., np.newaxis]
    eig_val_r, eig_vec_r = np.linalg.eigh(np.cov(r))
    eig_val_r = np.maximum(eig_val_r, 0)
    val_r = np.diag(np.sqrt(eig_val_r[::-1]))
    vec_r = eig_vec_r[:, ::-1]
    inv_r = np.diag(1.0 / (np.diag(val_r + np.spacing(1))))
    mat_c = val_r @ vec_r.T @ np.cov(z) @ vec_r @ val_r
    eig_val_c, eig_vec_c = np.linalg.eigh(mat_c)
    eig_val_c = np.maximum(eig_val_c, 0)
    val_c = np.diag(np.sqrt(eig_val_c))
    transfer_mat = vec_r @ inv_r @ eig_vec_c @ val_c @ eig_vec_c.T @ inv_r @ vec_r.T
    return mu_r, mu_z, transfer_mat


def color_transfer_apply(frame, variables):
    mu_r, mu_z, transfer_mat = variables
    r = frame.reshape(-1, frame.shape[-1]).T.astype(np.float64)
    return (np.dot(transfer_mat, r - mu_r) + mu_z).T.reshape(frame.shape)


def preprocess_color_transfer(frames_np, ref_np):
    """MKL color transfer from source frames toward reference. Matches original preprocess() with ncc=False."""
    frames_small = np.array([np.array(Image.fromarray(f).resize((256, 256))) for f in frames_np])
    variables = color_transfer_stats(frames_small, ref_np)
    output = np.array([color_transfer_apply(frame, variables) for frame in frames_np])
    output = (output - output.min()) / (output.max() - output.min())
    return (output * 255.0).astype(np.uint8)


# Diffusers to LDM key conversion

def convert_diffusers_unet_to_ldm(diffusers_sd, unet_config):
    import comfy.utils
    key_map = comfy.utils.unet_to_diffusers(unet_config)
    return {ldm_key: diffusers_sd[diff_key] for diff_key, ldm_key in key_map.items() if diff_key in diffusers_sd}


# SD1.5 UNet configs

SD15_UNET_CONFIG = {
    'image_size': 32, 'in_channels': 4, 'out_channels': 4, 'model_channels': 320,
    'num_res_blocks': [2, 2, 2, 2], 'channel_mult': [1, 2, 4, 4],
    'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0], 'transformer_depth_middle': 1,
    'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    'use_spatial_transformer': True, 'use_linear_in_transformer': False,
    'context_dim': 768, 'num_heads': 8, 'num_head_channels': -1,
    'use_temporal_attention': False, 'use_temporal_resblock': False,
    'legacy': False, 'use_checkpoint': False, 'adm_in_channels': None,
}

def get_referencenet_config():
    return SD15_UNET_CONFIG.copy()

def get_ldiffuser_config():
    config = SD15_UNET_CONFIG.copy()
    config['in_channels'] = 6
    config['out_channels'] = 3
    return config


# Identity LUT

def make_identity_lut(size=16):
    """Create a size^3 identity LUT. Returns (sqrt(size^3), sqrt(size^3), 3) array."""
    coords = np.linspace(0, 1, size, dtype=np.float64)
    # PIL Color3DLUT order: r varies fastest, then g, then b
    r, g, b = [a.ravel() for a in np.meshgrid(coords, coords, coords, indexing='ij')]
    lut = np.stack([b, g, r], axis=-1)  # b slowest → r fastest when meshgrid uses ij
    side = round(size ** 1.5)
    assert side * side == size ** 3, f"size^3={size**3} is not a perfect square"
    return lut.reshape(side, side, 3)


# DDIM Scheduler

class DDIMScheduler:
    """Minimal DDIM matching diffusers DDIMScheduler(beta_schedule='linear', clip_sample=True, steps_offset=1)."""

    def __init__(self, beta_start=0.00085, beta_end=0.012, num_train_timesteps=1000, clip_sample=True, steps_offset=1):
        betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        self.alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
        self.final_alpha_cumprod = 1.0
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.steps_offset = steps_offset
        self.timesteps = None

    def set_timesteps(self, num_inference_steps, device='cpu'):
        self._step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * self._step_ratio).round()[::-1].copy() + self.steps_offset
        self.timesteps = torch.from_numpy(timesteps).long().to(device)

    def scale_model_input(self, sample, timestep):
        return sample

    def step(self, model_output, timestep, sample):
        t = int(timestep)
        prev_t = t - self._step_ratio
        alpha_t = float(self.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod)
        alpha_prev = float(self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod)

        pred_x0 = (sample - math.sqrt(1 - alpha_t) * model_output) / math.sqrt(alpha_t)
        if self.clip_sample:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
        return math.sqrt(alpha_prev) * pred_x0 + math.sqrt(1 - alpha_prev) * model_output


# ReferenceNet attention bank

def torch_dfs(model):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class ReferenceNetAttention:
    """Monkey-patches BasicTransformerBlock for the attention bank mechanism.
    Writer (ReferenceNet): stores ref-src diff features. Reader (L-Diffuser): injects bank into self-attention."""

    def __init__(self, unet_model, mode="write", fusion_blocks="full"):
        self.unet = unet_model
        self.mode = mode
        self.fusion_blocks = fusion_blocks
        self._setup_hooks()

    def _get_attn_modules(self, model):
        if self.fusion_blocks == "midup":
            modules = [m for m in (torch_dfs(model.middle_block) + torch_dfs(model.output_blocks)) if isinstance(m, BasicTransformerBlock)]
        else:
            modules = [m for m in torch_dfs(model) if isinstance(m, BasicTransformerBlock)]
        return sorted(modules, key=lambda x: -x.norm1.normalized_shape[0])

    def _setup_hooks(self):
        MODE = self.mode

        def hacked_forward(self_block, x, context=None, transformer_options={}):
            if context is not None and isinstance(context, torch.Tensor) and context.ndim == 2:
                context = context.unsqueeze(1)

            if MODE == "write":
                n = self_block.norm1(x)
                ref_hidden, src_hidden = n.clone().chunk(2)
                self_block.bank.append(ref_hidden - src_hidden)
                try:
                    n = self_block.attn1(n, transformer_options=transformer_options)
                    x = n + x
                    if self_block.attn2 is not None:
                        n2 = self_block.norm2(x)
                        n2 = self_block.attn2(n2, context=context, transformer_options=transformer_options)
                        x = n2 + x
                    x = self_block.ff(self_block.norm3(x)) + x
                except RuntimeError:
                    pass  # expected for blocks with incomplete checkpoint weights (output_blocks.11)
                return x

            elif MODE == "read":
                n = self_block.norm1(x)
                modify_n = torch.cat([n] + self_block.bank, dim=1)
                hidden_states = self_block.attn1(modify_n, context=modify_n, transformer_options=transformer_options)[:, :x.shape[1], :] + x
                if self_block.attn2 is not None:
                    n2 = self_block.norm2(hidden_states)
                    hidden_states = self_block.attn2(n2, context=context, transformer_options=transformer_options) + hidden_states
                hidden_states = self_block.ff(self_block.norm3(hidden_states)) + hidden_states
                return hidden_states

            return self_block._original_forward(x, context=context, transformer_options=transformer_options)

        attn_modules = self._get_attn_modules(self.unet)
        for module in attn_modules:
            module._original_forward = module.forward
            module.forward = hacked_forward.__get__(module, BasicTransformerBlock)
            module.bank = []

    def update(self, writer):
        reader_dtype = self.unet.dtype
        for r, w in zip(self._get_attn_modules(self.unet), self._get_attn_modules(writer.unet)):
            r.bank = [v.clone().to(reader_dtype) for v in w.bank]

    def clear(self):
        for m in self._get_attn_modules(self.unet):
            m.bank.clear()

    def restore(self):
        for m in self._get_attn_modules(self.unet):
            if hasattr(m, '_original_forward'):
                m.forward = m._original_forward
                del m._original_forward
            if hasattr(m, 'bank'):
                m.bank.clear()
                del m.bank
