import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import torch.utils.checkpoint
from abc import abstractmethod
from typing import Dict, Optional, Sequence, Tuple
from .common import NoScaleDropout, MPFourier

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class UpBlock(TimestepBlock):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            embed_dim, 
            mlp_drop,
            num_res_blocks,  
            use_checkpoint,
            up = True
        ):

        super().__init__()
        self.blocks = nn.ModuleList()
        self.up = up
        for _ in range(num_res_blocks):
            self.blocks.append(
                ResBlock(
                    in_channels,
                    embed_dim,
                    mlp_drop,
                    out_channels = out_channels,
                    use_checkpoint = use_checkpoint,
                    use_scale_shift_norm = True
                )
            )
            in_channels = out_channels

        if self.up:
            self.upsample = ResBlock(
                in_channels,
                embed_dim,
                mlp_drop,
                out_channels = in_channels,
                use_checkpoint = use_checkpoint,
                use_scale_shift_norm = True,
                up = True
            )

    def forward(self, x, emb):
        for block in self.blocks:
            x = block(x, emb)
        if self.up:
            x = self.upsample(x,emb)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """
    def __init__(
            self, 
            channels, 
            use_conv = True,  
            out_channels = None
        ):

        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(
                self.channels, 
                self.out_channels, 
                3, 
                padding = 1
            )

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(
            x, 
            scale_factor = 2, 
            mode = "nearest"
        )
        
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """
    def __init__(
            self, 
            channels, 
            use_conv = True,  
            out_channels = None
        ):

        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2 
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, 
                self.out_channels, 
                3, 
                stride = stride, 
                padding = 1 
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(
                kernel_size = stride, 
                stride = stride
            )

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class DownBlock(TimestepBlock):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            embed_dim, 
            mlp_drop, 
            num_res_blocks,  
            use_checkpoint,
            down = True
        ):

        super().__init__()
        self.blocks = nn.ModuleList()
        self.down = down
        for _ in range(num_res_blocks):
            self.blocks.append(
                ResBlock(
                    in_channels,
                    embed_dim,
                    mlp_drop,
                    out_channels = out_channels,
                    use_checkpoint = use_checkpoint,
                    use_scale_shift_norm = True
                )
            )
            in_channels = out_channels
        if self.down:
            self.downsample = ResBlock(
                in_channels,
                embed_dim,
                mlp_drop,
                out_channels = in_channels,
                use_checkpoint = use_checkpoint,
                use_scale_shift_norm = True,
                down = True,
            )

    def forward(self, x, emb):
        for block in self.blocks:
            x = block(x, emb)
        if self.down:
            x = self.downsample(x, emb)
        return x
    
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=True,
            use_scale_shift_norm=True,
            use_checkpoint=False,
            up=False,
            down=False,
        ):

        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(
                channels, 
                self.out_channels, 
                3, 
                padding = 1
            ),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                (
                    2 * self.out_channels 
                    if use_scale_shift_norm 
                    else self.out_channels
                ),
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p = dropout),
            zero_module(
                nn.Conv2d(
                    self.out_channels, 
                    self.out_channels, 
                    3, 
                    padding = 1
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, 
                self.out_channels, 
                3, 
                padding = 1
            )
        else:
            self.skip_connection = nn.Conv2d(
                channels, 
                self.out_channels, 
                1
            )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, 
                x, 
                emb
            )
        else:
            return self._forward(x, emb)
        
    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

def unpatchify(x, channels):
    """
    Reconstruct images from patches.

    :param x: Input tensor of shape [B, N_patches, patch_dim]
    :param channels: Number of channels in the output image
    :return: Reconstructed images of shape [B, C, H, W]
    """
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1], f"Invalid number of patches: expected {h * w}, got {x.shape[1]}"
    assert patch_size ** 2 * channels == x.shape[2], "Invalid dimensions for unpatchify"
    x = einops.rearrange(
        x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)',
        h = h, 
        p1 = patch_size, 
        p2 = patch_size
    )
    return x

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class Attention(nn.Module):
    def __init__(
            self, 
            dim, 
            num_heads=8, 
            qkv_bias=False, 
            qk_scale=None, 
            attn_drop=0., 
            proj_drop=0.
        ):
        
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x)
        qkv = einops.rearrange(
            qkv, 
            'B L (K H D) -> K B H L D', 
            K = 3,
            H = self.num_heads
        ).float()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, 'B H L D -> B L (H D)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(
            self, 
            in_features, 
            hidden_features = None, 
            out_features = None, 
            act_layer = nn.GELU, 
            drop = 0.0, 
            norm_layer = nn.LayerNorm
        ):

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Define layers
        self.fc1 = nn.Linear(in_features, hidden_features)
        #self.norm1 = norm_layer(hidden_features)  # LayerNorm after the first linear layer
        self.act = act_layer(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x):
        # Apply the first linear layer, activation, dropout, and norm
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        #x = self.norm1(x)

        # Apply the second linear layer, norm, and dropout
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(
            self, 
            dim, 
            num_heads, 
            mlp_ratio = 4., 
            attn_drop = 0.0, 
            mlp_drop = 0.0, 
            act_layer = nn.GELU, 
            norm_layer = nn.LayerNorm, 
            skip = False,
            use_checkpoint = False
        ):

        super().__init__()
        self.norm1 = norm_layer(dim, eps = 1e-6)
        self.attn = Attention(
            dim, 
            num_heads = num_heads, 
            attn_drop = attn_drop
        )
        self.norm2 = norm_layer(dim, eps = 1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features = dim, 
            hidden_features = mlp_hidden_dim, 
            act_layer = act_layer, 
            drop = mlp_drop, 
            norm_layer = norm_layer
        )
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip = None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip = None):
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    This module splits the image into patches and projects them to a vector space.
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        """
        :param x: Input images of shape [B, C, H, W]
        :return: Patch embeddings of shape [B, N_patches, C]
        """
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, N_patches, embed_dim]
        return x

class FourierEmbed(nn.Module):
    def __init__(
            self,
            in_dim,
            embed_dim, 
            use_mp_fourier = True
        ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.use_mp_fourier = use_mp_fourier
        self.embed_layer = nn.Sequential(
            nn.Linear(self.in_dim, 2 * self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        )
        self.MPFourier_cond = MPFourier(self.embed_dim)

    def forward(self, x):
        if self.use_mp_fourier:
            return self.embed_layer(self.MPFourier_cond(x))
        else: 
            return self.embed_layer(x)

class Encoder(nn.Module):
    """
    Transformer-based U-Net model for diffusion denoising.
    """
    def __init__(
            self,
            img_size = 256, 
            in_chans = 3,
            in_conds = 2,
            model_channels = [128, 256, 768],
            num_res_blocks = [2, 2, 2, 2],
            depth = 12,
            num_heads = 12, 
            mlp_ratio = 4., 
            attn_drop = 0.0, 
            mlp_drop = 0.0, 
            norm_layer = nn.LayerNorm,
            use_checkpoint = False,         
            use_time = True,
            use_transf = True
        ):

        super().__init__()
        self.in_chans = in_chans
        self.in_conds = in_conds # number of conditions
        self.use_time = use_time
        self.use_transf = use_transf
        self.embed_dim = model_channels[-1]
        self.extras = 1
        
        if self.use_time:
            # Time embedding module for diffusion time-steps
            self.embed_diff_time = FourierEmbed(
                self.embed_dim, 
                self.embed_dim, 
                use_mp_fourier = True
            )
        # self.embed_re = FourierEmbed(
        #     1, 
        #     self.embed_dim, 
        #     use_mp_fourier = False
        # )
        self.embed_re = FourierEmbed(
            self.embed_dim, 
            self.embed_dim, 
            use_mp_fourier = True
        )
        self.embed_target_step = FourierEmbed(
            self.embed_dim, 
            self.embed_dim,
            use_mp_fourier = True
        )
        self.embed_total_step = FourierEmbed(
            self.embed_dim, 
            self.embed_dim, 
            use_mp_fourier = True
        )

        # one-hot encoding for fluid conditions
        # self.label_emb = nn.Embedding(time_interp_index, self.embed_dim)

        in_ch = int(model_channels[0])
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(
                        self.in_chans, 
                        in_ch, 
                        3, 
                        padding = 1
                    )
                )
            ]
        )
        
        for level, ch in enumerate(model_channels):
            for _ in range(num_res_blocks[level]):
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            in_ch,
                            self.embed_dim,
                            mlp_drop,
                            out_channels = ch,
                            use_checkpoint = use_checkpoint,                
                        )
                    )
                )
                in_ch = ch
            
            self.input_blocks.append(
                TimestepEmbedSequential(
                    ResBlock(
                        ch,
                        self.embed_dim,
                        mlp_drop,
                        out_channels = ch,
                        use_checkpoint = use_checkpoint,
                        down = True,
                    )
                )
            )

        # Transformer
        if self.use_transf:
            # Patch embedding module
            self.patch_embed = PatchEmbed()        
            self.num_patches = (img_size // 2 ** len(model_channels)) ** 2

            # Positional embeddings for patches and extra tokens
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.extras + self.num_patches, self.embed_dim)
            )

            # Encoder blocks (first half of the U-Net)
            self.tr_blocks = nn.ModuleList(
                [
                    Block(
                        dim = self.embed_dim, 
                        num_heads = num_heads, 
                        mlp_ratio = mlp_ratio, 
                        attn_drop = attn_drop,
                        mlp_drop = mlp_drop, 
                        norm_layer = norm_layer, 
                        use_checkpoint = use_checkpoint
                    )
                    for _ in range(depth // 2)
                ]
            )

            # Middle block
            self.mid_block = Block(
                dim = self.embed_dim, 
                num_heads = num_heads, 
                mlp_ratio = mlp_ratio, 
                attn_drop = attn_drop,
                mlp_drop = mlp_drop, 
                norm_layer = norm_layer, 
                use_checkpoint = use_checkpoint
            )

        self.drop = NoScaleDropout(0.1)
        self.initialize_weights()
        
    def initialize_weights(self):        
        def _init_weights(m):
            # Initialize weights
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        if self.use_transf:
            # Initialize parameters
            nn.init.trunc_normal_(
                self.pos_embed, 
                mean = 0.0, 
                std = 0.02, 
                a = -2.0, 
                b = 2.0
            ) 
            
    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {'pos_embed'}

    # MODFIED: Added total interp steps and the target interp step
    def forward(
            self, 
            x, 
            timesteps = None,
            fluid_condition = None, 
            cond_skips = None,
            target_interp_step = None,
            total_interp_steps = None
        ):
        """
        Forward pass.

        :param x: Input images of shape [B, C_in, H, W]
        :param timesteps: Timesteps tensor of shape [B]
        :param y: Optional class labels of shape [B]
        :return: Output images of shape [B, C_out, H, W]
        """

        # conditioning reynolds number
        if fluid_condition is not None:
            cond = self.embed_re(fluid_condition)

        # conditioning diffusion time-steps
        if self.use_time:
            cond += self.embed_diff_time(timesteps)
        
        # conditioning target interpolation step
        if target_interp_step is not None:
            cond += self.embed_target_step(target_interp_step)

        # conditioning total interpolation steps
        if total_interp_steps is not None:
            cond += self.embed_total_step(total_interp_steps)
            
        skips = []
        for layer, module in enumerate(self.input_blocks):
            x = module(x, cond)
            if cond_skips is not None:
                x = x + cond_skips[layer] * 0.1 # TODO: * 0.1 on skips
            skips.append(x)

        if self.use_transf:            
            x = self.patch_embed(x)  # Shape: [B, N_patches, embed_dim]
            cond = cond.unsqueeze(dim=1)
            x = torch.cat((cond, x), dim=1)

            B, L, D = x.shape
            
            # Add positional embeddings
            x = x + self.pos_embed
            
            # Transformers
            for blk in self.tr_blocks:
                x = blk(x)
                skips.append(x)  # Store for skip connections

            # Middle block
            x = self.mid_block(x)

        return x, skips

class Decoder(nn.Module):
    """
    Transformer-based U-Net model for diffusion denoising.
    """
    def __init__(
            self,
            img_size = 256, 
            out_chans = 3,
            in_conds = 1,
            model_channels = [128, 256, 768],
            num_res_blocks = [2, 2, 2, 2],
            depth = 12,
            num_heads = 12, 
            mlp_ratio = 4., 
            attn_drop = 0.0, 
            mlp_drop = 0.0, 
            norm_layer = nn.LayerNorm,
            use_checkpoint = False,
            use_transf = False,
            skip = True
        ):

        super().__init__()
        self.out_chans = out_chans  # Number of output channels
        self.in_conds = in_conds
        self.extras = 1
        self.use_transf = use_transf
        self.skip = skip
        self.embed_dim = model_channels[-1]
        self.patch_embed = PatchEmbed()

        # embedding module
        self.embed_diff_time = FourierEmbed(
            self.embed_dim, 
            self.embed_dim, 
            use_mp_fourier = True
        )
        # TODO: change back 
        # self.embed_re = FourierEmbed(
        #     1,
        #     self.embed_dim, 
        #     use_mp_fourier = False
        # )
        self.embed_re = FourierEmbed(
            self.embed_dim, 
            self.embed_dim, 
            use_mp_fourier = True
        )
        self.embed_target_step = FourierEmbed(
            self.embed_dim, 
            self.embed_dim, 
            use_mp_fourier = True
        )
        self.embed_total_step = FourierEmbed(
            self.embed_dim, 
            self.embed_dim, 
            use_mp_fourier = True
        )

        # Decoder blocks (second half of the U-Net)
        self.tr_blocks = nn.ModuleList(
            [
                Block(
                    dim = self.embed_dim, 
                    num_heads = num_heads, 
                    mlp_ratio = mlp_ratio, 
                    attn_drop = attn_drop,
                    mlp_drop = mlp_drop, 
                    norm_layer = norm_layer, 
                    skip = self.skip, 
                    use_checkpoint = use_checkpoint
                )
                for _ in range(depth // 2)
            ]
        )
        self.norm = norm_layer(self.embed_dim)  # Final normalization layer

        ch = int(model_channels[0])
        input_block_chans = [ch]
        for level, ch in enumerate(model_channels):
            for _ in range(num_res_blocks[level]):
                input_block_chans.append(ch)
            if level != len(model_channels) - 1:
                input_block_chans.append(ch)

        #self.output_blocks = nn.ModuleList([])
        
        self.output_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    ResBlock(
                        2 * ch,
                        self.embed_dim,
                        mlp_drop,
                        out_channels = ch,
                        use_checkpoint = use_checkpoint,
                        up = True,
                    )
                )
            ]
        )
        
        chans = input_block_chans.copy()
        for level, out_ch in list(enumerate(model_channels))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.embed_dim,
                        mlp_drop,
                        out_channels = out_ch,
                        use_checkpoint = use_checkpoint,
                    )
                ]
                ch = out_ch
                if level and i == num_res_blocks[level]:
                    layers.append(
                        ResBlock(
                            ch,
                            self.embed_dim,
                            mlp_drop,
                            out_channels = out_ch,
                            use_checkpoint = use_checkpoint,
                            up = True,
                        )
                    )
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.final_layer = nn.Sequential(
            nn.GroupNorm(32,ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_chans, 3, padding = 1)),
        )

        self.drop = NoScaleDropout(0.1)
        # Initialize parameters
        self.initialize_weights()

    def initialize_weights(self):        
        def _init_weights( m):
            # Initialize weights
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {'pos_embed'}

    def forward(
            self, 
            x, 
            skips,
            cond, 
            cond_skips,
            timesteps,
            fluid_condition = None,
            target_interp_step = None,
            total_interp_steps = None
        ):
        """
        Forward pass of the UViT model.

        :param x: Input images of shape [B, C_in, H, W]
        :param timesteps: Timesteps tensor of shape [B]
        :param y: Optional class labels of shape [B]
        :return: Output images of shape [B, C_out, H, W]
        """
        # conditioning diffusion timesteps
        time_token = self.embed_diff_time(timesteps)

        # conditioning reynolds number
        if fluid_condition is not None:
            # Add embedding if conditions are provided
            time_token += self.embed_re(fluid_condition)

        # conditioning target interpolation step
        if target_interp_step is not None:
            time_token += self.embed_target_step(target_interp_step)

        # conditioning total interpolation steps
        if total_interp_steps is not None:
            time_token += self.embed_total_step(total_interp_steps)

        x[:,0, :] = x[:, 0, :] + time_token        
        if self.use_transf:
            x = x + cond
            
        # Transformer
        for blk in self.tr_blocks:
            skip = skips.pop() 
            if self.use_transf:
                if self.skip:
                    skip = skip + 0.1 * cond_skips.pop() # TODO: * 0.1 on skips
                else:
                    cond_skips.pop()
                    
            x = blk(x,skip)  # Apply skip connection
            
        x = self.norm(x)
        #remove conditioning info
        x = x[:, self.extras:, :]
        
        # Decoder
        x = unpatchify(x, self.embed_dim)  # Shape: [B, C_out, H, W]
        if not self.use_transf:
            #Combine middle channel
            x = x + cond
                
        for module in self.output_blocks: 
            skip = skips.pop() + cond_skips.pop() * 0.1 # TODO: * 0.1 on skips
            x = module(torch.cat([x, skip], dim = 1), time_token)

        # Final convolutional layer
        x = self.final_layer(x)  # Shape: [B, C_out, H, W]
        return x

def FLEX(
        image_size = 256,
        in_channels = 1,
        out_channels = 1,
        model_size = 'small',
        mlp_ratio = 4, # originally 2
        attn_drop = 0.0, # originally 0.1
        mlp_drop = 0.2, # originally 0.1
        norm_layer = nn.LayerNorm,
        use_checkpoint = False,
        skip = True,
        use_transf = False
    ):
    '''
    INSTRUCTIONS:
    - mlp_ratio = 4 gives better performance.
    - use medium size may be better.
    '''

    # if model_size == 'small':
    #     model_channels = [64, 128, 128, 256]
    #     decoder_res_blocks = [2, 3, 3, 3]
    #     encoder_res_blocks = [2, 3, 3, 3]
    #     depth          = 13
    #     num_heads      = 4

    if model_size == 'small':
        model_channels = [64, 128, 256, 512]
        decoder_res_blocks = [2, 2, 2, 2]
        encoder_res_blocks = [2, 2, 2, 2]
        depth          = 13
        num_heads      = 8

    elif model_size == 'medium':
        model_channels = [64, 128, 256, 512]
        decoder_res_blocks = [2, 3, 3, 4]
        encoder_res_blocks = [2, 3, 3, 4]
        depth          = 13
        num_heads      = 8
        
    elif model_size == 'big':
        model_channels = [128, 256, 512, 1152]
        decoder_res_blocks = [2, 3, 3, 3]
        encoder_res_blocks = [2, 3, 3, 3]
        depth          = 21
        num_heads      = 16

    else:
        raise ValueError("--- Size not found! ---")
    
    base_encoder = Encoder(
        img_size = image_size,
        in_chans = in_channels,
        in_conds = 1, # noise
        model_channels = model_channels,
        num_res_blocks = encoder_res_blocks,
        depth = depth,       
        num_heads = num_heads,    
        mlp_ratio = mlp_ratio,
        attn_drop = attn_drop,
        mlp_drop = mlp_drop,
        norm_layer = norm_layer,
        use_checkpoint = use_checkpoint,
    )

    task_encoder = Encoder(
        img_size = image_size,
        in_chans = in_channels, # in_channels + 1
        in_conds = 1,
        use_time = False,
        model_channels = model_channels,
        num_res_blocks = encoder_res_blocks,
        depth = depth,       
        num_heads = num_heads,    
        mlp_ratio = mlp_ratio,
        attn_drop = attn_drop,
        mlp_drop = mlp_drop,
        norm_layer = norm_layer,
        use_checkpoint = use_checkpoint,
        use_transf = use_transf,
    )

    task_encoder_end = Encoder(
        img_size = image_size,
        in_chans = in_channels,
        in_conds = 1,
        use_time = False,
        model_channels = model_channels,
        num_res_blocks = encoder_res_blocks,
        depth = depth,       
        num_heads = num_heads,    
        mlp_ratio = mlp_ratio,
        attn_drop = attn_drop,
        mlp_drop = mlp_drop,
        norm_layer = norm_layer,
        use_checkpoint = use_checkpoint,
        use_transf = use_transf,
    )
    
    base_decoder = Decoder(
        img_size = image_size,
        out_chans = out_channels,
        model_channels = model_channels,
        num_res_blocks = decoder_res_blocks,
        depth = depth,       
        num_heads = num_heads,    
        mlp_ratio = mlp_ratio,
        attn_drop = attn_drop,
        mlp_drop = mlp_drop,
        norm_layer = norm_layer,
        use_checkpoint = use_checkpoint,
        use_transf = use_transf,
        skip = skip,
    )

    return base_encoder, task_encoder, task_encoder_end, base_decoder


def _group_norm_channels(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class FourierScalarEmbedding(nn.Module):
    """
    Embed scalar inputs with fixed sinusoidal/Fourier features.

    Input:
      x: [B, 3] containing [tau, dt, log(Re)]
    Output:
      emb: [B, out_dim]
    """

    def __init__(
            self,
            in_dim: int = 3,
            num_bands: int = 16,
            out_dim: int = 256,
            max_frequency: float = 16.0,
        ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_bands = num_bands
        self.out_dim = out_dim

        freqs = torch.logspace(
            0.0,
            math.log2(max_frequency),
            num_bands,
            base=2.0,
        )
        self.register_buffer("freqs", freqs, persistent=False)
        feature_dim = in_dim * (2 * num_bands + 1)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        angles = x.unsqueeze(-1) * self.freqs.view(1, 1, -1) * (2.0 * math.pi)
        features = torch.cat(
            [x, torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)],
            dim=1,
        )
        return self.proj(features)


class ScalarConditioner(nn.Module):
    """
    Preprocess scalar conditioning and return FiLM parameters plus a latent embedding.

    Input scalars:
      tau: [B] or [B, 1]
      dt:  [B] or [B, 1]
      Re:  [B] or [B, 1]

    Output:
      scale: [B, target_dim]
      shift: [B, target_dim]
      emb:   [B, latent_dim]
    """

    def __init__(
            self,
            target_dim: int,
            latent_dim: Optional[int] = None,
            hidden_dim: Optional[int] = None,
            num_bands: int = 16,
        ) -> None:
        super().__init__()
        latent_dim = latent_dim or target_dim
        hidden_dim = hidden_dim or max(latent_dim, target_dim)
        self.scalar_embed = FourierScalarEmbedding(
            in_dim=3,
            num_bands=num_bands,
            out_dim=hidden_dim,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.to_film = nn.Linear(latent_dim, 2 * target_dim)

    @staticmethod
    def _flatten_scalar(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            return x[None]
        if x.ndim == 1:
            return x
        return x.reshape(x.shape[0], -1)[:, 0]

    def forward(
            self,
            tau: torch.Tensor,
            dt: torch.Tensor,
            reynolds: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau = self._flatten_scalar(tau).float()
        dt = self._flatten_scalar(dt).float()
        reynolds = self._flatten_scalar(reynolds).float().clamp_min(1e-8).log()

        scalars = torch.stack([tau, dt, reynolds], dim=1)
        emb = self.mlp(self.scalar_embed(scalars))
        if self.to_film is None:
            return None, None, emb
        scale, shift = torch.chunk(self.to_film(emb), 2, dim=1)
        return scale, shift, emb


class SpatialConditionFusion(nn.Module):
    """
    Fuse two shared-encoder endpoint feature maps at the same spatial resolution.

    Inputs:
      f0: [B, C, H, W]
      f1: [B, C, H, W]
    Output:
      cond_map: [B, out_channels, H, W]
    """

    def __init__(self, channels: int, out_channels: Optional[int] = None) -> None:
        super().__init__()
        out_channels = out_channels or channels
        hidden_channels = max(out_channels, channels)
        self.net = nn.Sequential(
            nn.Conv2d(4 * channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        f_avg = 0.5 * (f0 + f1)
        f_diff = f1 - f0
        return self.net(torch.cat([f0, f1, f_avg, f_diff], dim=1))


class _ConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        groups = _group_norm_channels(out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedSpatialConditionEncoder(nn.Module):
    """
    Shared-weight endpoint encoder used for both x0 and x1.

    Returned maps are keyed by spatial resolution:
      maps[H] -> [B, C, H, W]
    """

    expects_separate_endpoints = True

    def __init__(
            self,
            in_channels: int,
            model_channels: Sequence[int],
        ) -> None:
        super().__init__()
        first_channels = int(model_channels[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, first_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_norm_channels(first_channels), first_channels),
            nn.SiLU(),
        )

        stages = []
        stage_channels = [first_channels]
        in_ch = first_channels
        for stage_idx, out_ch in enumerate(model_channels):
            downsample = stage_idx > 0
            stages.append(_ConvStage(in_ch, int(out_ch), downsample=downsample))
            in_ch = int(out_ch)
            stage_channels.append(in_ch)
        stages.append(_ConvStage(in_ch, in_ch, downsample=True))
        stage_channels.append(in_ch)
        self.stages = nn.ModuleList(stages)
        self.stage_channels = stage_channels
        self.fusions = nn.ModuleDict(
            {
                str(channels): SpatialConditionFusion(channels, channels)
                for channels in sorted(set(stage_channels))
            }
        )

    def _encode_single(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        features: Dict[int, torch.Tensor] = {}
        x = self.stem(x)
        features[x.shape[-1]] = x
        for stage in self.stages:
            x = stage(x)
            features[x.shape[-1]] = x
        return features

    def forward(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            **_: torch.Tensor,
        ) -> Dict[int, torch.Tensor]:
        f0 = self._encode_single(x0)
        f1 = self._encode_single(x1)
        fused = {}
        for resolution, feat0 in f0.items():
            feat1 = f1[resolution]
            fused[resolution] = self.fusions[str(feat0.shape[1])](feat0, feat1)
        return fused


class ConditionedFeatureModulation(nn.Module):
    """
    Apply scalar FiLM and spatial additive/gated injection to a hidden map.

    Inputs:
      h:        [B, C, H, W]
      cond_map: [B, Cc, H, W] or None
    """

    def __init__(
            self,
            hidden_channels: int,
            cond_channels: int,
            mode: str = "gated",
            use_scalar_film: bool = True,
            scalar_latent_dim: Optional[int] = None,
        ) -> None:
        super().__init__()
        if mode not in {"additive", "gated"}:
            raise ValueError("mode must be 'additive' or 'gated'")

        self.mode = mode
        self.use_scalar_film = use_scalar_film
        self.scalar_conditioner = (
            ScalarConditioner(
                target_dim=hidden_channels,
                latent_dim=scalar_latent_dim or hidden_channels,
            )
            if use_scalar_film
            else None
        )
        self.cond_proj = nn.Conv2d(cond_channels, hidden_channels, kernel_size=1)
        self.gate = (
            nn.Conv2d(cond_channels, hidden_channels, kernel_size=1)
            if mode == "gated"
            else None
        )

    def forward(
            self,
            h: torch.Tensor,
            cond_map: Optional[torch.Tensor],
            tau: Optional[torch.Tensor] = None,
            dt: Optional[torch.Tensor] = None,
            reynolds: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        if self.use_scalar_film:
            if tau is None or dt is None or reynolds is None:
                raise ValueError("Scalar FiLM requires tau, dt, and reynolds")
            scale, shift, _ = self.scalar_conditioner(tau, dt, reynolds)
            h = h * (1.0 + scale[..., None, None].to(h.dtype))
            h = h + shift[..., None, None].to(h.dtype)

        if cond_map is None:
            return h

        cond_delta = self.cond_proj(cond_map)
        if self.gate is not None:
            cond_delta = torch.sigmoid(self.gate(cond_map)) * cond_delta
        return h + cond_delta


class GatedConditionInjection(ConditionedFeatureModulation):
    def __init__(
            self,
            hidden_channels: int,
            cond_channels: int,
            mode: str = "gated",
            use_scalar_film: bool = True,
            scalar_latent_dim: Optional[int] = None,
        ) -> None:
        super().__init__(
            hidden_channels=hidden_channels,
            cond_channels=cond_channels,
            mode=mode,
            use_scalar_film=use_scalar_film,
            scalar_latent_dim=scalar_latent_dim,
        )


FlexEncoder = Encoder
FlexDecoder = Decoder


class ConditionAwareEncoder(FlexEncoder):
    def __init__(
            self,
            *args,
            model_channels: Sequence[int],
            use_scalar_film: bool = True,
            use_spatial_cond: bool = True,
            spatial_cond_scales: Optional[Sequence[int]] = None,
            spatial_cond_mode: str = "gated",
            **kwargs,
        ) -> None:
        super().__init__(*args, model_channels=model_channels, **kwargs)
        self.use_scalar_film = use_scalar_film
        self.use_spatial_cond = use_spatial_cond
        self.spatial_cond_scales = {
            int(scale) for scale in (spatial_cond_scales or (32, 16))
        }
        self.scalar_conditioner = ScalarConditioner(
            target_dim=self.embed_dim,
            latent_dim=self.embed_dim,
        )
        # This backbone branch only uses the latent scalar embedding, not the
        # per-channel FiLM head, so prune the unused projection parameters.
        self.scalar_conditioner.to_film = None
        cond_channels_by_scale = {}
        if len(model_channels) >= 3:
            cond_channels_by_scale[32] = int(model_channels[2])
        if len(model_channels) >= 4:
            cond_channels_by_scale[16] = int(model_channels[3])
        self.cond_channels_by_scale = cond_channels_by_scale
        self.spatial_injections = nn.ModuleDict(
            {
                f"{int(hidden_channels)}_{int(cond_channels)}": GatedConditionInjection(
                    hidden_channels=int(hidden_channels),
                    cond_channels=int(cond_channels),
                    mode=spatial_cond_mode,
                    # The FLEX timestep embedding already modulates every ResBlock.
                    use_scalar_film=False,
                )
                for hidden_channels in sorted(set(model_channels))
                for cond_channels in sorted(set(cond_channels_by_scale.values()))
            }
        )
        self._prune_unused_modules()

    def _prune_unused_modules(self) -> None:
        # FLEX uses the scalar conditioner instead of the inherited scalar/time-step
        # side embeddings for tau/dt/Re when scalar FiLM is enabled.
        if self.use_scalar_film:
            self.embed_re = nn.Identity()
            self.embed_target_step = nn.Identity()
            self.embed_total_step = nn.Identity()

    def _build_condition_embedding(
            self,
            x: torch.Tensor,
            timesteps: Optional[torch.Tensor],
            fluid_condition: Optional[torch.Tensor],
            target_interp_step: Optional[torch.Tensor],
            total_interp_steps: Optional[torch.Tensor],
            scalar_cond: Optional[Dict[str, torch.Tensor]],
        ) -> torch.Tensor:
        cond = torch.zeros(
            x.shape[0],
            self.embed_dim,
            device=x.device,
            dtype=x.dtype,
        )
        if self.use_time and timesteps is not None:
            cond = cond + self.embed_diff_time(timesteps).to(x.dtype)

        if self.use_scalar_film and scalar_cond is not None:
            _, _, scalar_emb = self.scalar_conditioner(
                scalar_cond["tau"],
                scalar_cond["dt"],
                scalar_cond["reynolds"],
            )
            cond = cond + scalar_emb.to(x.dtype)
        else:
            if fluid_condition is not None:
                cond = cond + self.embed_re(fluid_condition).to(x.dtype)
            if target_interp_step is not None:
                cond = cond + self.embed_target_step(target_interp_step).to(x.dtype)
            if total_interp_steps is not None:
                cond = cond + self.embed_total_step(total_interp_steps).to(x.dtype)
        return cond

    def _maybe_inject_spatial(
            self,
            x: torch.Tensor,
            spatial_cond_maps: Optional[Dict[int, torch.Tensor]],
        ) -> torch.Tensor:
        if (not self.use_spatial_cond) or (spatial_cond_maps is None) or (x.ndim != 4):
            return x
        resolution = int(x.shape[-1])
        if resolution not in self.spatial_cond_scales:
            return x
        cond_map = spatial_cond_maps.get(resolution)
        if cond_map is None:
            return x
        key = f"{int(x.shape[1])}_{int(cond_map.shape[1])}"
        if key not in self.spatial_injections:
            return x
        return self.spatial_injections[key](x, cond_map)

    def forward(
            self,
            x: torch.Tensor,
            timesteps: Optional[torch.Tensor] = None,
            fluid_condition: Optional[torch.Tensor] = None,
            cond_skips=None,
            target_interp_step: Optional[torch.Tensor] = None,
            total_interp_steps: Optional[torch.Tensor] = None,
            scalar_cond: Optional[Dict[str, torch.Tensor]] = None,
            spatial_cond_maps: Optional[Dict[int, torch.Tensor]] = None,
        ):
        cond = self._build_condition_embedding(
            x=x,
            timesteps=timesteps,
            fluid_condition=fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=total_interp_steps,
            scalar_cond=scalar_cond,
        )

        skips = []
        for layer, module in enumerate(self.input_blocks):
            x = module(x, cond)
            if cond_skips is not None:
                x = x + cond_skips[layer] * 0.1
            x = self._maybe_inject_spatial(x, spatial_cond_maps)
            skips.append(x)

        if self.use_transf:
            x = self.patch_embed(x)
            cond_token = cond.unsqueeze(dim=1)
            x = torch.cat((cond_token, x), dim=1)
            x = x + self.pos_embed
            for blk in self.tr_blocks:
                x = blk(x)
                skips.append(x)
            x = self.mid_block(x)

        return x, skips


class ConditionAwareDecoder(FlexDecoder):
    def __init__(
            self,
            *args,
            model_channels: Sequence[int],
            use_scalar_film: bool = True,
            use_spatial_cond: bool = True,
            spatial_cond_scales: Optional[Sequence[int]] = None,
            spatial_cond_mode: str = "gated",
            **kwargs,
        ) -> None:
        super().__init__(*args, model_channels=model_channels, **kwargs)
        self.use_scalar_film = use_scalar_film
        self.use_spatial_cond = use_spatial_cond
        self.spatial_cond_scales = {
            int(scale) for scale in (spatial_cond_scales or (32, 16))
        }
        self.scalar_conditioner = ScalarConditioner(
            target_dim=self.embed_dim,
            latent_dim=self.embed_dim,
        )
        self.scalar_conditioner.to_film = None
        cond_channels_by_scale = {}
        if len(model_channels) >= 3:
            cond_channels_by_scale[32] = int(model_channels[2])
        if len(model_channels) >= 4:
            cond_channels_by_scale[16] = int(model_channels[3])
        self.cond_channels_by_scale = cond_channels_by_scale
        self.spatial_injections = nn.ModuleDict(
            {
                f"{int(hidden_channels)}_{int(cond_channels)}": GatedConditionInjection(
                    hidden_channels=int(hidden_channels),
                    cond_channels=int(cond_channels),
                    mode=spatial_cond_mode,
                    use_scalar_film=False,
                )
                for hidden_channels in sorted(set(model_channels))
                for cond_channels in sorted(set(cond_channels_by_scale.values()))
            }
        )
        self._prune_unused_modules()

    def _prune_unused_modules(self) -> None:
        if self.use_scalar_film:
            self.embed_re = nn.Identity()
            self.embed_target_step = nn.Identity()
            self.embed_total_step = nn.Identity()

        # The FLEX diffusion path is purely convolutional when use_transf=False.
        # Replace the inherited transformer decoder stack with parameter-free
        # modules so DDP does not wait for gradients on never-used parameters.
        if not self.use_transf:
            self.tr_blocks = nn.ModuleList()
            self.norm = nn.Identity()

    def _build_condition_embedding(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            fluid_condition: Optional[torch.Tensor],
            target_interp_step: Optional[torch.Tensor],
            total_interp_steps: Optional[torch.Tensor],
            scalar_cond: Optional[Dict[str, torch.Tensor]],
        ) -> torch.Tensor:
        time_token = self.embed_diff_time(timesteps).to(x.dtype)
        if self.use_scalar_film and scalar_cond is not None:
            _, _, scalar_emb = self.scalar_conditioner(
                scalar_cond["tau"],
                scalar_cond["dt"],
                scalar_cond["reynolds"],
            )
            time_token = time_token + scalar_emb.to(x.dtype)
        else:
            if fluid_condition is not None:
                time_token = time_token + self.embed_re(fluid_condition).to(x.dtype)
            if target_interp_step is not None:
                time_token = time_token + self.embed_target_step(target_interp_step).to(x.dtype)
            if total_interp_steps is not None:
                time_token = time_token + self.embed_total_step(total_interp_steps).to(x.dtype)
        return time_token

    def _maybe_inject_spatial(
            self,
            x: torch.Tensor,
            spatial_cond_maps: Optional[Dict[int, torch.Tensor]],
        ) -> torch.Tensor:
        if (not self.use_spatial_cond) or (spatial_cond_maps is None) or (x.ndim != 4):
            return x
        resolution = int(x.shape[-1])
        if resolution not in self.spatial_cond_scales:
            return x
        cond_map = spatial_cond_maps.get(resolution)
        if cond_map is None:
            return x
        key = f"{int(x.shape[1])}_{int(cond_map.shape[1])}"
        if key not in self.spatial_injections:
            return x
        return self.spatial_injections[key](x, cond_map)

    def forward(
            self,
            x: torch.Tensor,
            skips,
            cond,
            cond_skips,
            timesteps: torch.Tensor,
            fluid_condition: Optional[torch.Tensor] = None,
            target_interp_step: Optional[torch.Tensor] = None,
            total_interp_steps: Optional[torch.Tensor] = None,
            scalar_cond: Optional[Dict[str, torch.Tensor]] = None,
            spatial_cond_maps: Optional[Dict[int, torch.Tensor]] = None,
        ) -> torch.Tensor:
        time_token = self._build_condition_embedding(
            x=x,
            timesteps=timesteps,
            fluid_condition=fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=total_interp_steps,
            scalar_cond=scalar_cond,
        )

        if x.ndim == 3:
            x[:, 0, :] = x[:, 0, :] + time_token
            if self.use_transf and cond is not None:
                x = x + cond
            for blk in self.tr_blocks:
                skip = skips.pop()
                if self.use_transf and cond_skips is not None:
                    if self.skip:
                        skip = skip + 0.1 * cond_skips.pop()
                    else:
                        cond_skips.pop()
                x = blk(x, skip)
            x = self.norm(x)
            x = x[:, self.extras:, :]
            x = unpatchify(x, self.embed_dim)
        else:
            x = self._maybe_inject_spatial(x, spatial_cond_maps)

        for module in self.output_blocks:
            skip = skips.pop()
            if cond_skips is not None:
                skip = skip + cond_skips.pop() * 0.1
            x = module(torch.cat([x, skip], dim=1), time_token)
            x = self._maybe_inject_spatial(x, spatial_cond_maps)

        return self.final_layer(x)


def FLEX(
        image_size: int = 256,
        in_channels: int = 1,
        out_channels: int = 1,
        model_size: str = "small",
        mlp_ratio: int = 4,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.2,
        norm_layer=nn.LayerNorm,
        use_checkpoint: bool = False,
        skip: bool = True,
        use_transf: bool = False,
        use_scalar_film: bool = True,
        use_spatial_cond: bool = True,
        spatial_cond_scales: Optional[Sequence[int]] = None,
        spatial_cond_mode: str = "gated",
    ):
    if model_size == "small":
        model_channels = [64, 128, 256, 512]
        decoder_res_blocks = [2, 2, 2, 2]
        encoder_res_blocks = [2, 2, 2, 2]
        depth = 13
        num_heads = 8
    elif model_size == "medium":
        model_channels = [64, 128, 256, 512]
        decoder_res_blocks = [2, 3, 3, 4]
        encoder_res_blocks = [2, 3, 3, 4]
        depth = 13
        num_heads = 8
    elif model_size == "big":
        model_channels = [128, 256, 512, 1152]
        decoder_res_blocks = [2, 3, 3, 3]
        encoder_res_blocks = [2, 3, 3, 3]
        depth = 21
        num_heads = 16
    else:
        raise ValueError("--- Size not found! ---")

    base_encoder = ConditionAwareEncoder(
        img_size=image_size,
        in_chans=in_channels,
        in_conds=1,
        model_channels=model_channels,
        num_res_blocks=encoder_res_blocks,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        attn_drop=attn_drop,
        mlp_drop=mlp_drop,
        norm_layer=norm_layer,
        use_checkpoint=use_checkpoint,
        use_transf=use_transf,
        use_scalar_film=use_scalar_film,
        use_spatial_cond=use_spatial_cond,
        spatial_cond_scales=spatial_cond_scales,
        spatial_cond_mode=spatial_cond_mode,
    )

    shared_spatial_encoder = SharedSpatialConditionEncoder(
        in_channels=in_channels,
        model_channels=model_channels,
    )

    base_decoder = ConditionAwareDecoder(
        img_size=image_size,
        out_chans=out_channels,
        model_channels=model_channels,
        num_res_blocks=decoder_res_blocks,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        attn_drop=attn_drop,
        mlp_drop=mlp_drop,
        norm_layer=norm_layer,
        use_checkpoint=use_checkpoint,
        use_transf=use_transf,
        skip=skip,
        use_scalar_film=use_scalar_film,
        use_spatial_cond=use_spatial_cond,
        spatial_cond_scales=spatial_cond_scales,
        spatial_cond_mode=spatial_cond_mode,
    )

    return base_encoder, shared_spatial_encoder, None, base_decoder
