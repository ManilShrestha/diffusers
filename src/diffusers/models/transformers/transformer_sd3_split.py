
import torch
import torch.nn as nn

import diffusers

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.modeling_utils import ModelMixin
from ...models.normalization import AdaLayerNormContinuous

from ..embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed

# from typing import Any, Dict, List, Optional, Union
# from ..modeling_outputs import Transformer2DModelOutput
# from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
# from ...models.attention import JointTransformerBlock
# from ...models.attention_processor import Attention, AttentionProcessor, FusedJointAttnProcessor2_0


class SD3Transformer2DModelClientSplit(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    Client side transformer. Preferably first block will be processed in client side.
    Hence, latent noise prediction and states remain hidden
    """
    def __init__(self, config, blocks, time_text_embed_state_dict, context_embedder_state_dict, pos_embed_state_dict):
        super().__init__()
        self.config_temp= config
        self.out_channels = self.config_temp.out_channels if self.config_temp.out_channels is not None else self.config_temp.in_channels
        self.inner_dim = self.config_temp.num_attention_heads * self.config_temp.attention_head_dim

        self.pos_embed = PatchEmbed(
            height=self.config_temp.sample_size,
            width=self.config_temp.sample_size,
            patch_size=self.config_temp.patch_size,
            in_channels=self.config_temp.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=self.config_temp.pos_embed_max_size,
        )
        self.pos_embed.load_state_dict(pos_embed_state_dict)

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config_temp.pooled_projection_dim
        )
        self.time_text_embed.load_state_dict(time_text_embed_state_dict)

        self.context_embedder = nn.Linear(self.config_temp.joint_attention_dim, self.config_temp.caption_projection_dim)
        self.context_embedder.load_state_dict(context_embedder_state_dict)

        self.transformer_blocks = blocks

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections, timestep):
        height_, width_ = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if not isinstance(self.transformer_blocks, diffusers.models.attention.JointTransformerBlock):
            for block in self.transformer_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )
        else:
            encoder_hidden_states, hidden_states = self.transformer_blocks(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
            )

        return hidden_states, encoder_hidden_states, temb, height_, width_

    @property
    def config(self):
        return self.config_temp


class SD3Transformer2DModelServerSplit(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    def __init__(self, config, blocks, norm_out_state_dict, proj_out_state_dict, has_last_block=False):
        super().__init__()
        self.config_temp= config
        self.is_last_block = is_last_block

        self.inner_dim = self.config_temp.num_attention_heads * self.config_temp.attention_head_dim
        self.transformer_blocks = blocks

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.norm_out.load_state_dict(norm_out_state_dict)

        self.proj_out = nn.Linear(self.inner_dim, self.config_temp.patch_size * self.config_temp.patch_size * self.config_temp.out_channels, bias=True)
        self.proj_out.load_state_dict(proj_out_state_dict)

    def forward(self, hidden_states, encoder_hidden_states, temb, height_, width_):
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
            )
        
        # If not last block, return the intermediate data for next blocks to process
        if not self.is_last_block:
            return hidden_states, encoder_hidden_states, temb, height_, width_

        # If it is the last block, further process and return the noise prediction
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        patch_size = self.config_temp.patch_size

        height = height_ // patch_size
        width = width_ // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.config_temp.out_channels)
        )
        
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.config_temp.out_channels, height * patch_size, width * patch_size)
        )

        return (output,)

    @property
    def config(self):
        return self.config_temp