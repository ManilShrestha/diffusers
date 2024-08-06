
import torch
import torch.nn as nn

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


class SD3Transformer2DModelPart1(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    def __init__(self, config, blocks):
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
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config_temp.pooled_projection_dim
        )
        self.context_embedder = nn.Linear(self.config_temp.joint_attention_dim, self.config_temp.caption_projection_dim)

        self.transformer_blocks = blocks

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections, timestep):
        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
            )

        return hidden_states, encoder_hidden_states, temb

    @property
    def config(self):
        return self.config_temp

class SD3Transformer2DModelPart2(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    def __init__(self, config, blocks):
        super().__init__()
        self.config_temp= config
        self.inner_dim = self.config_temp.num_attention_heads * self.config_temp.attention_head_dim

        self.transformer_blocks = blocks

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, self.config_temp.patch_size * self.config_temp.patch_size * self.config_temp.out_channels, bias=True)

    def forward(self, hidden_states, encoder_hidden_states, temb):
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        patch_size = self.config_temp.patch_size

        height = self.config_temp.sample_size // patch_size
        width = self.config_temp.sample_size // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.config_temp.out_channels)
        )
        print(hidden_states.shape)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.config_temp.out_channels, height * patch_size, width * patch_size)
        )

        return (output,)

    @property
    def config(self):
        return self.config_temp