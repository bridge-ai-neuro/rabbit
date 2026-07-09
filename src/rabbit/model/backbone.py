"""Pretrained speech backbone (wav2vec2 / WavLM) + optional LoRA adapters.

This module is the audio half of RABBiT: raw 16 kHz waveform in, frame-rate
hidden states out. The feature encoder (CNN) is always frozen. The transformer
half can be either:

  * **LoRA-adapted** (``lora.enabled=True`` and ``lora.rank > 0``) — only LoRA
    parameters update during training; the canonical RABBiT setup.
  * **Frozen** (``lora.enabled=False`` and you do not unfreeze it elsewhere) —
    feature-extractor baseline.
  * **Fully fine-tuned** (``lora.enabled=False`` and the backbone is left
    trainable by the optimiser group setup) — uncommon for RABBiT.

The self-supervised contrastive head (Wav2Vec2ForPreTraining masked-prediction
loss) is intentionally *not* loaded here. It is unused in every released
RABBiT checkpoint (the SSL loss weight is 0 throughout the paper). If you
want to enable SSL during training, swap in ``Wav2Vec2ForPreTraining`` at the
trainer level.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WavLMModel,
)


def is_wavlm(model_name: str) -> bool:
    return "wavlm" in model_name.lower()


class SpeechBackbone(nn.Module):
    """HuggingFace wav2vec2 or WavLM with optional LoRA, frozen feature extractor.

    Args:
        model_name: HF identifier (e.g. ``facebook/wav2vec2-base-960h``,
            ``microsoft/wavlm-large``).
        lora_rank: 0 = no LoRA, full backbone trainable.
            >0 = wrap attention projections with rank-``lora_rank`` adapters.
        lora_alpha: LoRA scaling factor (typically ``2 * lora_rank``).
        lora_dropout: dropout applied inside LoRA adapters.

    Attributes:
        hidden_size: dimensionality of the per-frame hidden state
            (768 for wav2vec2-base, 1024 for wav2vec2-large and WavLM-large).
        processor: HF processor for audio preprocessing.
        sampling_rate: native input sample rate (16 kHz for both backbones).
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        sampling_rate: int = 16_000,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.sampling_rate = sampling_rate
        self.is_wavlm = is_wavlm(model_name)

        if self.is_wavlm:
            base = WavLMModel.from_pretrained(model_name)
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        else:
            base = Wav2Vec2Model.from_pretrained(model_name)
            self.processor = Wav2Vec2Processor.from_pretrained(
                model_name, return_tensors="pt", return_attention_mask=True,
            )

        # Feature encoder (CNN) is always frozen — only the transformer is tuned.
        base.freeze_feature_encoder()
        self.hidden_size = base.config.hidden_size
        self._feat_lengths_fn = base._get_feat_extract_output_lengths
        self._feature_attn_mask_fn = base._get_feature_vector_attention_mask

        if lora_rank > 0:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=[
                    "attention.q_proj", "attention.k_proj",
                    "attention.v_proj", "attention.out_proj",
                ],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone = get_peft_model(base, lora_config)
        else:
            self.backbone = base

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the speech transformer.

        Args:
            input_values: (B, T_audio) waveform at ``self.sampling_rate``.
            attention_mask: (B, T_audio) optional 1/0 mask over audio samples.

        Returns:
            hidden_states: (B, T_frames, hidden_size) frame-rate features.
            feature_attention_mask: (B, T_frames) bool mask over frames, or
                None when no input attention mask was passed.
        """
        out = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = out.last_hidden_state

        feature_attention_mask = None
        if attention_mask is not None:
            feature_attention_mask = self._feature_attn_mask_fn(
                hidden_states.shape[1], attention_mask,
            )
        return hidden_states, feature_attention_mask

    def feat_extract_output_lengths(self, input_length: int) -> int:
        return int(self._feat_lengths_fn(input_length))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
