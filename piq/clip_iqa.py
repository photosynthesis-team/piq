r""" This module implements CLIP-IQA metric in PyTorch.

The metric is proposed in:
"Exploring CLIP for Assessing the Look and Feel of Images"
by Jianyi Wang, Kelvin C.K. Chan and Chen Change Loy.
AAAI 2023.

This implementation is inspired by the offisial implementation but avoids using MMCV and MMEDIT libraries.
Ref url: https://github.com/IceClear/CLIP-IQA

WARNING: Please note that this implementation assumes batch size = 1. 
Chosing different batch size may hurt the performance.
"""
import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, List, Optional, Union

from piq.feature_extractors.clip import load
from piq.tokenizers.clip import SimpleTokenizer, tokenize


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPIQA(nn.Module):
    def __init__(self,
                 device: str = 'cuda',
                 prompt_pairs: Optional[List[Tuple[str, str]]] = None,
                 data_range: Union[float, int] = 1.
                 ) -> None:
        super().__init__()

        self.feature_extractor = load(device=device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        tokenizer = SimpleTokenizer()
        if prompt_pairs is None:
            tokens = tokenize(["Good photo.", "Bad photo."], tokenizer=tokenizer)
        else:
            tokens = tokenize(prompt_pairs, tokenizer=tokenizer)

        tokens = tokens.to(device)

        anchors = self.feature_extractor.encode_text(tokens).float()
        self.anchors = anchors / anchors.norm(dim=-1, keepdim=True)

        self.device = device
        self.data_range = data_range
        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = logit_scale.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[0].to(self.device)
        x = x.permute(2, 0, 1).float() / self.data_range
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)

        with torch.no_grad():
            image_features = self.feature_extractor.encode_image(x, pos_embedding=False).float()

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = self.logit_scale * image_features @ self.anchors.t()

        probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(dim=-1)
        result = probs[..., 0].mean(dim=1, keepdim=True)
        return result.detach()
