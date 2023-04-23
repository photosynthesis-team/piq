r"""This module implements CLIP-IQA metric in PyTorch.

The metric is proposed in:
"Exploring CLIP for Assessing the Look and Feel of Images"
by Jianyi Wang, Kelvin C.K. Chan and Chen Change Loy.
AAAI 2023.
https://arxiv.org/abs/2207.12396

This implementation is inspired by the offisial implementation but avoids using MMCV and MMEDIT libraries.
Ref url: https://github.com/IceClear/CLIP-IQA
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
    r"""Creates a criterion that measures image quality based on a general notion of text-to-image similarity
    learned by the CLIP[1] model during its large-scale pre-training on a large dataset with paired texts and images. 

    The method is based on the idea that two antonyms ("Good photo" and "Bad photo") can be used as anchors in the
    text embedding space representing good and bad images in terms of their image quality.

    After the anchors are defined, one can use them to determine the quality of a given image in the following way:
    1. Compute the image embedding of the image of interest using the pre-trained CLIP model;
    2. Compute the text embeddings of the selected anchor antonyms;
    3. Compute the angle (cosine similarity) between the image embedding (1) and both text embeddings (2);
    4. Compute the Softmax of cosine similarities (3) -> CLIP-IQA[2] score.

    This method is proposed to eliminate the linguistic ambiguity of the naive approach 
    (using a single prompt, e.g., "Good photo").

    This method has an extension called CLIP-IQA+[2] proposed in the same research paper. 
    It uses the same approach but also fine-tunes the CLIP weights using the CoOp[3] fine-tuning algorithm.

    Args:
        prompt_pairs: Alternative antonyms to be used as textual anchors.
        data_range: Maximum value range of images (usually 1.0 or 255).

    Examples:
        >>> from piq import CLIPIQA
        >>> clipiqa = CLIPIQA()
        >>> x = torch.rand(1, 224, 224, 3)
        >>> score = clipiqa(x)

    References:
        [1] Radford, Alec, et al. "Learning transferable visual models from natural language supervision." 
        International conference on machine learning. PMLR, 2021.
        [2] Wang, Jianyi, Kelvin CK Chan, and Chen Change Loy. "Exploring CLIP for Assessing the Look
        and Feel of Images." arXiv preprint arXiv:2207.12396 (2022).
        [3] Zhou, Kaiyang, et al. "Learning to prompt for vision-language models." International 
        Journal of Computer Vision 130.9 (2022): 2337-2348.

    Warning: 
        Please note that this implementation assumes batch size = 1. 
        Chosing different batch size may hurt the performance.
    """
    def __init__(self,
                 prompt_pairs: Optional[List[Tuple[str, str]]] = None,
                 data_range: Union[float, int] = 1.
                 ) -> None:
        super().__init__()

        self.feature_extractor = load().eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        tokenizer = SimpleTokenizer()
        if prompt_pairs is None:
            tokens = tokenize(["Good photo.", "Bad photo."], tokenizer=tokenizer)
        else:
            tokens = tokenize(prompt_pairs, tokenizer=tokenizer)

        anchors = self.feature_extractor.encode_text(tokens).float()
        self.anchors = anchors / anchors.norm(dim=-1, keepdim=True)

        self.data_range = data_range
        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = logit_scale.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computation of CLIP-IQA metric for a given image :math:`x`.

        Args: 
            x: An input tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            The value of CLI-IQA score in [0, 1] range.
        """
        x = x.permute(0, 3, 1, 2).float() / self.data_range
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)

        self.anchors = self.anchors.to(x)
        self.feature_extractor = self.feature_extractor.to(x)

        with torch.no_grad():
            image_features = self.feature_extractor.encode_image(x, pos_embedding=False).float()

        # Normalized features.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity as logits.
        logits_per_image = self.logit_scale * image_features @ self.anchors.t()

        probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(dim=-1)
        result = probs[..., 0].mean(dim=1, keepdim=True)
        return result.detach()
