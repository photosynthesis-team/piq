r"""This module implements CLIP-IQA metric in PyTorch.

The metric is proposed in:
"Exploring CLIP for Assessing the Look and Feel of Images"
by Jianyi Wang, Kelvin C.K. Chan and Chen Change Loy.
AAAI 2023.
https://arxiv.org/abs/2207.12396

This implementation is inspired by the offisial implementation but avoids using MMCV and MMEDIT libraries.
Ref url: https://github.com/IceClear/CLIP-IQA
"""
import os
import torch

from torch.nn.modules.loss import _Loss
from typing import Union

from piq.feature_extractors import clip
from piq.utils.common import download_tensor, _validate_input


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TOKENS_URL = "https://github.com/photosynthesis-team/piq/releases/download/v0.7.1/clipiqa_tokens.pt"


class CLIPIQA(_Loss):
    r"""Creates a criterion that measures image quality based on a general notion of text-to-image similarity
    learned by the CLIP model (Radford et al., 2021) during its large-scale pre-training on a large dataset
    with paired texts and images.

    The method is based on the idea that two antonyms ("Good photo" and "Bad photo") can be used as anchors in the
    text embedding space representing good and bad images in terms of their image quality.

    After the anchors are defined, one can use them to determine the quality of a given image in the following way:
    1. Compute the image embedding of the image of interest using the pre-trained CLIP model;
    2. Compute the text embeddings of the selected anchor antonyms;
    3. Compute the angle (cosine similarity) between the image embedding (1) and both text embeddings (2);
    4. Compute the Softmax of cosine similarities (3) -> CLIP-IQA score (Wang et al., 2022).

    This method is proposed to eliminate the linguistic ambiguity of the naive approach
    (using a single prompt, e.g., "Good photo").

    This method has an extension called CLIP-IQA+ proposed in the same research paper.
    It uses the same approach but also fine-tunes the CLIP weights using the CoOp
    fine-tuning algorithm (Zhou et al., 2022).

    Note:
        The initial computation of the metric is performed in `float32` and other dtypes (i.e. `float16`, `float64`)
        are not supported. We preserve this behaviour for reproducibility perposes. Also, at the time of writing
        conv2d is not supported for `float16` tensors on CPU.

    Warning:
        In order to avoid implicit dtype conversion and normalization of input tensors, they are copied.
        Note that it may consume extra memory, which might be noticeable on large batch sizes.

    Args:
        data_range: Maximum value range of images (usually 1.0 or 255).

    Examples:
        >>> from piq import CLIPIQA
        >>> clipiqa = CLIPIQA()
        >>> x = torch.rand(1, 3, 224, 224)
        >>> score = clipiqa(x)

    References:
        Radford, Alec, et al. "Learning transferable visual models from natural language supervision."
        International conference on machine learning. PMLR, 2021.

        Wang, Jianyi, Kelvin CK Chan, and Chen Change Loy. "Exploring CLIP for Assessing the Look
        and Feel of Images." arXiv preprint arXiv:2207.12396 (2022).

        Zhou, Kaiyang, et al. "Learning to prompt for vision-language models." International
        Journal of Computer Vision 130.9 (2022): 2337-2348.
    """
    def __init__(self, data_range: Union[float, int] = 1.) -> None:
        super().__init__()

        self.feature_extractor = clip.load().eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Pre-computed tokens for prompt pairs: "Good photo.", "Bad photo.".
        tokens = download_tensor(TOKENS_URL, os.path.expanduser("~/.cache/clip"))

        anchors = self.feature_extractor.encode_text(tokens).float()
        anchors = anchors / anchors.norm(dim=-1, keepdim=True)

        self.data_range = float(data_range)
        default_mean = torch.tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        default_std = torch.tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)
        self.logit_scale = self.feature_extractor.logit_scale.exp()

        # Take advantage of Torch buffers. CLIPIQA.to(device) will move these to the device as well.
        self.register_buffer("anchors", anchors)
        self.register_buffer("default_mean", default_mean)
        self.register_buffer("default_std", default_std)

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        r"""Computation of CLIP-IQA metric for a given image :math:`x`.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
                The metric is designed in such a way that it expects:
                - A 4D PyTorch tensor;
                - The tensor might have flexible data ranges depending on `data_range` value;
                - The tensor must have channels first format.

        Returns:
            The value of CLI-IQA score in [0, 1] range.
        """
        _validate_input([x_input], dim_range=(4, 4), data_range=(0., 255.), check_for_channels_first=True)

        x = x_input.clone()
        x = x.float() / self.data_range
        x = (x - self.default_mean) / self.default_std

        # Device for nn.Module cannot be cached through the buffer so it has to be done here.
        self.feature_extractor = self.feature_extractor.to(x)

        with torch.no_grad():
            image_features = self.feature_extractor.encode_image(x, pos_embedding=False).float()

        # Normalized features.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity as logits.
        logits_per_image = self.logit_scale * image_features @ self.anchors.t()

        probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(dim=-1)
        result = probs[..., 0]
        return result.detach()
