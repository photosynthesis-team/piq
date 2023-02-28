r"""CLIP-IQA metric, proposed by

Exploring CLIP for Assessing the Look and Feel of Images.
Jianyi Wang Kelvin C.K. Chan Chen Change Loy.
AAAI 2023.

Ref url: https://github.com/IceClear/CLIP-IQA
"""
import clip
import collections
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple, List, Optional, Union
from itertools import repeat

from pyiqa.archs.arch_util import load_file_from_url

from piq.feature_extractors.clip import load


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def symm_pad(im: torch.Tensor, padding: Tuple[int, int, int, int]):
    """Symmetric padding same as tensorflow.
    Ref: https://discuss.pytorch.org/t/symmetric-padding/19866/3
    """
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    def reflect(x, minx, maxx):
        """ Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length """
        rng = maxx - minx
        double_rng = 2 * rng
        mod = np.fmod(x - minx, double_rng)
        normed_mod = np.where(mod < 0, mod + double_rng, mod)
        out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]


def excact_padding_2d(x, kernel, stride=1, dilation=1, mode='same'):
    assert len(x.shape) == 4, f'Only support 4D tensor input, but got {x.shape}'
    kernel = to_2tuple(kernel)
    stride = to_2tuple(stride)
    dilation = to_2tuple(dilation)
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride[0])
    w2 = math.ceil(w / stride[1])
    pad_row = (h2 - 1) * stride[0] + (kernel[0] - 1) * dilation[0] + 1 - h
    pad_col = (w2 - 1) * stride[1] + (kernel[1] - 1) * dilation[1] + 1 - w
    pad_l, pad_r, pad_t, pad_b = (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2)

    mode = mode if mode != 'same' else 'constant'
    if mode != 'symmetric':
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    elif mode == 'symmetric':
        x = symm_pad(x, (pad_l, pad_r, pad_t, pad_b))

    return x



def extract_2d_patches(x, kernel, stride=1, dilation=1, padding='same'):
    """
    Ref: https://stackoverflow.com/a/65886666
    """
    b, c, h, w = x.shape
    if padding != 'none':
        x = excact_padding_2d(x, kernel, stride, dilation, mode=padding)

    # Extract patches
    patches = F.unfold(x, kernel, dilation, stride=stride)
    b, _, pnum = patches.shape
    patches = patches.transpose(1, 2).reshape(b, pnum, c, kernel, kernel)
    return 


default_url = {
    'clipiqa+': 
    'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/CLIP-IQA+_learned_prompts-603f3273.pth',
}

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class PromptLearner(nn.Module):
    """
    Disclaimer:
        This implementation follows exactly the official codes in: https://github.com/IceClear/CLIP-IQA. 
        We have no idea why some tricks are implemented like this, which include
            1. Using n_ctx prefix characters "X"
            2. Appending extra "." at the end
            3. Insert the original text embedding at the middle
    """
    def __init__(self, clip_model, n_ctx=16) -> None:
        super().__init__()

        # For the following codes about prompts, we follow the official codes to get the same results
        prompt_prefix = " ".join(["X"] * n_ctx) + ' '
        init_prompts = [prompt_prefix + 'Good photo..', prompt_prefix + 'Bad photo..']
        with torch.no_grad():
            txt_token = clip.tokenize(init_prompts)
            self.tokenized_prompts = txt_token
            init_embedding = clip_model.token_embedding(txt_token)

        self.ctx = nn.Parameter(torch.load(load_file_from_url(default_url['clipiqa+'])))
        self.n_ctx = n_ctx

        self.n_cls = len(init_prompts)
        self.name_lens = [3, 3]  # hard coded length, which does not include the extra "." at the end

        self.register_buffer("token_prefix", init_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", init_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def get_prompts_with_middel_class(self,):

        ctx = self.ctx.to(self.token_prefix)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        half_n_ctx = self.n_ctx // 2
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i: i + 1, :, :]
            class_i = self.token_suffix[i: i + 1, :name_len, :]
            suffix_i = self.token_suffix[i: i + 1, name_len:, :]
            ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
            ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
            prompt = torch.cat(
                [
                    prefix_i,     # (1, 1, dim)
                    ctx_i_half1,  # (1, n_ctx//2, dim)
                    class_i,      # (1, name_len, dim)
                    ctx_i_half2,  # (1, n_ctx//2, dim)
                    suffix_i,     # (1, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)
        return prompts

    def forward(self, clip_model):
        prompts = self.get_prompts_with_middel_class()
        # self.get_prompts_with_middel_class
        x = prompts + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ clip_model.text_projection

        return x


class CLIPIQA(nn.Module):
    def __init__(self,
                 model_type: str = 'clipiqa',
                 backbone: str ='RN50',
                 device: str = 'cuda',
                 prompt_pairs: Optional[List[Tuple[str, str]]] = None,
                 data_range: Union[float, int] = 1.
                 ) -> None:
        super().__init__()

        self.clip_model = load(backbone, device=device).eval()

        self.prompt_pairs = prompt_pairs
        if self.prompt_pairs is None:
            self.prompt_pairs = clip.tokenize([
                'Good image', 'bad image'
            ])

        self.model_type = model_type
        if model_type == 'clipiqa+':
            self.prompt_learner = PromptLearner(self.clip_model)

        self.data_range = data_range
        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= float(self.data_range)
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)

        with torch.no_grad():
            if self.model_type == 'clipiqa':
                prompts = self.prompt_pairs.to(x.device)
                logits_per_image, logits_per_text = self.clip_model(x, prompts, pos_embedding=False)
            elif self.model_type == 'clipiqa+':
                learned_prompt_feature = self.prompt_learner(self.clip_model)
                logits_per_image, logits_per_text = self.clip_model(
                    x, None, text_features=learned_prompt_feature, pos_embedding=False)

        probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(dim=-1)
        return probs[..., 0].mean(dim=1, keepdim=True)
