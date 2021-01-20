<div align="center">

![piq_logo](https://user-images.githubusercontent.com/15848838/95228385-ed106500-0807-11eb-8584-d3fdbdd47ac7.jpeg)

[![PyPI version][pypi-version-shield]][pypi-version-url]
[![Conda version][conda-version-shield]][conda-version-url]  
![CI flake-8 style check][ci-flake-8-style-check-shield]
![CI testing][ci-testing]
[![codecov][codecov-shield]][codecov-url]  
[![Quality Gate Status][quality-gate-status-shield]][quality-gate-status-url]
[![Reliability Rating][reliability-rating-badge]][reliability-rating-url]
</div>

<!-- ABOUT THE PROJECT -->

Collection of measures and metrics for image quality assessment.
- Unified interface, which is easy to use and extend.
- Written on pure PyTorch with bare minima of additional dependencies.
- Extensive user input validation. You code will not crash in the middle of the training.
- Fast (GPU computations available) and reliable.
- Most metrics can be backpropagated for model optimization.
- Supports python 3.6-3.8.


<!-- GETTING STARTED -->
### Getting started

The first group of metrics takes image or images as input, e.g., PSNR, SSIM, BRISQUE.
We have a functional interface, which returns a metric value, and a class interface, which allows us to use any metric 
as a loss function.

```python
import torch
from piq import ssim, SSIMLoss

prediction = torch.rand(4, 3, 256, 256, requires_grad=True)
target = torch.rand(4, 3, 256, 256)

ssim_index: torch.Tensor = ssim(prediction, target, data_range=1.)

loss = SSIMLoss(data_range=1.)
output: torch.Tensor = loss(prediction, target)
output.backward()
```

The second group takes a list of image features, e.g., IS, FID, KID.
Image features can be extracted by some feature extractor network separately or by using the `compute_feats` method of a
class.   
**Important note**: `compute_feats` consumes a data loader of a predefined format.

```python
import torch
from torch.utils.data import DataLoader
from piq import FID

first_dl, second_dl = DataLoader(), DataLoader()
fid_metric = FID()
first_feats = fid_metric.compute_feats(first_dl)
second_feats = fid_metric.compute_feats(second_dl)
fid: torch.Tensor = fid_metric(first_feats, second_feats)
```

If you already have image features, use the class interface for score computation:

```python
import torch
from piq import FID

prediction_feats = torch.rand(10000, 1024)
target_feats = torch.rand(10000, 1024)
msid_metric = MSID()
msid: torch.Tensor = msid_metric(prediction_feats, target_feats)
```

For a full list of examples, see [image metrics](examples/image_metrics.py) and [feature metrics](examples/feature_metrics.py) examples.

<!-- IMAGE METRICS -->
### Full Reference metrics

| Acronym | Year | Metric                                                                                               |
|:-------:|:----:|------------------------------------------------------------------------------------------------------|
|    -    | 2016 | [Content score](https://arxiv.org/abs/1508.06576)                                                    |
|    -    | 2016 | [Style score](https://arxiv.org/abs/1508.06576)                                                      |
|  DISTS  | ? | [Deep Image Structure and Texture Similarity](https://arxiv.org/abs/2004.07728)                      |
|  FSIM   | ? | [Feature Similarity Index Measure](https://www4.comp.polyu.edu.hk/~cslzhang/IQA/TIP_IQA_FSIM. pdf)      |
|   GMSD  | 2013 | [Gradient Magnitude Similarity Deviation](https://arxiv.org/abs/1308.3052)                           |
| HaarPSI | 2018 | [Haar Perceptual Similarity Index](https://arxiv.org/abs/1607.06140)                                 |
|  LPIPS  | 2018 | [Learned Perceptual Image Patch Similarity](https://arxiv.org/abs/1801.03924)                        |
|   MDSI  | 2016 | [Mean Deviation Similarity Index](https://arxiv.org/abs/1608.07433)                                  |
| MS-SSIM | 2004 | [Multi-Scale Structural Similarity](https://ieeexplore.ieee.org/abstract/document/1292216/)          |
| MS-GMSD | 2017 | [Multi-Scale Gradient Magnitude Similiarity Deviation](https://ieeexplore.ieee.org/document/7952357) |
|   PSNR  |   ?  | [Peak Signal-to-Noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)               |
|    TV   | 1937 | [Total Variation](https://en.wikipedia.org/wiki/Total_variation)                                     |
|   SSIM  | 2004 | [Structural Similarity](https://en.wikipedia.org/wiki/Structural_similarity)                         |
|   GMSD  | 2013 | [Gradient Magnitude Similarity Deviation](https://arxiv.org/abs/1308.3052)                           |
| PieAPP | ? | [Perceptual Image-Error Assessment through Pairwise Preference](https://arxiv.org/abs/1806.02067)        |
| VIFp | ? | [Visual Information Fidelity](https://live.ece.utexas.edu/research/Quality/VIF.htm)            |
| VSI | ? | [Visual Saliency-induced Index](https://ieeexplore.ieee.org/document/6873260)        |


### No Reference metrics

| Acronym | Year | Metric                                                                                               |
|:-------:|:----:|------------------------------------------------------------------------------------------------------|
| BRISQUE | 2012 | [Blind/Referenceless Image Spatial Quality Evaluator](https://ieeexplore.ieee.org/document/6272356)  |



<!-- FEATURE METRICS -->
### Feature metrics

| Acronym | Year | Metric                                                                                               |
|:-------:|:----:|------------------------------------------------------------------------------------------------------|
| FID | ? | [Frechet Inception Distance](https://arxiv.org/abs/1706.08500)  |
| GS  | ? | [Geometry Score](https://arxiv.org/abs/1802.02664) | 
|  IS  |  ? | [Inception Score](https://arxiv.org/abs/1606.03498) |
|  KID |  ? | [Kernel Inception Distance](https://arxiv.org/abs/1801.01401)
| MSID | ?  | [Multi-Scale Intrinsic Distance](https://arxiv.org/abs/1905.11141) |



### Benchmark

`c` means chromatic version of metric.

| Acronym | TID2013<br />SRCC / KRCC (piq) | TID2013 <br />SRCC / KRCC (orig)  | KADID10k<br /> SRCC / KRCC (piq) | KADID10k<br /> SRCC / KRCC (original) |
|:-------:|:----:|:----:|:----:|:----:|
| PSNR | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| IW-PSNR | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| SSIM | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| MS-SSIM | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| IW-SSIM | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| SR-SIM | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| VIFp | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| GMSD | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| MS-GMSD | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| MS-GMSDc | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| FSIM| 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| FSIMc | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| VSI | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| MDSI | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| HaarPSI | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| LPIPS-VGG | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| PieAPP | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |
| DISTS | 0.6869 / 0.4958 | 0.687 / 0.496 [source](link) |


### Overview

*PyTorch Image Quality* (former [PhotoSynthesis.Metrics](https://pypi.org/project/photosynthesis-metrics/0.4.0/)) helps you to concentrate on your experiments without the boilerplate code.
The library contains a set of measures and metrics that is continually getting extended.
For measures/metrics that can be used as loss functions, corresponding PyTorch modules are implemented.


#### Installation

`$ pip install piq`

`$ conda install piq -c photosynthesis-team -c conda-forge -c PyTorch`

If you want to use the latest features straight from the master, clone the repo:
```sh
$ git clone https://github.com/photosynthesis-team/piq.git
```

<!-- ROADMAP -->
#### Roadmap

See the [open issues](https://github.com/photosynthesis-team/piq/issues) for a list of proposed
features and known issues.


<!-- COMMUNITY -->
### Community


<!-- CONTRIBUTING -->
#### Contributing

We appreciate all your contributions. If you plan to:
- contribute back bug-fixes; please do so without any further discussion
- close one of [open issues](https://github.com/photosynthesis-team/piq/issues), please do so if no one has been assigned to it
- contribute new features, utility functions, or extensions; please first open an issue and discuss the feature with us

Please see the [contribution guide](CONTRIBUTING.md) for more information.


<!-- CONTACT -->
#### Contacts

**Sergey Kastryulin** - [@snk4tr](https://github.com/snk4tr) - `snk4tr@gmail.com`  
**Djamil Zakirov** - [@zakajd](https://github.com/zakajd) - `djamilzak@gmail.com`  
**Denis Prokopenko** - [@denproc](https://github.com/denproc) - `d.prokopenko@outlook.com`


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[ci-flake-8-style-check-shield]: https://github.com/photosynthesis-team/piq/workflows/flake-8%20style%20check/badge.svg
[ci-testing]: https://github.com/photosynthesis-team/piq/workflows/testing/badge.svg
[pypi-version-shield]: https://badge.fury.io/py/piq.svg
[pypi-version-url]: https://badge.fury.io/py/piq
[conda-version-shield]: https://anaconda.org/photosynthesis-team/piq/badges/version.svg
[conda-version-url]: https://anaconda.org/photosynthesis-team/piq
[quality-gate-status-shield]: https://sonarcloud.io/api/project_badges/measure?project=photosynthesis-team_photosynthesis.metrics&metric=alert_status
[quality-gate-status-url]: https://sonarcloud.io/dashboard?id=photosynthesis-team_photosynthesis.metrics
[maintainability-raiting-shield]: https://sonarcloud.io/api/project_badges/measure?project=photosynthesis-team_photosynthesis.metrics&metric=sqale_rating

[reliability-rating-badge]: https://sonarcloud.io/api/project_badges/measure?project=photosynthesis-team_photosynthesis.metrics&metric=reliability_rating
[codecov-shield]:https://codecov.io/gh/photosynthesis-team/piq/branch/master/graph/badge.svg
[codecov-url]:https://codecov.io/gh/photosynthesis-team/piq
