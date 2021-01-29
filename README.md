<div align="center">  

![piq_logo](https://user-images.githubusercontent.com/15848838/95228385-ed106500-0807-11eb-8584-d3fdbdd47ac7.jpeg) 

[![PyPI version][pypi-version-shield]][pypi-version-url] [![Conda version][conda-version-shield]][conda-version-url] ![CI flake-8 style check][ci-flake-8-style-check-shield] ![CI testing][ci-testing] [![codecov][codecov-shield]][codecov-url] [![Quality Gate Status][quality-gate-status-shield]][quality-gate-status-url]

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
|    TV   | 1937 | [Total Variation](https://en.wikipedia.org/wiki/Total_variation)                                     |
|   PSNR  |   -  | [Peak Signal-to-Noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)               |
|   SSIM  | 2003 | [Structural Similarity](https://en.wikipedia.org/wiki/Structural_similarity)                         |
| MS-SSIM | 2004 | [Multi-Scale Structural Similarity](https://ieeexplore.ieee.org/abstract/document/1292216/)          |
|   VIFp  | 2004 | [Visual Information Fidelity](https://ieeexplore.ieee.org/document/1576816)                          |
|   FSIM  | 2011 | [Feature Similarity Index Measure](https://ieeexplore.ieee.org/document/5705575)                     |
| IW-PSNR | 2011 | [Information Weighted PSNR](https://ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf)               |
| IW-SSIM | 2011 | [Information Weighted SSIM](https://ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf)               |
| SR-SIM  | 2012 | [Spectral Residual Based Similarity](https://sse.tongji.edu.cn/linzhang/ICIP12/ICIP-SR-SIM.pdf)      |
|   GMSD  | 2013 | [Gradient Magnitude Similarity Deviation](https://arxiv.org/abs/1308.3052)                           |
|   VSI   | 2014 | [Visual Saliency-induced Index](https://ieeexplore.ieee.org/document/6873260)                        |
|    -    | 2016 | [Content Score](https://arxiv.org/abs/1508.06576)                                                    |
|    -    | 2016 | [Style Score](https://arxiv.org/abs/1508.06576)                                                      |
| HaarPSI | 2016 | [Haar Perceptual Similarity Index](https://arxiv.org/abs/1607.06140)                                 |
|   MDSI  | 2016 | [Mean Deviation Similarity Index](https://arxiv.org/abs/1608.07433)                                  |
| MS-GMSD | 2017 | [Multi-Scale Gradient Magnitude Similiarity Deviation](https://ieeexplore.ieee.org/document/7952357) |
|  LPIPS  | 2018 | [Learned Perceptual Image Patch Similarity](https://arxiv.org/abs/1801.03924)                        |
|  PieAPP | 2018 | [Perceptual Image-Error Assessment through Pairwise Preference](https://arxiv.org/abs/1806.02067)    |
|  DISTS  | 2020 | [Deep Image Structure and Texture Similarity][dists]                      |


### No Reference metrics

| Acronym | Year | Metric                                                                                               |
|:-------:|:----:|------------------------------------------------------------------------------------------------------|
| BRISQUE | 2012 | [Blind/Referenceless Image Spatial Quality Evaluator](https://ieeexplore.ieee.org/document/6272356)  |



<!-- FEATURE METRICS -->
### Feature metrics

|Acronym| Year | Metric                                                            |
|:-----:|:----:|-------------------------------------------------------------------|
| IS    | 2016 | [Inception Score](https://arxiv.org/abs/1606.03498)               |
| FID   | 2017 | [Frechet Inception Distance](https://arxiv.org/abs/1706.08500)    |
| GS    | 2018 | [Geometry Score](https://arxiv.org/abs/1802.02664)                | 
| KID   | 2018 | [Kernel Inception Distance](https://arxiv.org/abs/1801.01401)     |
| MSID  | 2019 | [Multi-Scale Intrinsic Distance](https://arxiv.org/abs/1905.11141)|



### Benchmark

As part of our library we provide code to benchmark all metrics on a set of common Mean Opinon Scores databases.
Currently only [TID2013][tid2013] and [KADID10k][kadid10k] are supported. You need to download them separately and provide path to images as an argument to the script.

Here is an example how to evaluate SSIM and MS-SSIM metrics on TID2013 dataset:
```bash
python3 tests/results_benchmark.py --dataset tid2013 --metrics SSIM MS-SSIM --path ~/datasets/tid2013 --batch_size 16
```

We report [Spearman's rank correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) (SRCC) and [Kendall rank correlation coefficient](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) (KRCC). We do not report [Pearson linear correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) (PLCC) because it's highly dependent on fitting method and is biased towards simple examples.

`c` means chromatic version.

| Acronym |TID2013<br />SRCC / KRCC (piq)| TID2013<br />SRCC / KRCC | KADID10k<br /> SRCC / KRCC (piq) | KADID10k<br /> SRCC / KRCC |
|:-------:|:---------------:|:---------------------------------:|:---------------:|:-------------------------------:|
|   PSNR  | 0.6869 / 0.4958 | 0.687 / 0.496  [source][tid2013]  | 0.6757 / 0.4876 |     -   /    -   |
|   SSIM  | 0.5544 / 0.3883 | 0.637 / 0.464  [source][tid2013]  | 0.6329 / 0.4571 | 0.718 / 0.532 [source][kadid10k]|
| MS-SSIM | 0.7983 / 0.5965 | 0.787 / 0.608  [source][tid2013]  | 0.8020 / 0.6088 | 0.802 / 0.609 [source][kadid10k]|
|   VIFp  | 0.6102 / 0.4579 | 0.610 / 0.457  [source][tid2013]  | 0.6500 / 0.4770 | 0.650 / 0.477 [source][kadid10k]|
|   FSIM  | 0.8015 / 0.6289 | 0.801 / 0.630  [source][tid2013]  | 0.8294 / 0.6390 | 0.829 / 0.639 [source][kadid10k]|
|   FSIMc | 0.8509 / 0.6665 | 0.851 / 0.667  [source][tid2013]  | 0.8537 / 0.6650 | 0.854 / 0.665 [source][kadid10k]|
| IW-PSNR |    -   /    -   | 0.6913 /   -   [source][eval2019] |    -   /    -   |     -   /    -   |
| IW-SSIM |    -   /    -   | 0.7779 / 0.5977 [source][eval2019]|    -   /    -   |     -   /    -   |
| SR-SIM  |    -   /    -   | 0.8076 / 0.6406 [source][eval2019]|    -   /    -   | 0.839 / 0.652 [source][kadid10k]|
| SR-SIMc |    -   /    -   |    -   /    -                     |    -   /    -   |     -   /    -   |
|   GMSD  | 0.8038 / 0.6334 | 0.8030 / 0.6352 [source](GMSD)    | 0.8474 / 0.6640 | 0.847 / 0.664 [source][kadid10k]|
|   VSI   | 0.8949 / 0.7159 | 0.8965 / 0.7183 [source][eval2019]| 0.8780 / 0.6899 | 0.861 / 0.678 [source][kadid10k]|
| Content | 0.7049 / 0.5173 |    -   /    -                     | 0.7237 / 0.5326 |     -   /    -   |
|  Style  | 0.5384 / 0.3720 |    -   /    -                     | 0.6470 / 0.4646 |     -   /    -   |
| HaarPSI | 0.8732 / 0.6923 | 0.8732 / 0.6923 [source](HaarPSI) | 0.8849 / 0.6995 | 0.885 / 0.699 [source][kadid10k]|
|   MDSI  | 0.8899 / 0.7123 | 0.8899 / 0.7123 [source](MDSI)    | 0.8853 / 0.7023 | 0.885 / 0.702 [source][kadid10k]|
| MS-GMSD | 0.8121 / 0.6455 |0.8139 / 0.6467 [source](MS-GMSD)  | 0.8523 / 0.6692 |     -   /    -   |
| MS-GMSDc| 0.8875 / 0.7105 | 0.687 / 0.496 [source](MS-GMSD)   | 0.8697 / 0.6831 |     -   /    -   |
|LPIPS-VGG| 0.6696 / 0.4970 | 0.670 / 0.497 [source][dists]     | 0.7201 / 0.5313 |     -   /    -   |
|  PieAPP | 0.8355 / 0.6495 | 0.875 / 0.710 [source][dists]     | 0.8655 / 0.6758 |     -   /    -   |
|  DISTS  | 0.7077 / 0.5212 | 0.830 / 0.639 [source][dists]     | 0.8137 / 0.6254 |     -   /    -   |


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


<!-- CITATION -->
### Citation

If you use PIQ in your project, please, cite it as follows.

```tex
@misc{piq,
  title={{PyTorch Image Quality}: Metrics and Measure for Image Quality Assessment},
  url={https://github.com/photosynthesis-team/piq},
  note={Open-source software available at https://github.com/photosynthesis-team/piq},
  author={
    Sergey Kastryulin and
    Djamil Zakirov and
    Denis Prokopenko},
  year={2019},
}
```


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

[codecov-shield]:https://codecov.io/gh/photosynthesis-team/piq/branch/master/graph/badge.svg
[codecov-url]:https://codecov.io/gh/photosynthesis-team/piq

[tid2013]:http://www.ponomarenko.info/tid2013.htm
[kadid10k]:http://database.mmsp-kn.de/kadid-10k-database.html
[eval2019]:https://ieeexplore.ieee.org/abstract/document/8847307
[dists]:https://arxiv.org/abs/2004.07728
