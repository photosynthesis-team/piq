<div align="center">
 
![piq_logo](https://user-images.githubusercontent.com/15848838/95228385-ed106500-0807-11eb-8584-d3fdbdd47ac7.jpeg)

[![License][license-shield]][license-url]
[![PyPI version][pypi-version-shield]][pypi-version-url]
[![Conda version][conda-version-shield]][conda-version-url]  
![CI flake-8 style check][ci-flake-8-style-check-shield]
![CI testing][ci-testing]
[![codecov][codecov-shield]][codecov-url]  
[![Quality Gate Status][quality-gate-status-shield]][quality-gate-status-url]
[![Maintainability Rating][maintainability-raiting-shield]][maintainability-raiting-url]
[![Reliability Rating][reliability-rating-badge]][reliability-rating-url]
</div>

<!-- ABOUT THE PROJECT -->

Collection of measures and metrics for image quality assessment.
- Unified interface, which easy to use and extend.
- Written on pure PyTorch with bare minima of additional dependencies.
- Fast (GPU computations available) and reliable.
- Most metrics can be backpropagated for model optimization.
- Supports python 3.6-3.8


<!-- GETTING STARTED -->
### Getting started  

First group of metrics takes image or images as input, e.g PSNR, SSIM, BRISQUE. 
We have functional interface, which returns metric value and class interface, which allows to use any metric as a loss.

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

Second group takes a list of image features e.g. IS, FID, KID.
Image features can be extracted by some feature extractor network separetely, or by using `_compute_feats` method of a class. Note, that `_compute_feats` consumes a dataloader of predefined format.

```python
import torch
from  torch.utils.data import DataLoader
from piq import FID

first_dl, second_dl = DataLoader(), DataLoader()
fid_metric = FID() 
first_feats = fid_metric._compute_feats(first_dl)
second_feats = fid_metric._compute_feats(second_dl)
fid: torch.Tensor = fid_metric(first_feats, second_feats)
```

If you already have image features, use class interface for score computation:

```python
import torch
from piq import FID

prediction_feats = torch.rand(10000, 1024)
target_feats = torch.rand(10000, 1024)
msid_metric = MSID()
msid: torch.Tensor = msid_metric(prediction_feats, target_feats)
```

For a full list of examples, see [image metrics](examples/image_metrics.py) and [feature metrics](examples/feature_metrics.py) examles.

<!-- IMAGE METRICS -->
### Image metrics

 1. [Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE)](https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf)
 2. [Content score](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)
 3. [Deep Image Structure and Texture Similarity measure (DISTS)](https://arxiv.org/abs/2004.07728)
 4. [Feature Similarity Index Measure (FSIM)](https://www4.comp.polyu.edu.hk/~cslzhang/IQA/TIP_IQA_FSIM.pdf)
 5. [Gradient Magnitude Similarity Deviation (GMSD)](https://arxiv.org/pdf/1308.3052.pdf)
 6. [Haar Wavelet-Based Perceptual Similarity Index (HaarPSI)](http://www.math.uni-bremen.de/cda/HaarPSI/publications/HaarPSI_preprint_v4.pdf)
 7. [Learned Perceptual Image Patch Similarity measure (LPIPS)](https://arxiv.org/abs/1801.03924) 
 8. [Mean Deviation Similarity Index (MDSI)](https://ieeexplore.ieee.org/abstract/document/7556976/)
 9. [Multi-Scale Structural Similarity (MS-SSIM)](https://ieeexplore.ieee.org/document/1292216)
 10. [Multi-Scale Gradient Magnitude Similarity Deviation (MS-GMSD)](http://www.cse.ust.hk/~psander/docs/gradsim.pdf)
 11. [Peak Signal-to-Noise Ratio (PSNR)](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
 12. [Structural Similarity (SSIM)](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf)
 13. [Style score](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)
 14. [Total Variation (TV)](https://en.wikipedia.org/wiki/Total_variation)
 15. [Visual Information Fidelity (VIF)](https://live.ece.utexas.edu/research/Quality/VIF.htm)
 16. [Visual Saliency-induced Index (VSI)](https://ieeexplore.ieee.org/document/6873260) 

<!-- FEATURE METRICS -->
### Feature metrics
1. [Frechet Inception Distance(FID)](https://arxiv.org/abs/1706.08500)
2. [Geometry Score (GS)](https://arxiv.org/abs/1802.02664)
3. [Inception Score(IS)](https://arxiv.org/abs/1606.03498)
4. [Kernel Inception Distance(KID)](https://arxiv.org/abs/1801.01401)
5. [Multi-Scale Intrinsic Distance (MSID)](https://arxiv.org/abs/1905.11141) 

### Overview

*PyTorch Image Quality* (former [PhotoSynthesis.Metrics](https://pypi.org/project/photosynthesis-metrics/0.4.0/)) helps you to concentrate on your experiments without the boilerplate code.
The library contains a set of measures and metrics that is constantly getting extended. 
For measures/metrics that can be used as loss functions, corresponding PyTorch modules are implemented.
 

#### Installation

`$ pip install piq`

`$ conda install piq -c photosynthesis-team -c conda-forge -c pytorch`
 
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

We appreciate all contributions. If you plan to: 
- contribute back bug-fixes, please do so without any further discussion
- close one of [open issues](https://github.com/photosynthesis-team/piq/issues), please do so if no one has been assigned to it
- contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us

Please see the [contribution guide](CONTRIBUTING.md) for more information.


<!-- CONTACT -->
#### Contact

**Sergey Kastryulin** - [@snk4tr](https://github.com/snk4tr) - `snk4tr@gmail.com`

Project Link: [https://github.com/photosynthesis-team/piq](https://github.com/photosynthesis-team/piq)  
PhotoSynthesis Team: [https://github.com/photosynthesis-team](https://github.com/photosynthesis-team)

Other projects by PhotoSynthesis Team:  
* [PhotoSynthesis.Models](https://github.com/photosynthesis-team/photosynthesis.models)

<!-- ACKNOWLEDGEMENTS -->
#### Acknowledgements

* **Pavel Parunin** - [@PavelParunin](https://github.com/ParuninPavel) - idea proposal and development
* **Djamil Zakirov** - [@zakajd](https://github.com/zakajd) - development
* **Denis Prokopenko** - [@denproc](https://github.com/denproc) - development



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[license-url]: https://github.com/photosynthesis-team/piq/blob/master/LICENSE
[ci-flake-8-style-check-shield]: https://github.com/photosynthesis-team/piq/workflows/flake-8%20style%20check/badge.svg
[ci-testing]: https://github.com/photosynthesis-team/piq/workflows/testing/badge.svg
[pypi-version-shield]: https://badge.fury.io/py/piq.svg
[pypi-version-url]: https://badge.fury.io/py/piq
[conda-version-shield]: https://anaconda.org/photosynthesis-team/piq/badges/version.svg
[conda-version-url]: https://anaconda.org/photosynthesis-team/piq
[quality-gate-status-shield]: https://sonarcloud.io/api/project_badges/measure?project=photosynthesis-team_photosynthesis.metrics&metric=alert_status
[quality-gate-status-url]: https://sonarcloud.io/dashboard?id=photosynthesis-team_photosynthesis.metrics
[maintainability-raiting-shield]: https://sonarcloud.io/api/project_badges/measure?project=photosynthesis-team_photosynthesis.metrics&metric=sqale_rating
[maintainability-raiting-url]: https://sonarcloud.io/dashboard?id=photosynthesis-team_photosynthesis.metrics
[reliability-rating-badge]: https://sonarcloud.io/api/project_badges/measure?project=photosynthesis-team_photosynthesis.metrics&metric=reliability_rating
[reliability-rating-url]:https://sonarcloud.io/dashboard?id=photosynthesis-team_photosynthesis.metrics
[codecov-shield]:https://codecov.io/gh/photosynthesis-team/piq/branch/master/graph/badge.svg
[codecov-url]:https://codecov.io/gh/photosynthesis-team/piq