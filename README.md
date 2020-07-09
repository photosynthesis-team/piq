<div align="center">

# PyTorch Image Quality
[![License][license-shield]][license-url]
[![PyPI version][pypi-version-shield]][pypi-version-url]  
![CI flake-8 style check][ci-flake-8-style-check-shield]
![CI testing][ci-testing]
[![codecov][codecov-shield]][codecov-url]  
[![Quality Gate Status][quality-gate-status-shield]][quality-gate-status-url]
[![Maintainability Rating][maintainability-raiting-shield]][maintainability-raiting-url]
[![Reliability Rating][reliability-rating-badge]][reliability-rating-url]
</div>

<!-- ABOUT THE PROJECT -->

Collection of measures and metrics for automatic image quality assessment in various image-to-image tasks such as 
denoising, super-resolution, image generation etc. 
This easy to use yet flexible and extensive library is developed with focus on reliability and 
reproducibility of results.
Use your favourite measures as losses for training neural networks with ready-to-use PyTorch modules.  

<!-- GETTING STARTED -->
### Getting started  

```python
import torch
from piq import ssim

prediction = torch.rand(3, 3, 256, 256)
target = torch.rand(3, 3, 256, 256)
ssim_index = ssim(prediction, target, data_range=1.)
```


<!-- EXAMPLES -->
### Examples

<!-- BRISQUE EXAMPLES -->
<details>
<summary>Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE)</summary>
<p>

To compute [BRISQUE score](https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf) as a measure, use lower case function from the library:
```python
import torch
from piq import brisque
from typing import Union, Tuple

prediction = torch.rand(3, 3, 256, 256)
brisque_index: torch.Tensor = brisque(prediction, data_range=1.)
```

In order to use BRISQUE as a loss function, use corresponding PyTorch module:
```python
import torch
from piq import BRISQUELoss

loss = BRISQUELoss(data_range=1.)
prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
output: torch.Tensor = loss(prediction)
output.backward()
```
</p>
</details>

<!-- FSIM EXAMPLES -->
 <details>
 <summary>Feature Similarity Index Measure (FSIM)</summary>
 <p>

  To compute [FSIM](https://www4.comp.polyu.edu.hk/~cslzhang/IQA/TIP_IQA_FSIM.pdf) as a measure, use lower case function from the library:
 ```python
 import torch
 from piq import fsim

 prediction = torch.rand(3, 3, 256, 256)
 target = torch.rand(3, 3, 256, 256)
 vsi_index: torch.Tensor = fsim(prediction, target, data_range=1.)
 ```

  In order to use FSIM as a loss function, use corresponding PyTorch module:
 ```python
 import torch
 from piq import FSIMLoss

 loss = FSIMLoss(data_range=1.)
 prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
 target = torch.rand(3, 3, 256, 256)
 output: torch.Tensor = loss(prediction, target)
 output.backward()
 ```
 </p>
 </details>
 
 <!-- FID EXAMPLES -->
 <details>
 <summary>Frechet Inception Distance(FID)</summary>
 <p>
 
 Use `FID` class to compute [FID score](https://arxiv.org/abs/1706.08500) from image features, 
 pre-extracted from some feature extractor network:
 ```python
 import torch
 from piq import FID
 
 fid_metric = FID()
 prediction_feats = torch.rand(10000, 1024)
 target_feats = torch.rand(10000, 1024)
 msid: torch.Tensor = fid_metric(prediction_feats, target_feats)
 ```
  
 If image features are not available, extract them using `_compute_feats` of `FID` class. 
 Please note that `_compute_feats` consumes a data loader of predefined format.
 ```python
 import torch
 from  torch.utils.data import DataLoader
 from piq import FID
 
 first_dl, second_dl = DataLoader(), DataLoader()
 fid_metric = FID() 
 first_feats = fid_metric._compute_feats(first_dl)
 second_feats = fid_metric._compute_feats(second_dl)
 msid: torch.Tensor = fid_metric(first_feats, second_feats)
 ```  
 </p>
 </details>
 
 <!-- GS EXAMPLES -->
 <details>
 <summary>Geometry Score (GS)</summary>
 <p>
 
 Use `GS` class to compute [Geometry Score](https://arxiv.org/abs/1802.02664) from image features, 
 pre-extracted from some feature extractor network. Computation is heavily CPU dependent, adjust `num_workers` parameter according to your system configuration:
 ```python
 import torch
 from piq import GS
 
 gs_metric = GS(sample_size=64, num_iters=100, i_max=100, num_workers=4)
 prediction_feats = torch.rand(10000, 1024)
 target_feats = torch.rand(10000, 1024)
 gs: torch.Tensor = gs_metric(prediction_feats, target_feats)
 ```
 
 GS metric requiers `gudhi` library which is not installed by default. 
 If you use conda, write: `conda install -c conda-forge gudhi`, otherwise follow [installation guide](http://gudhi.gforge.inria.fr/python/latest/installation.html).
 </p>
 </details>
 
 <!-- GMSD EXAMPLES -->
 <details>
 <summary>Gradient Magnitude Similarity Deviation (GMSD)</summary>
 <p>
 
 This is port of MATLAB version from the authors of original paper.
 It can be used both as a measure and as a loss function. In any case it should me minimized.
 Usually values of GMSD lie in [0, 0.35] interval.
 ```python
 import torch
 from piq import GMSDLoss
 
 loss = GMSDLoss(data_range=1.)
 prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
 target = torch.rand(3, 3, 256, 256)
 output: torch.Tensor = loss(prediction, target)
 output.backward()
 ```
 </p>
 </details>
 
 <!-- IS EXAMPLES -->
 <details>
 <summary>Inception Score(IS)</summary>
 <p>
 
 Use `inception_score` function to compute [IS](https://arxiv.org/abs/1606.03498) from image features, 
 pre-extracted from some feature extractor network. Note, that we follow recomendations from paper [A Note on the Inception Score](https://arxiv.org/pdf/1801.01973.pdf), which proposed small modification to original algorithm:
 ```python
 import torch
 from piq import inception_score
 
 prediction_feats = torch.rand(10000, 1024)
 mean, variance = inception_score(prediction_feats, num_splits=10)
 ```
  
 To compute difference between IS for 2 sets of image features, use `IS` class.
 ```python
 import torch
 from piq import IS
 
 
 is_metric = IS(distance='l1') 
 prediction_feats = torch.rand(10000, 1024)
 target_feats = torch.rand(10000, 1024)
 distance: torch.Tensor = is_metric(prediction_feats, target_feats)
 ```  
 </p>
 </details>
 
 <!-- KID EXAMPLES -->
 <details>
 <summary>Kernel Inception Distance(KID)</summary>
 <p>
 
 Use `KID` class to compute [KID score](https://arxiv.org/abs/1801.01401) from image features, 
 pre-extracted from some feature extractor network:
 ```python
 import torch
 from piq import KID
 
 kid_metric = KID()
 prediction_feats = torch.rand(10000, 1024)
 target_feats = torch.rand(10000, 1024)
 kid: torch.Tensor = kid_metric(prediction_feats, target_feats)
 ```
  
 If image features are not available, extract them using `_compute_feats` of `KID` class. 
 Please note that `_compute_feats` consumes a data loader of predefined format. 
 ```python
 import torch
 from  torch.utils.data import DataLoader
 from piq import KID
 
 first_dl, second_dl = DataLoader(), DataLoader()
 kid_metric = KID() 
 first_feats = kid_metric._compute_feats(first_dl)
 second_feats = kid_metric._compute_feats(second_dl)
 kid: torch.Tensor = kid_metric(first_feats, second_feats)
 ```  
 </p>
 </details>
 
 <!-- MSID EXAMPLES -->
 <details>
 <summary>Multi-Scale Intrinsic Distance (MSID)</summary>
 <p>
 
 Use `MSID` class to compute [MSID score](https://arxiv.org/abs/1905.11141) from image features, 
 pre-extracted from some feature extractor network: 
 ```python
 import torch
 from piq import MSID
 
 msid_metric = MSID()
 prediction_feats = torch.rand(10000, 1024)
 target_feats = torch.rand(10000, 1024)
 msid: torch.Tensor = msid_metric(prediction_feats, target_feats)
 ```
 
 If image features are not available, extract them using `_compute_feats` of `MSID` class. 
 Please note that `_compute_feats` consumes a data loader of predefined format.
 ```python
 import torch
 from  torch.utils.data import DataLoader
 from piq import MSID
 
 first_dl, second_dl = DataLoader(), DataLoader()
 msid_metric = MSID() 
 first_feats = msid_metric._compute_feats(first_dl)
 second_feats = msid_metric._compute_feats(second_dl)
 msid: torch.Tensor = msid_metric(first_feats, second_feats)
 ```  
 </p>
 </details>
 
 <!-- MS-SSIM EXAMPLES -->
 <details>
 <summary>Multi-Scale Structural Similarity (MS-SSIM)</summary>
 <p>
 
 To compute MS-SSIM index as a measure, use lower case function from the library:
 ```python
 import torch
 from piq import multi_scale_ssim
 
 prediction = torch.rand(3, 3, 256, 256)
 target = torch.rand(3, 3, 256, 256) 
 ms_ssim_index: torch.Tensor = multi_scale_ssim(prediction, target, data_range=1.)
 ```
 
 In order to use MS-SSIM as a loss function, use corresponding PyTorch module:
 ```python
 import torch
 from piq import MultiScaleSSIMLoss
 
 loss = MultiScaleSSIMLoss(data_range=1.)
 prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
 target = torch.rand(3, 3, 256, 256)
 output: torch.Tensor = loss(prediction, target)
 output.backward()
 ```
 </p>
 </details>
 
 <!-- MultiScale GMSD EXAMPLES -->
 <details>
 <summary>MultiScale GMSD (MS-GMSD)</summary>
 <p>
 
 It can be used both as a measure and as a loss function. In any case it should me minimized.
 By defualt scale weights are initialized with values from the paper. You can change them by passing a list of 4 variables to `scale_weights` argument during initialization. Both GMSD and MS-GMSD computed for greyscale images, but to take contrast changes into account authors propoced to also add chromatic component. Use flag `chromatic` to use MS-GMSDc version of the loss
 ```python
 import torch
 from piq import MultiScaleGMSDLoss
 
 loss = MultiScaleGMSDLoss(chromatic=True, data_range=1.)
 prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
 target = torch.rand(3, 3, 256, 256)
 output: torch.Tensor = loss(prediction, target)
 output.backward()
 ```
 </p>
 </details>

<!-- PSNR EXAMPLES -->
<details>
<summary>Peak Signal-to-Noise Ratio (PSNR)</summary>
<p>

To compute PSNR as a measure, use lower case function from the library.
By default it computes average of PSNR if more than 1 image is included in batch.
You can specify other reduction methods by `reduction` flag.

```python
import torch
from piq import psnr
from typing import Union, Tuple

prediction = torch.rand(3, 3, 256, 256)
target = torch.rand(3, 3, 256, 256) 
psnr_mean = psnr(prediction, target, data_range=1., reduction='mean')
psnr_per_image = psnr(prediction, target, data_range=1., reduction='none')
```

Note: Colour images are first converted to YCbCr format and only luminance component is considered.
</p>
</details>

<!-- SSIM EXAMPLES -->
<details>
<summary>Structural Similarity (SSIM)</summary>
<p>

To compute SSIM index as a measure, use lower case function from the library:
```python
import torch
from piq import ssim
from typing import Union, Tuple

prediction = torch.rand(3, 3, 256, 256)
target = torch.rand(3, 3, 256, 256) 
ssim_index: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = ssim(prediction, target, data_range=1.)
```

In order to use SSIM as a loss function, use corresponding PyTorch module:
```python
import torch
from piq import SSIMLoss

loss = SSIMLoss(data_range=1.)
prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
target = torch.rand(3, 3, 256, 256)
output: torch.Tensor = loss(prediction, target)
output.backward()
```
</p>
</details>

<!-- TV EXAMPLES -->
<details>
<summary>Total Variation (TV)</summary>
<p>

To compute TV as a measure, use lower case function from the library:
```python
import torch
from piq import total_variation

data = torch.rand(3, 3, 256, 256) 
tv: torch.Tensor = total_variation(data)
```

In order to use TV as a loss function, use corresponding PyTorch module:
```python
import torch
from piq import TVLoss

loss = TVLoss()
prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
output: torch.Tensor = loss(prediction)
output.backward()
```
</p>
</details>

<!-- VIF EXAMPLES -->
<details>
<summary>Visual Information Fidelity (VIF)</summary>
<p>

To compute VIF as a measure, use lower case function from the library:
```python
import torch
from piq import vif_p

predicted = torch.rand(3, 3, 256, 256)
target = torch.rand(3, 3, 256, 256)
vif: torch.Tensor = vif_p(predicted, target, data_range=1.)
```

In order to use VIF as a loss function, use corresponding PyTorch class:
```python
import torch
from piq import VIFLoss

loss = VIFLoss(sigma_n_sq=2.0, data_range=1.)
prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
target = torch.rand(3, 3, 256, 256)
output: torch.Tensor = loss(prediction, target)
output.backward()
```

Note, that VIFLoss returns `1 - VIF` value.
</p>
</details>

<!-- VSI EXAMPLES -->
<details>
<summary>Visual Saliency-induced Index (VSI)</summary>
<p>

To compute [VSI score](https://ieeexplore.ieee.org/document/6873260) as a measure, use lower case function from the library:
```python
import torch
from piq import vsi

prediction = torch.rand(3, 3, 256, 256)
target = torch.rand(3, 3, 256, 256)
vsi_index: torch.Tensor = vsi(prediction, target, data_range=1.)
```

In order to use VSI as a loss function, use corresponding PyTorch module:
```python
import torch
from piq import VSILoss

loss = VSILoss(data_range=1.)
prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
target = torch.rand(3, 3, 256, 256)
output: torch.Tensor = loss(prediction, target)
output.backward()
```
</p>
</details>


### Overview

*PyTorch Image Quality* (former [PhotoSynthesis.Metrics](https://pypi.org/project/photosynthesis-metrics/0.4.0/)) helps you to concentrate on your experiments without the boilerplate code.
The library contains a set of measures and metrics that is constantly getting extended. 
For measures/metrics that can be used as loss functions, corresponding PyTorch modules are implemented.
 


#### Installation

`$ pip install piq`
 
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
[quality-gate-status-shield]: https://sonarcloud.io/api/project_badges/measure?project=photosynthesis-team_photosynthesis.metrics&metric=alert_status
[quality-gate-status-url]: https://sonarcloud.io/dashboard?id=photosynthesis-team_photosynthesis.metrics
[maintainability-raiting-shield]: https://sonarcloud.io/api/project_badges/measure?project=photosynthesis-team_photosynthesis.metrics&metric=sqale_rating
[maintainability-raiting-url]: https://sonarcloud.io/dashboard?id=photosynthesis-team_photosynthesis.metrics
[reliability-rating-badge]: https://sonarcloud.io/api/project_badges/measure?project=photosynthesis-team_photosynthesis.metrics&metric=reliability_rating
[reliability-rating-url]:https://sonarcloud.io/dashboard?id=photosynthesis-team_photosynthesis.metrics
[codecov-shield]:https://codecov.io/gh/photosynthesis-team/piq/branch/master/graph/badge.svg
[codecov-url]:https://codecov.io/gh/photosynthesis-team/piq