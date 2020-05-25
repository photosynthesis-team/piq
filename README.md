# PhotoSynthesis.Metrics
![CI flake-8 style check][ci-flake-8-style-check-shield]
![CI testing][ci-testing]  
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![PyPI version][pypi-version-shield]][pypi-version-url]


<!-- ABOUT THE PROJECT -->

PyTorch library with measures and metrics for various image-to-image tasks like denoising, super-resolution, 
image generation etc. This easy to use yet flexible and extensive library is developed with focus on reliability 
and reproducibility of results. Use your favourite measures as losses for training neural networks with ready-to-use 
PyTorch modules.  


<!-- GETTING STARTED -->
### Getting started  

```python
import torch
from photosynthesis_metrics import ssim

prediction = torch.rand(3, 3, 256, 256)
target = torch.rand(3, 3, 256, 256)
ssim_index = ssim(prediction, target, data_range=1.)
```


<!-- MINIMAL EXAMPLES -->
### Minimal examples

<!-- SSIM EXAMPLES -->
<details>
<summary>Structural Similarity (SSIM)</summary>
<p>

To compute SSIM index as a measure, use lower case function from the library:
```python
import torch
from photosynthesis_metrics import ssim
from typing import Union, Tuple

prediction = torch.rand(3, 3, 256, 256)
target = torch.rand(3, 3, 256, 256) 
ssim_index: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = ssim(prediction, target, data_range=1.)
```

In order to use SSIM as a loss function, use corresponding PyTorch module:
```python
import torch
from photosynthesis_metrics import SSIMLoss

loss = SSIMLoss()
prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
target = torch.rand(3, 3, 256, 256)
output: torch.Tensor = loss(prediction, target, data_range=1.)
output.backward()
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
from photosynthesis_metrics import multi_scale_ssim

prediction = torch.rand(3, 3, 256, 256)
target = torch.rand(3, 3, 256, 256) 
ms_ssim_index: torch.Tensor = multi_scale_ssim(prediction, target, data_range=1.)
```

In order to use MS-SSIM as a loss function, use corresponding PyTorch module:
```python
import torch
from photosynthesis_metrics import MultiScaleSSIMLoss

loss = MultiScaleSSIMLoss()
prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
target = torch.rand(3, 3, 256, 256)
output: torch.Tensor = loss(prediction, target, data_range=1.)
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
from photosynthesis_metrics import total_variation

data = torch.rand(3, 3, 256, 256) 
tv: torch.Tensor = total_variation(data)
```

In order to use TV as a loss function, use corresponding PyTorch module:
```python
import torch
from photosynthesis_metrics import TVLoss

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
from photosynthesis_metrics import vif_p

predicted = torch.rand(3, 3, 256, 256)
target = torch.rand(3, 3, 256, 256)
vif: torch.Tensor = vif_p(predicted, target, data_range=1.)
```

In order to use VIF as a loss function, use corresponding PyTorch class:
```python
import torch
from photosynthesis_metrics import VIFLoss

loss = VIFLoss(sigma_n_sq=2.0, data_range=1.)
prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
target = torch.rand(3, 3, 256, 256)
ouput: torch.Tensor = loss(prediction, target)
output.backward()
```

Note, that VIFLoss returns `1 - VIF` value.
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
from photosynthesis_metrics import MSID

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
from photosynthesis_metrics import MSID

first_dl, second_dl = DataLoader(), DataLoader()
msid_metric = MSID() 
first_feats = msid_metric._compute_feats(first_dl)
second_feats = msid_metric._compute_feats(second_dl)
msid: torch.Tensor = msid_metric(first_feats, second_feats)
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
from photosynthesis_metrics import FID

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
from photosynthesis_metrics import FID

first_dl, second_dl = DataLoader(), DataLoader()
fid_metric = FID() 
first_feats = fid_metric._compute_feats(first_dl)
second_feats = fid_metric._compute_feats(second_dl)
msid: torch.Tensor = fid_metric(first_feats, second_feats)
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
from photosynthesis_metrics import KID

kid_metric = KID()
prediction_feats = torch.rand(10000, 1024)
target_feats = torch.rand(10000, 1024)
msid: torch.Tensor = kid_metric(prediction_feats, target_feats)
```
 
If image features are not available, extract them using `_compute_feats` of `KID` class. 
Please note that `_compute_feats` consumes a data loader of predefined format. 
```python
import torch
from  torch.utils.data import DataLoader
from photosynthesis_metrics import KID

first_dl, second_dl = DataLoader(), DataLoader()
kid_metric = KID() 
first_feats = kid_metric._compute_feats(first_dl)
second_feats = kid_metric._compute_feats(second_dl)
msid: torch.Tensor = kid_metric(first_feats, second_feats)
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
from photosynthesis_metrics import inception_score

prediction_feats = torch.rand(10000, 1024)
mean: torch.Tensor, variance: torch.Tensor = inception_score(prediction_feats, num_splits=10)
```
 
To compute difference between IS for 2 sets of image features, use `IS` class.
```python
import torch
from photosynthesis_metrics import IS


is_metric = IS(distance='l1') 
prediction_feats = torch.rand(10000, 1024)
target_feats = torch.rand(10000, 1024)
distance: torch.Tensor = is_metric(prediction_feats, target_feats)
```  
</p>
</details>

<!-- TABLE OF CONTENTS -->
### Table of Contents

* [Overview](#overview)
    * [Installation](#installation)
    * [Roadmap](#roadmap)
* [Community](#community)
    * [Contributing](#contributing)
    * [Contact](#contact)
    * [Acknowledgements](#acknowledgements)


### Overview

*PhotoSynthesis.Metrics* helps you to concentrate on your experiments without the boilerplate code.
The library contains a set of measures and metrics that is constantly getting extended. 
For measures/metrics that can be used as loss functions, corresponding PyTorch modules are implemented.
 


#### Installation

`$ pip install photosynthesis-metrics`
 
If you want to use the latest features straight from the master, clone the repo:
```sh
$ git clone https://github.com/photosynthesis-team/photosynthesis.metrics.git
```

<!-- ROADMAP -->
#### Roadmap

See the [open issues](https://github.com/photosynthesis-team/photosynthesis.metrics/issues) for a list of proposed 
features and known issues.


<!-- COMMUNITY -->
### Community


<!-- CONTRIBUTING -->
#### Contributing

We appreciate all contributions. If you plan to: 
- contribute back bug-fixes, please do so without any further discussion
- close one of [open issues](https://github.com/photosynthesis-team/photosynthesis.metrics/issues), please do so if no one has been assigned to it
- contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us

Please see the [contribution guide](CONTRIBUTING.md) for more information.


<!-- CONTACT -->
#### Contact

**Sergey Kastryulin** - [@snk4tr](https://github.com/snk4tr) - `snk4tr@gmail.com`

Project Link: [https://github.com/photosynthesis-team/photosynthesis.metrics](https://github.com/photosynthesis-team/photosynthesis.metrics)  
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
[license-url]: https://github.com/photosynthesis-team/photosynthesis.metrics/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/sergey-kastryulin/
[ci-flake-8-style-check-shield]: https://github.com/photosynthesis-team/photosynthesis.metrics/workflows/flake-8%20style%20check/badge.svg
[ci-testing]: https://github.com/photosynthesis-team/photosynthesis.metrics/workflows/testing/badge.svg
[pypi-version-shield]: https://badge.fury.io/py/photosynthesis-metrics.svg
[pypi-version-url]: https://badge.fury.io/py/photosynthesis-metrics