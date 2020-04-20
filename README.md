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

<details>
<summary>SSIM</summary>
<p>

To compute SSIM index as measure, use lower case function from the library:
```python
import torch
from photosynthesis_metrics import ssim

prediction = torch.rand(3, 3, 256, 256)
target = torch.rand(3, 3, 256, 256) 
ssim_index = ssim(prediction, target, data_range=1.)
```

In order to use SSIM as a loss function, use corresponding PyTorch module:
```python
import torch
from photosynthesis_metrics import SSIMLoss

loss = SSIMLoss()
prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
target = torch.rand(3, 3, 256, 256)
output = loss(prediction, target, data_range=1.)
output.backward()
```
</p>
</details>

<details>
<summary>MS-SSIM</summary>
<p>

To compute MS-SSIM index as measure, use lower case function from the library:
```python
import torch
from photosynthesis_metrics import multi_scale_ssim

prediction = torch.rand(3, 3, 256, 256)
target = torch.rand(3, 3, 256, 256) 
ssim_index = multi_scale_ssim(prediction, target, data_range=1.)
```

In order to use MS-SSIM as a loss function, use corresponding PyTorch module:
```python
import torch
from photosynthesis_metrics import MultiScaleSSIMLoss

loss = MultiScaleSSIMLoss()
prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
target = torch.rand(3, 3, 256, 256)
output = loss(prediction, target, data_range=1.)
output.backward()
```

</p>
</details>


<!-- TABLE OF CONTENTS -->
### Table of Contents

* [Overview](###overview)
    * [Installation](####installation)
    * [Roadmap](####roadmap)
* [Community](###community)
    * [Contributing](####contributing)
    * [Contact](####contact)
    * [Acknowledgements](####acknowledgements)


### Overview

**PhotoSynthesis.Metrics** helps you to concentrate on your experiments without the boilerplate code.
The library contains a set of measures and metrics that is constantly getting extended. 
For measures/metrics that can be used as loss functions, corresponding PyTorch modules are implemented.
Please refer to the [Minimal examples](##Minimal examples) for more usage details. 
 


#### Installation

`$ pip install photosynthesis-metrics`
 
If you want to use the latest features straight from the master, clone the repo:
```sh
$ git clone https://github.com/photosynthesis-team/photosynthesis.metrics.git
```

<!-- ROADMAP -->
#### Roadmap

See the [open issues](https://github.com/photosynthesis-team/photosynthesis.metrics/issues) for a list of proposed 
features (and known issues).


<!-- COMMUNITY -->
###Community


<!-- CONTRIBUTING -->
#### Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please follow [Google Python style guide](http://google.github.io/styleguide/pyguide.html) as a guidance on your code style 
decisions. The code will be checked with [flake-8 linter](http://flake8.pycqa.org/en/latest/) during the CI pipeline. 
Use [commitizen](https://github.com/commitizen/cz-cli) commit style where possible for simplification of understanding of 
performed changes.    


<!-- CONTACT -->
#### Contact

**Sergey Kastryulin** - [@snk4tr](https://twitter.com/snk4tr) - `snk4tr@gmail.com`

Project Link: [https://github.com/photosynthesis-team/photosynthesis.metrics](https://github.com/photosynthesis-team/photosynthesis.metrics)  
PhotoSynthesis Team: [https://github.com/photosynthesis-team](https://github.com/photosynthesis-team)

Other projects by PhotoSynthesis Team:  
PhotoSynthesis.Models: [https://github.com/photosynthesis-team/photosynthesis.models](https://github.com/photosynthesis-team/photosynthesis.models)

<!-- ACKNOWLEDGEMENTS -->
#### Acknowledgements

* **Pavel Parunin** - [@PavelParunin](https://github.com/ParuninPavel) - idea proposal and development
* **Djamil Zakirov** - [@zakajd](https://github.com/zakajd) - development



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