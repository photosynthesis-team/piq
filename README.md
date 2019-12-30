# PhotoSynthesis.Metrics
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

The project is intended to become a easy to use yet extensive library with metrics for 
various image-to-image tasks like denoising, super-resolution, image generation etc.


### Prerequisites

* [PyTorch 1.3+](https://pytorch.org) :white_check_mark:
* [segmentation_models.pytorch 0.1.0+](https://github.com/qubvel/segmentation_models.pytorch) :white_check_mark:  

### Installation
 
Clone the repo:
```sh
$ git clone https://github.com/photosynthesis-team/photosynthesis.metrics.git
```

Wheel and pip installations will be added later.

<!-- USAGE EXAMPLES -->
## Usage

```python
from photosynthesis_metrics.fid import compute_fid

fid = compute_fid(gt_stack, denoised_stack)
```
<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/photosynthesis-team/photosynthesis.metrics/issues) for a list of proposed 
features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

**Sergey Kastryulin** - [@snk4tr](https://twitter.com/snk4tr) - `snk4tr@gmail.com`

Project Link: [https://github.com/photosynthesis-team/photosynthesis.metrics](https://github.com/photosynthesis-team/photosynthesis.metrics)  
PhotoSynthesis Team: [https://github.com/photosynthesis-team](https://github.com/photosynthesis-team)

Other projects by PhotoSynthesis Team:  
PhotoSynthesis.Models: [https://github.com/photosynthesis-team/photosynthesis.models](https://github.com/photosynthesis-team/photosynthesis.models)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* **Pavel Parunin** - [@PavelParunin](https://github.com/ParuninPavel) - idea proposal and development



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/photosynthesis-team/photosynthesis.metrics/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/sergey-kastryulin/