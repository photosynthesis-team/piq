
.. image:: https://raw.githubusercontent.com/photosynthesis-team/piq/master/docs/source/_static/piq_logo_main.png
    :target: https://github.com/photosynthesis-team/piq

..

  PyTorch Image Quality (PIQ) is not endorsed by Facebook, Inc.;

  PyTorch, the PyTorch logo and any related marks are trademarks of Facebook, Inc.

|pypy| |conda| |flake8| |tests| |codecov| |quality_gate|

.. |pypy| image:: https://badge.fury.io/py/piq.svg
   :target: https://pypi.org/project/piq/
   :alt: Pypi Version
.. |conda| image:: https://anaconda.org/photosynthesis-team/piq/badges/version.svg
   :target: https://anaconda.org/photosynthesis-team/piq
   :alt: Conda Version
.. |flake8| image:: https://github.com/photosynthesis-team/piq/workflows/flake-8%20style%20check/badge.svg
   :alt: CI flake-8 style check
.. |tests| image:: https://github.com/photosynthesis-team/piq/workflows/testing/badge.svg
   :alt: CI testing
.. |codecov| image:: https://codecov.io/gh/photosynthesis-team/piq/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/photosynthesis-team/piq
   :alt: codecov
.. |quality_gate| image:: https://sonarcloud.io/api/project_badges/measure?project=photosynthesis-team_photosynthesis.metrics&metric=alert_status
   :target: https://sonarcloud.io/dashboard?id=photosynthesis-team_photosynthesis.metrics
   :alt: Quality Gate Status



.. intro-section-start

`PyTorch Image Quality (PIQ) <https://github.com/photosynthesis-team/piq>`_ is a collection of measures and metrics for
image quality assessment. PIQ helps you to concentrate on your experiments without the boilerplate code.
The library contains a set of measures and metrics that is continually getting extended.
For measures/metrics that can be used as loss functions, corresponding PyTorch modules are implemented.

We provide:

* Unified interface, which is easy to use and extend.
* Written on pure PyTorch with bare minima of additional dependencies.
* Extensive user input validation. Your code will not crash in the middle of the training.
* Fast (GPU computations available) and reliable.
* Most metrics can be backpropagated for model optimization.
* Supports python 3.7-3.10.

PIQ was initially named `PhotoSynthesis.Metrics <https://pypi.org/project/photosynthesis-metrics/0.4.0/>`_.

.. intro-section-end

.. installation-section-start

Installation
------------
`PyTorch Image Quality (PIQ) <https://github.com/photosynthesis-team/piq>`_ can be installed using ``pip``, ``conda`` or ``git``.


If you use ``pip``, you can install it with:

.. code-block:: sh

    $ pip install piq


If you use ``conda``, you can install it with:

.. code-block:: sh

    $ conda install piq -c photosynthesis-team -c conda-forge -c PyTorch


If you want to use the latest features straight from the master, clone `PIQ repo <https://github.com/photosynthesis-team/piq>`_:

.. code-block:: sh

   git clone https://github.com/photosynthesis-team/piq.git
   cd piq
   python setup.py install

.. installation-section-end

.. documentation-section-start

Documentation
-------------

The full documentation is available at https://piq.readthedocs.io.

.. documentation-section-end

.. usage-examples-start

Usage Examples
---------------

Image-Based metrics
^^^^^^^^^^^^^^^^^^^
The group of metrics (such as PSNR, SSIM, BRISQUE) takes an image or a pair of images as input to compute a distance between them.
We have a functional interface, which returns a metric value, and a class interface, which allows to use any metric
as a loss function.

.. code-block:: python

   import torch
   from piq import ssim, SSIMLoss

   x = torch.rand(4, 3, 256, 256, requires_grad=True)
   y = torch.rand(4, 3, 256, 256)

   ssim_index: torch.Tensor = ssim(x, y, data_range=1.)

   loss = SSIMLoss(data_range=1.)
   output: torch.Tensor = loss(x, y)
   output.backward()

For a full list of examples, see `image metrics <https://github.com/photosynthesis-team/piq/blob/master/examples/image_metrics.py>`_ examples.

Distribution-Based metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

The group of metrics (such as IS, FID, KID) takes a list of image features to compute the distance between distributions.
Image features can be extracted by some feature extractor network separately or by using the ``compute_feats`` method of a
class.

Note:
    ``compute_feats`` consumes a data loader of a predefined format.

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from piq import FID

   first_dl, second_dl = DataLoader(), DataLoader()
   fid_metric = FID()
   first_feats = fid_metric.compute_feats(first_dl)
   second_feats = fid_metric.compute_feats(second_dl)
   fid: torch.Tensor = fid_metric(first_feats, second_feats)


If you already have image features, use the class interface for score computation:

.. code-block:: python

    import torch
    from piq import FID

    x_feats = torch.rand(10000, 1024)
    y_feats = torch.rand(10000, 1024)
    msid_metric = MSID()
    msid: torch.Tensor = msid_metric(x_feats, y_feats)


For a full list of examples, see `feature metrics <https://github.com/photosynthesis-team/piq/blob/master/examples/feature_metrics.py>`_ examples.

.. usage-examples-end

.. list-of-metrics-start

List of metrics
---------------

Full-Reference (FR)
^^^^^^^^^^^^^^^^^^^

===========  ======  ==========
Acronym      Year    Metric
===========  ======  ==========
PSNR         \-      `Peak Signal-to-Noise Ratio <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_
SSIM         2003    `Structural Similarity <https://en.wikipedia.org/wiki/Structural_similarity>`_
MS-SSIM      2004    `Multi-Scale Structural Similarity <https://ieeexplore.ieee.org/abstract/document/1292216>`_
IW-SSIM      2011    `Information Content Weighted Structural Similarity Index <https://ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf>`_
VIFp         2004    `Visual Information Fidelity <https://ieeexplore.ieee.org/document/1576816>`_
FSIM         2011    `Feature Similarity Index Measure <https://ieeexplore.ieee.org/document/5705575>`_
SR-SIM       2012    `Spectral Residual Based Similarity <https://sse.tongji.edu.cn/linzhang/ICIP12/ICIP-SR-SIM.pdf>`_
GMSD         2013    `Gradient Magnitude Similarity Deviation <https://arxiv.org/abs/1308.3052>`_
MS-GMSD      2017    `Multi-Scale Gradient Magnitude Similarity Deviation <https://ieeexplore.ieee.org/document/7952357>`_
VSI          2014    `Visual Saliency-induced Index <https://ieeexplore.ieee.org/document/6873260>`_
DSS          2015    `DCT Subband Similarity Index <https://ieeexplore.ieee.org/document/7351172>`_
\-           2016    `Content Score <https://arxiv.org/abs/1508.06576>`_
\-           2016    `Style Score <https://arxiv.org/abs/1508.06576>`_
HaarPSI      2016    `Haar Perceptual Similarity Index <https://arxiv.org/abs/1607.06140>`_
MDSI         2016    `Mean Deviation Similarity Index <https://arxiv.org/abs/1608.07433>`_
LPIPS        2018    `Learned Perceptual Image Patch Similarity <https://arxiv.org/abs/1801.03924>`_
PieAPP       2018    `Perceptual Image-Error Assessment through Pairwise Preference <https://arxiv.org/abs/1806.02067>`_
DISTS        2020    `Deep Image Structure and Texture Similarity <https://arxiv.org/abs/2004.07728>`_
===========  ======  ==========

No-Reference (NR)
^^^^^^^^^^^^^^^^^

===========  ======  ==========
Acronym      Year    Metric
===========  ======  ==========
TV           1937    `Total Variation <https://en.wikipedia.org/wiki/Total_variation>`_
BRISQUE      2012    `Blind/Referenceless Image Spatial Quality Evaluator <https://ieeexplore.ieee.org/document/6272356>`_
CLIP-IQA     2022    `CLIP-IQA <https://arxiv.org/pdf/2207.12396.pdf>`_
===========  ======  ==========

Distribution-Based (DB)
^^^^^^^^^^^^^^^^^^^^^^^

===========  ======  ==========
Acronym      Year    Metric
===========  ======  ==========
IS           2016    `Inception Score <https://arxiv.org/abs/1606.03498>`_
FID          2017    `Frechet Inception Distance <https://arxiv.org/abs/1706.08500>`_
GS           2018    `Geometry Score <https://arxiv.org/abs/1802.02664>`_
KID          2018    `Kernel Inception Distance <https://arxiv.org/abs/1801.01401>`_
MSID         2019    `Multi-Scale Intrinsic Distance <https://arxiv.org/abs/1905.11141>`_
PR           2019    `Improved Precision and Recall <https://arxiv.org/abs/1904.06991>`_
===========  ======  ==========

.. list-of-metrics-end

.. benchmark-section-start

Benchmark
---------

As part of our library we provide `code to benchmark <tests/results_benchmark.py>`_ all metrics on a set of common Mean Opinon Scores databases.
Currently we support several Full-Reference (`TID2013`_,  `KADID10k`_ and `PIPAL`_) and No-Reference (`KonIQ10k`_ and `LIVE-itW`_) datasets.
You need to download them separately and provide path to images as an argument to the script.

Here is an example how to evaluate SSIM and MS-SSIM metrics on TID2013 dataset:

.. code-block:: bash

   python3 tests/results_benchmark.py --dataset tid2013 --metrics SSIM MS-SSIM --path ~/datasets/tid2013 --batch_size 16

Below we provide a comparison between `Spearman's Rank Correlation Coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_ (SRCC) values obtained with PIQ and reported in surveys.
Closer SRCC values indicate the higher degree of agreement between results of computations on given datasets.
We do not report `Kendall rank correlation coefficient <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_ (KRCC)
as it is highly correlated with SRCC and provides limited additional information.
We do not report `Pearson linear correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_ (PLCC)
as it's highly dependent on fitting method and is biased towards simple examples.

For metrics that can take greyscale or colour images, ``c`` means chromatic version.

Full-Reference (FR) Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
===========  ===========================  ===========================  ===========================
     \                  TID2013                    KADID10k                       PIPAL
-----------  ---------------------------  ---------------------------  ---------------------------
  Source            PIQ / Reference            PIQ / Reference                PIQ / Reference
===========  ===========================  ===========================  ===========================
PSNR         0.69 / 0.69 `TID2013`_       0.68 / -                     0.41 / 0.41 `PIPAL`_
SSIM         0.72 / 0.64 `TID2013`_       0.72 / 0.72 `KADID10k`_      0.50 / 0.53 `PIPAL`_
MS-SSIM      0.80 / 0.79 `TID2013`_       0.80 / 0.80 `KADID10k`_      0.55 / 0.46 `PIPAL`_
IW-SSIM      0.78 / 0.78 `Eval2019`_      0.85 / 0.85 `KADID10k`_      0.60 / -
VIFp         0.61 / 0.61 `TID2013`_       0.65 / 0.65 `KADID10k`_      0.50 / -
FSIM         0.80 / 0.80 `TID2013`_       0.83 / 0.83 `KADID10k`_      0.59 / 0.60 `PIPAL`_
FSIMc        0.85 / 0.85 `TID2013`_       0.85 / 0.85 `KADID10k`_      0.59 / -
SR-SIM       0.81 / 0.81 `Eval2019`_      0.84 / 0.84 `KADID10k`_      0.57 / -
SR-SIMc      0.87 / -                     0.87 / -                     0.57 / -
GMSD         0.80 / 0.80 `MS-GMSD`_       0.85 / 0.85 `KADID10k`_      0.58 / -
VSI          0.90 / 0.90 `Eval2019`_      0.88 / 0.86 `KADID10k`_      0.54 / -
DSS          0.79 / 0.79 `Eval2019`_      0.86 / 0.86 `KADID10k`_      0.63 / -
Content      0.71 / -                     0.72 / -                     0.45 / -
Style        0.54 / -                     0.65 / -                     0.34 / -
HaarPSI      0.87 / 0.87 `HaarPSI`_       0.89 / 0.89 `KADID10k`_      0.59 / -
MDSI         0.89 / 0.89 `MDSI`_          0.89 / 0.89 `KADID10k`_      0.59 / -
MS-GMSD      0.81 / 0.81 `MS-GMSD`_       0.85 / -                     0.59 / -
MS-GMSDc     0.89 / 0.89 `MS-GMSD`_       0.87 / -                     0.59 / -
LPIPS-VGG    0.67 / 0.67 `DISTS`_         0.72 / -                     0.57 / 0.58 `PIPAL`_
PieAPP       0.84 / 0.88 `DISTS`_         0.87 / -                     0.70 / 0.71 `PIPAL`_
DISTS        0.81 / 0.83 `DISTS`_         0.88 / -                     0.62 / 0.66 `PIPAL`_
BRISQUE      0.37 / 0.84 `Eval2019`_      0.33 / 0.53 `KADID10k`_      0.21 / -
CLIP-IQA     0.50 / -                     0.48 / -                     0.26 / -
IS           0.26 / -                     0.25 / -                     0.09 / -
FID          0.67 / -                     0.66 / -                     0.18 / -
KID          0.42 / -                     0.66 / -                     0.12 / -
MSID         0.21 / -                     0.32 / -                     0.01 / -
GS           0.37 / -                     0.37 / -                     0.02 / -
===========  ===========================  ===========================  ===========================

No-Reference (NR) Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^
===========  ===========================  ===========================
     \                  KonIQ10k                    LIVE-itW
-----------  ---------------------------  ---------------------------
  Source            PIQ / Reference            PIQ / Reference
===========  ===========================  ===========================
BRISQUE      0.22 / -                     0.31 / -
CLIP-IQA     0.68 / 0.68 `CLIP-IQA off`_  0.64 / 0.64 `CLIP-IQA off`_
===========  ===========================  ===========================

.. _TID2013: http://www.ponomarenko.info/tid2013.htm
.. _KADID10k: http://database.mmsp-kn.de/kadid-10k-database.html
.. _Eval2019: https://ieeexplore.ieee.org/abstract/document/8847307/
.. _`MDSI`: https://arxiv.org/abs/1608.07433
.. _MS-GMSD: https://ieeexplore.ieee.org/document/7952357
.. _DISTS: https://arxiv.org/abs/2004.07728
.. _HaarPSI: https://arxiv.org/abs/1607.06140
.. _PIPAL: https://arxiv.org/pdf/2011.15002.pdf
.. _IW-SSIM: https://ieeexplore.ieee.org/document/7442122
.. _KonIQ10k: http://database.mmsp-kn.de/koniq-10k-database.html
.. _LIVE-itW: https://live.ece.utexas.edu/research/ChallengeDB/index.html
.. _CLIP-IQA off: https://github.com/IceClear/CLIP-IQA

Unlike FR and NR IQMs, designed to compute an image-wise distance, the DB metrics compare distributions of *sets* of images.
To address these problems, we adopt a different way of computing the DB IQMs proposed in `<https://arxiv.org/abs/2203.07809>`_.
Instead of extracting features from the whole images, we crop them into overlapping tiles of size ``96 × 96`` with ``stride = 32``.
This pre-processing allows us to treat each pair of images as a pair of distributions of tiles, enabling further comparison.
The other stages of computing the DB IQMs are kept intact.

.. benchmark-section-end

.. assertions-section-start

Assertions
----------
In PIQ we use assertions to raise meaningful messages when some component doesn't receive an input of the expected type.
This makes prototyping and debugging easier, but it might hurt the performance.
To disable all checks, use the Python ``-O`` flag: ``python -O your_script.py``

.. assertions-section-end


Roadmap
-------

See the `open issues <https://github.com/photosynthesis-team/piq/issues>`_ for a list of proposed
features and known issues.

Contributing
------------

If you would like to help develop this library, you'll find more information in our `contribution guide <CONTRIBUTING.rst>`_.

.. citation-section-start

Citation
--------
If you use PIQ in your project, please, cite it as follows.

.. code-block:: tex

   @misc{kastryulin2022piq,
     title = {PyTorch Image Quality: Metrics for Image Quality Assessment},
     url = {https://arxiv.org/abs/2208.14818},
     author = {Kastryulin, Sergey and Zakirov, Jamil and Prokopenko, Denis and Dylov, Dmitry V.},
     doi = {10.48550/ARXIV.2208.14818},
     publisher = {arXiv},
     year = {2022}
   }

.. code-block:: tex

   @misc{piq,
     title={{PyTorch Image Quality}: Metrics and Measure for Image Quality Assessment},
     url={https://github.com/photosynthesis-team/piq},
     note={Open-source software available at https://github.com/photosynthesis-team/piq},
     author={Sergey Kastryulin and Dzhamil Zakirov and Denis Prokopenko},
     year={2019}
   }

.. citation-section-end

.. contacts-section-start

Contacts
--------

**Sergey Kastryulin** - `@snk4tr <https://github.com/snk4tr>`_ - ``snk4tr@gmail.com``

**Jamil Zakirov** - `@zakajd <https://github.com/zakajd>`_ - ``djamilzak@gmail.com``

**Denis Prokopenko** - `@denproc <https://github.com/denproc>`_ - ``d.prokopenko@outlook.com``

.. contacts-section-end
