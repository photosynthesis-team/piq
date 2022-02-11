
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
* Extensive user input validation. You code will not crash in the middle of the training.
* Fast (GPU computations available) and reliable.
* Most metrics can be backpropagated for model optimization.
* Supports python 3.6-3.8.

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

Image-based metrics
^^^^^^^^^^^^^^^^^^^
The group of metrics (such as PSNR, SSIM, BRISQUE) takes image or images as input.
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

Feature-based metrics
^^^^^^^^^^^^^^^^^^^^^

The group of metrics (such as IS, FID, KID) takes a list of image features.
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

Full Reference
^^^^^^^^^^^^^^

===========  ======  ==========
Acronym      Year    Metric
===========  ======  ==========
PSNR         \-      `Peak Signal-to-Noise Ratio <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_
SSIM         2003    `Structural Similarity <https://en.wikipedia.org/wiki/Structural_similarity>`_
MS-SSIM      2004    `Multi-Scale Structural Similarity <https://ieeexplore.ieee.org/abstract/document/1292216>`_
VIFp         2004    `Visual Information Fidelity <https://ieeexplore.ieee.org/document/1576816>`_
FSIM         2011    `Feature Similarity Index Measure <https://ieeexplore.ieee.org/document/5705575>`_
SR-SIM       2012    `Spectral Residual Based Similarity <https://sse.tongji.edu.cn/linzhang/ICIP12/ICIP-SR-SIM.pdf>`_
GMSD         2013    `Gradient Magnitude Similarity Deviation <https://arxiv.org/abs/1308.3052>`_
VSI          2014    `Visual Saliency-induced Index <https://ieeexplore.ieee.org/document/6873260>`_
DSS          2015    `DCT Subband Similarity Index <https://ieeexplore.ieee.org/document/7351172>`_
\-           2016    `Content Score <https://arxiv.org/abs/1508.06576>`_
\-           2016    `Style Score <https://arxiv.org/abs/1508.06576>`_
HaarPSI      2016    `Haar Perceptual Similarity Index <https://arxiv.org/abs/1607.06140>`_
MDSI         2016    `Mean Deviation Similarity Index <https://arxiv.org/abs/1608.07433>`_
MS-GMSD      2017    `Multi-Scale Gradient Magnitude Similarity Deviation <https://ieeexplore.ieee.org/document/7952357>`_
LPIPS        2018    `Learned Perceptual Image Patch Similarity <https://arxiv.org/abs/1801.03924>`_
PieAPP       2018    `Perceptual Image-Error Assessment through Pairwise Preference <https://arxiv.org/abs/1806.02067>`_
DISTS        2020    `Deep Image Structure and Texture Similarity <https://arxiv.org/abs/2004.07728>`_
===========  ======  ==========

No Reference
^^^^^^^^^^^^

===========  ======  ==========
Acronym      Year    Metric
===========  ======  ==========
TV           1937    `Total Variation <https://en.wikipedia.org/wiki/Total_variation>`_
BRISQUE      2012    `Blind/Referenceless Image Spatial Quality Evaluator <https://ieeexplore.ieee.org/document/6272356>`_
===========  ======  ==========

Feature based
^^^^^^^^^^^^^

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

As part of our library we provide code to benchmark all metrics on a set of common Mean Opinon Scores databases.
Currently we support `TID2013`_,  `KADID10k`_ and `PIPAL`_.
You need to download them separately and provide path to images as an argument to the script.

Here is an example how to evaluate SSIM and MS-SSIM metrics on TID2013 dataset:

.. code-block:: bash

   python3 tests/results_benchmark.py --dataset tid2013 --metrics SSIM MS-SSIM --path ~/datasets/tid2013 --batch_size 16

We report `Spearman's Rank Correlation cCoefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_ (SRCC)
and `Kendall rank correlation coefficient <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_ (KRCC).
We do not report `Pearson linear correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_ (PLCC)
as it's highly dependent on fitting method and is biased towards simple examples.

For metrics that can take greyscale or colour images, ``c`` means chromatic version.

===========  =================  ================================  =================  ================================  =================  ================================
     \                      TID2013                                              KADID10k                                             PIPAL
-----------  ---------------------------------------------------  ---------------------------------------------------  ---------------------------------------------------
  Acronym    SRCC / KRCC (PIQ)             SRCC / KRCC            SRCC / KRCC (PIQ)             SRCC / KRCC            SRCC / KRCC (PIQ)             SRCC / KRCC
===========  =================  ================================  =================  ================================  =================  ================================
PSNR         0.687 / 0.496      0.687 / 0.496 `TID2013`_          0.676 / 0.488      \- / -                            0.407 / 0.276      0.407 / 0.277 `PIPAL`_
SSIM         0.720 / 0.527      0.637 / 0.464 `TID2013`_          0.724 / 0.537      0.718 / 0.532 `KADID10k`_         0.498 / 0.345      0.529 / 0.369 `PIPAL`_
MS-SSIM      0.798 / 0.597      0.787 / 0.608 `TID2013`_          0.802 / 0.609      0.802 / 0.609 `KADID10k`_         0.552 / 0.389      0.462 / -
VIFp         0.610 / 0.458      0.610 / 0.457 `TID2013`_          0.650 / 0.477      0.650 / 0.477 `KADID10k`_         0.497 / 0.345      \- / -
FSIM         0.802 / 0.629      0.801 / 0.630 `TID2013`_          0.830 / 0.639      0.829 / 0.639 `KADID10k`_         0.588 / 0.415      0.596 / 0.421 `PIPAL`_
FSIMc        0.851 / 0.667      0.851 / 0.667 `TID2013`_          0.854 / 0.665      0.854 / 0.665 `KADID10k`_         0.590 / 0.416      \- / -
SR-SIM       0.807 / 0.641      0.808 / 0.641 `Eval2019`_         0.839 / 0.652      0.839 / 0.652 `KADID10k`_         0.565 / 0.399      \- / -
SR-SIMc      0.870 / 0.692      \- / -                            0.869 / 0.685      \- / -                            0.569 / 0.401      \- / -
GMSD         0.804 / 0.633      0.803 / 0.635 `MS-GMSD`_          0.847 / 0.664      0.847 / 0.664 `KADID10k`_         0.584 / 0.414      \- / -
VSI          0.895 / 0.716      0.897 / 0.718 `Eval2019`_         0.878 / 0.690      0.861 / 0.678 `KADID10k`_         0.539 / 0.375      \- / -
DSS          0.791 / 0.614      0.792 / - `Eval2019`_             0.860 / 0.674      0.860 / 0.674 `KADID10k`_         0.632 / 0.456      \- / -
Content      0.705 / 0.517      \- / -                            0.724 / 0.533      \- / -                            0.450 / 0.307      \- / -
Style        0.538 / 0.372      \- / -                            0.647 / 0.465      \- / -                            0.343 / 0.231      \- / -
HaarPSI      0.873 / 0.692      0.873 / 0.692 `HaarPSI`_          0.885 / 0.700      0.885 / 0.699 `KADID10k`_         0.589 / 0.417      \- / -
MDSI         0.890 / 0.712      0.890 / 0.712 `MDSI`_             0.885 / 0.702      0.885 / 0.702 `KADID10k`_         0.585 / 0.408      \- / -
MS-GMSD      0.812 / 0.646      0.814 / 0.647 `MS-GMSD`_          0.852 / 0.669      \- / -                            0.585 / 0.414      \- / -
MS-GMSDc     0.888 / 0.711      0.687 / 0.496 `MS-GMSD`_          0.870 / 0.683      \- / -                            0.587 / 0.416      \- / -
LPIPS-VGG    0.670 / 0.497      0.670 / 0.497 `DISTS`_            0.720 / 0.531      \- / -                            0.573 / 0.404      0.577 / 0.408 `PIPAL`_
PieAPP       0.836 / 0.650      0.875 / 0.710 `DISTS`_            0.866 / 0.676      \- / -                            0.698 / 0.509      0.711 / 0.521 `PIPAL`_
DISTS        0.805 / 0.613      0.830 / 0.639 `DISTS`_            0.875 / 0.695      \- / -                            0.617 / 0.438      0.664 / 0.477 `PIPAL`_
===========  =================  ================================  =================  ================================  =================  ================================

.. _TID2013: http://www.ponomarenko.info/tid2013.htm
.. _KADID10k: http://database.mmsp-kn.de/kadid-10k-database.html
.. _Eval2019: https://ieeexplore.ieee.org/abstract/document/8847307/
.. _`MDSI`: https://arxiv.org/abs/1608.07433
.. _MS-GMSD: https://ieeexplore.ieee.org/document/7952357
.. _DISTS: https://arxiv.org/abs/2004.07728
.. _HaarPSI: https://arxiv.org/abs/1607.06140
.. _PIPAL: https://arxiv.org/pdf/2011.15002.pdf

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

   @misc{piq,
     title={{PyTorch Image Quality}: Metrics and Measure for Image Quality Assessment},
     url={https://github.com/photosynthesis-team/piq},
     note={Open-source software available at https://github.com/photosynthesis-team/piq},
     author={Sergey Kastryulin and Dzhamil Zakirov and Denis Prokopenko},
     year={2019},
   }

.. citation-section-end

.. contacts-section-start

Contacts
--------

**Sergey Kastryulin** - `@snk4tr <https://github.com/snk4tr>`_ - ``snk4tr@gmail.com``

**Djamil Zakirov** - `@zakajd <https://github.com/zakajd>`_ - ``djamilzak@gmail.com``

**Denis Prokopenko** - `@denproc <https://github.com/denproc>`_ - ``d.prokopenko@outlook.com``

.. contacts-section-end
