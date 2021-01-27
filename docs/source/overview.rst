Overview
========

PIQ helps you to concentrate on your experiments without the boilerplate code.
The library contains a set of measures and metrics that is continually getting extended.
For measures/metrics that can be used as loss functions, corresponding PyTorch modules are implemented.

PIQ provides:

* Unified interface, which is easy to use and extend.
* Written on pure PyTorch with bare minima of additional dependencies.
* Extensive user input validation. You code will not crash in the middle of the training.
* Fast (GPU computations available) and reliable.
* Most metrics can be backpropagated for model optimization.
* Supports python 3.6-3.8.

Benchmark
^^^^^^^^^

We report `Spearman's rank correlation coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_
(SRCC) and `Kendall rank correlation coefficient <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_ (KRCC).
We do not report `Pearson linear correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_ (PLCC)
because it's highly dependent on fitting method and is biased towards simple examples.

``c`` means chromatic version.

===========  ==================================  ==================================  ================================  ===========================
Acronym      TID2013: SRCC / KRCC (piq)          TID2013: SRCC / KRCC                KADID10k: SRCC / KRCC (piq)       KADID10k: SRCC / KRCC
===========  ==================================  ==================================  ================================  ===========================
PSNR         0.6869 / 0.4958                      0.687 / 0.496  `TID2013`_            0.6757 / 0.4876                    \-   /    -
SSIM         0.5544 / 0.3883                      0.637 / 0.464  `TID2013`_            0.6329 / 0.4571                   0.718 / 0.532 `KADID10k`_
MS-SSIM      0.7983 / 0.5965                      0.787 / 0.608  `TID2013`_            0.8020 / 0.6088                   0.802 / 0.609 `KADID10k`_
VIFp         0.6102 / 0.4579                      0.610 / 0.457  `TID2013`_            0.6500 / 0.4770                   0.650 / 0.477 `KADID10k`_
FSIM         0.8015 / 0.6289                      0.801 / 0.630  `TID2013`_            0.8294 / 0.6390                   0.829 / 0.639 `KADID10k`_
FSIMc        0.8509 / 0.6665                      0.851 / 0.667  `TID2013`_            0.8537 / 0.6650                   0.854 / 0.665 `KADID10k`_
IW-PSNR        \-   /    -                       0.6913 /   \-   `TID2013`_              \-   /    -                      \-   /    -
IW-SSIM        \-   /    -                       0.7779 / 0.5977 `TID2013`_              \-   /    -                      \-   /    -
SR-SIM         \-   /    -                       0.8076 / 0.6406 `TID2013`_              \-   /    -                     0.839 / 0.652 `KADID10k`_
SR-SIMc        \-   /    -                         \-   /    -                           \-   /    -                      \-   /    -
GMSD         0.8038 / 0.6334                     0.8030 / 0.6352 `TID2013`_            0.8474 / 0.6640                   0.847 / 0.664 `KADID10k`_
VSI          0.8949 / 0.7159                     0.8965 / 0.7183 `TID2013`_            0.8780 / 0.6899                   0.861 / 0.678 `KADID10k`_
Content      0.7049 / 0.5173                       \-   /    -                         0.7237 / 0.5326                    \-   /    -
Style        0.5384 / 0.3720                       \-   /    -                         0.6470 / 0.4646                    \-   /    -
HaarPSI      0.8732 / 0.6923                     0.8732 / 0.6923 `TID2013`_            0.8849 / 0.6995                   0.885 / 0.699 `KADID10k`_
MDSI         0.8899 / 0.7123                     0.8899 / 0.7123 `TID2013`_            0.8853 / 0.7023                   0.885 / 0.702 `KADID10k`_
MS-GMSD      0.8121 / 0.6455                     0.8139 / 0.6467 `TID2013`_            0.8523 / 0.6692                    \-   /    -
MS-GMSDc     0.8875 / 0.7105                      0.687 / 0.496  `TID2013`_            0.8697 / 0.6831                    \-   /    -
LPIPS-VGG    0.6696 / 0.4970                      0.670 / 0.497  `TID2013`_            0.7201 / 0.5313                    \-   /    -
PieAPP       0.8355 / 0.6495                      0.875 / 0.710  `TID2013`_            0.8655 / 0.6758                    \-   /    -
DISTS        0.7077 / 0.5212                      0.830 / 0.639  `TID2013`_            0.8137 / 0.6254                    \-   /    -
===========  ==================================  ==================================  ================================  ===========================

As part of our library we provide code to benchmark all metrics on a set of common Mean Opinon Scores databases.
Currently only `TID2013`_
and `KADID10k`_ are supported.
You need to download them separately and provide path to images as an argument to the script.

Here is an example how to evaluate SSIM and MS-SSIM metrics on TID2013 dataset:
::

    python3 tests/results_benchmark.py --dataset tid2013 --metrics SSIM MS-SSIM --path ~/datasets/tid2013 --batch_size 16


.. _TID2013: http://www.ponomarenko.info/tid2013.htm
.. _KADID10k: http://database.mmsp-kn.de/kadid-10k-database.html