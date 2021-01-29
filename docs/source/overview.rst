Overview
========

`PyTorch Image Quality (PIQ) <https://github.com/photosynthesis-team/piq>`_ is a collection of measures and metrics for image quality assessment.
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

.. csv-table::
   :file: _files/benchmark.csv
   :header-rows: 2

As part of our library we provide code to benchmark all metrics on a set of common Mean Opinon Scores databases.
Currently only `TID2013`_
and `KADID10k`_ are supported.
You need to download them separately and provide path to images as an argument to the script.

Here is an example how to evaluate SSIM and MS-SSIM metrics on TID2013 dataset:
::

    python3 tests/results_benchmark.py --dataset tid2013 --metrics SSIM MS-SSIM --path ~/datasets/tid2013 --batch_size 16


.. _TID2013: http://www.ponomarenko.info/tid2013.htm
.. _KADID10k: http://database.mmsp-kn.de/kadid-10k-database.html