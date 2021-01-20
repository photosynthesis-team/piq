## Benchmark
As part of our library we provide code to benchmark all metrics on a set of common Mean Opinon Scores databases.
Currently only [TID2013](http://www.ponomarenko.info/tid2013) and [KADID10k](http://database.mmsp-kn.de/kadid-10k-database.html) are supported. You need to download them separately and provide valid path to folder as an argument to the script.

Here is an example how to evaluate SSIM and MS-SSIM metrics on TID2013 dataset:
`python3 results_benchmark.py --dataset tid2013 --metrics SSIM MS-SSIM --path ~/datasets/tid2013 --batch_size 16`

We report [Spearman's rank correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) (SRCC) and [Kendall rank correlation coefficient](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) (KRCC). We do not report [Pearson linear correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) because it's highly dependent on fitting method and is biased towards simple examples.

