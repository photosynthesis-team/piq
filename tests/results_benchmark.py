"""Code for computation of PLCC, SRCC and KRCC between 
    PIQ metrics predictions and ground truth scores from MOS databases.
"""
import argparse
import functools

import piq
import torch
import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.io import imread
from typing import Optional, List, Callable


METRICS = {
    # Full Reference
    "PSNR": functools.partial(piq.psnr, reduction='none'),
    "SSIM": functools.partial(piq.ssim, reduction='none'),
    "MS-SSIM": functools.partial(piq.multi_scale_ssim, reduction='none'),
    "VIFp": functools.partial(piq.vif_p, reduction='none'), 
    "GMSD": functools.partial(piq.gmsd, reduction='none'),  
    "MS-GMSD": functools.partial(piq.multi_scale_gmsd, reduction='none'),
    "MS-GMSDc": functools.partial(piq.multi_scale_gmsd, chromatic=True, reduction='none'),
    "FSIM": functools.partial(piq.fsim, chromatic=False, reduction='none'),
    "FSIMc": functools.partial(piq.fsim, chromatic=True, reduction='none'),
    "VSI": functools.partial(piq.vsi, reduction='none'),  
    "MDSI": functools.partial(piq.mdsi, reduction='none'),
    "LPIPS-vgg": piq.LPIPS(replace_pooling=False, reduction='none'),
    "DISTS": piq.DISTS(reduction='none'),
    "PieAPP": piq.PieAPP(reduction='none'),

    # No Reference
    "BRISQUE": functools.partial(piq.brisque, reduction='none')
}


class TID2013(torch.utils.data.Dataset):
    """
    Args:
        root: Root directory path.

    Returns:
        x: image with some kind of distortion in [0, 1] range
        y: image without distortion in [0, 1] range
        score: MOS score for this pair of images
    """
    _filename = "mos_with_names.txt"

    def __init__(self, root: Path = "datasets/tid2013"):
        assert root.exists(),\
            "You need to download TID2013 dataset first. Check http://www.ponomarenko.info/tid2013"

        df = pd.read_csv(
            root / self._filename,
            sep=' ',
            names=['score', 'dist_img'],
            header=None
        )
        df["ref_img"] = df["dist_img"].apply(lambda x: f"reference_images/{(x[:3] + x[-4:]).upper()}")
        df["dist_img"] = df["dist_img"].apply(lambda x: f"distorted_images/{x}")

        self.scores = df['score'].to_numpy()
        self.df = df[["dist_img", 'ref_img', 'score']]
        self.root = root

    def __getitem__(self, index):
        x_path = self.root / self.df.iloc[index][0]
        y_path = self.root / self.df.iloc[index][1]
        score = self.scores[index]

        # Load image and ref. Convert to tensor and [0, 1] range
        x = torch.tensor(imread(x_path)).permute(2, 0, 1) / 255
        y = torch.tensor(imread(y_path)).permute(2, 0, 1) / 255

        return x, y, score

    def __len__(self):
        return len(self.df)


class KADID10k(TID2013):
    _filename = "dmos.csv"

    def __init__(self, root: Path = "datasets/kadid10k"):
        assert root.exists(),\
            "You need to download KADID10K dataset first. Check http://database.mmsp-kn.de/kadid-10k-database.html"

        # Read file mith DMOS
        self.df = pd.read_csv(root / self._filename)
        self.df.rename(columns={"dmos": "score"}, inplace=True)
        self.scores = self.df["score"].to_numpy()
        self.df = self.df[["dist_img", 'ref_img', 'score']]

        self.root = root / "images"


DATASETS = {
    "tid2013": TID2013,
    "kadid10k": KADID10k,
}


def eval_metric(loader: torch.utils.data.DataLoader, metric: Callable, device: str) -> List:
    """Evaluate metric on a given dataset.
    Args:
        loader: PyTorch dataloader that returns batch of distorted images, reference images and scores.
        metric: Should support `metric(x, y)` or `metric(x)` call.
        device: Computation device. 

    Returns:
        gt_scores: Ground truth values
        metric_scores: Predicted values as torch.Tensors.
    """
    assert isinstance(loader, torch.utils.data.DataLoader), "Expect loader to be DataLoader class"
    assert callable(metric), f"Expected metric to be callable, got {type(metric)} instead!"

    gt_scores = []
    metric_scores = []

    for (distorted_images, reference_images, scores) in tqdm.tqdm(loader, ncols=50):
        distorted_images, reference_images = distorted_images.to(device), reference_images.to(device)
        gt_scores.append(scores.cpu())

        # Full Reference methods
        metric_score = metric(distorted_images, reference_images).cpu()
        metric_scores.append(metric_score.cpu())

    return torch.cat(gt_scores).numpy(), torch.cat(metric_scores).numpy()


def main(dataset_name: str, path: Path, metrics: List, batch_size: int, device: str) -> None:
    
    # Init dataset and dataloader
    dataset = DATASETS[dataset_name](root=path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

    for name in metrics:
        gt_scores, metric_scores = eval_metric(loader, METRICS[name], device=device)
        print(f"{name}: SRCC {abs(spearmanr(gt_scores, metric_scores)[0]):0.4f}",
                      f"KRCC {abs(kendalltau(gt_scores, metric_scores)[0]):0.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PIQ metrics")

    # General
    parser.add_argument("--dataset", type=str, help="Dataset name", choices=list(DATASETS.keys()))
    parser.add_argument("--path", type=Path, help="Path to dataset")
    parser.add_argument('--metrics', nargs='+', default=[], help='Metrics to benchmark', choices=list(METRICS.keys()))
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='Computation device')

    args = parser.parse_args()
    print(f"Parameters used for benchmark: {args}")
    main(
        dataset_name=args.dataset, 
        path=args.path,
        metrics=args.metrics,
        batch_size=args.batch_size,
        device=args.device
    )