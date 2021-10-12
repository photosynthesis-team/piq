"""Code for computation of PLCC, SRCC and KRCC between
    PIQ metrics predictions and ground truth scores from MOS databases.
"""
import piq
import tqdm
import torch
import argparse
import functools

import pandas as pd
import numpy as np

from typing import List, Callable, Tuple
from pathlib import Path
from skimage.io import imread
from scipy.stats import spearmanr, kendalltau
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass

from piq.feature_extractors import InceptionV3


@dataclass
class Metric:
    name: str
    functor: Callable
    category: str  # FR - full-reference, NR - no-reference, DB - distribution-based

    def __post_init__(self):
        valid_categories = {'FR', 'NR', 'DB'}
        assert self.category in valid_categories, f'Provided category [{self.category}] is invalid. ' \
                                                  f'Provide one of: {valid_categories}'


torch.multiprocessing.set_sharing_strategy('file_system')

METRICS = {
    # Full-reference
    "PSNR": Metric(name="PSNR", functor=functools.partial(piq.psnr, reduction='none'), category='FR'),
    "SSIM": Metric(name="SSIM", functor=functools.partial(piq.ssim, reduction='none'), category='FR'),
    "MS-SSIM": Metric(name="MS-SSIM", functor=functools.partial(piq.multi_scale_ssim, reduction='none'), category='FR'),
    "VIFp": Metric(name="VIFp", functor=functools.partial(piq.vif_p, reduction='none'), category='FR'),
    "GMSD": Metric(name="GMSD", functor=functools.partial(piq.gmsd, reduction='none'), category='FR'),
    "MS-GMSD": Metric(name="MS-GMSD", functor=functools.partial(piq.multi_scale_gmsd, reduction='none'), category='FR'),
    "MS-GMSDc": Metric(name="MS-GMSDc", functor=functools.partial(piq.multi_scale_gmsd,
                                                                  chromatic=True, reduction='none'), category='FR'),
    "FSIM": Metric(name="FSIM", functor=functools.partial(piq.fsim, chromatic=False, reduction='none'), category='FR'),
    "FSIMc": Metric(name="FSIMc", functor=functools.partial(piq.fsim, chromatic=True, reduction='none'), category='FR'),
    "VSI": Metric(name="VSI", functor=functools.partial(piq.vsi, reduction='none'), category='FR'),
    "HaarPSI": Metric(name="HaarPSI", functor=functools.partial(piq.haarpsi, reduction='none'), category='FR'),
    "MDSI": Metric(name="MDSI", functor=functools.partial(piq.mdsi, reduction='none'), category='FR'),
    "LPIPS-vgg": Metric(name="LPIPS-vgg", functor=piq.LPIPS(replace_pooling=False, reduction='none'), category='FR'),
    "DISTS": Metric(name="DISTS", functor=piq.DISTS(reduction='none'), category='FR'),
    "PieAPP": Metric(name="PieAPP", functor=piq.PieAPP(reduction='none'), category='FR'),
    "Content": Metric(name="Content", functor=piq.ContentLoss(reduction='none'), category='FR'),
    "Style": Metric(name="Style", functor=piq.StyleLoss(reduction='none'), category='FR'),
    "DSS": Metric(name="DSS", functor=functools.partial(piq.dss, reduction='none'), category='FR'),

    # No-reference
    "BRISQUE": Metric(name="BRISQUE", functor=functools.partial(piq.brisque, reduction='none'), category='NR'),

    # Distribution-based
    "IS": Metric(name="IS", functor=piq.IS(distance='l1'), category='DB'),
    "FID": Metric(name="FID", functor=piq.FID(), category='DB'),
    "KID": Metric(name="KID", functor=piq.KID(), category='DB'),
    "MSID": Metric(name="MSID", functor=piq.MSID(), category='DB')
}


class TID2013(Dataset):
    """
    Args:
        root: Root directory path.
    Returns:
        x: image with some kind of distortion in [0, 1] range
        y: image without distortion in [0, 1] range
        score: MOS score for this pair of images
    """
    _filename = "mos_with_names.txt"

    def __init__(self, root: Path = "datasets/tid2013") -> None:
        assert root.exists(), \
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

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_path = self.root / self.df.iloc[index][0]
        y_path = self.root / self.df.iloc[index][1]
        score = self.scores[index]

        # Load image and ref. Convert to tensor and [0, 1] range
        x = torch.tensor(imread(x_path)).permute(2, 0, 1) / 255
        y = torch.tensor(imread(y_path)).permute(2, 0, 1) / 255

        return x, y, score

    def __len__(self) -> int:
        return len(self.df)


class KADID10k(TID2013):
    _filename = "dmos.csv"

    def __init__(self, root: Path = "datasets/kadid10k"):
        super().__init__()
        assert root.exists(), \
            "You need to download KADID10K dataset first. Check http://database.mmsp-kn.de/kadid-10k-database.html"

        # Read file mith DMOS
        self.df = pd.read_csv(root / self._filename)
        self.df.rename(columns={"dmos": "score"}, inplace=True)
        self.scores = self.df["score"].to_numpy()
        self.df = self.df[["dist_img", 'ref_img', 'score']]

        self.root = root / "images"


class PIPAL(TID2013):
    """Class to evaluate on train set of PIPAL dataset"""

    def __init__(self, root: Path = Path("data/raw/pipal")):
        assert root.exists(), \
            "You need to download PIPAL dataset. Check https://www.jasongt.com/projectpages/pipal.html"

        assert (root / "Train_Dist").exists(), \
            "Please place all distorted files into single folder named `Train_Dist`."

        # Read files with labels and merge them into single DF
        dfs = []
        for filename in (root / "Train_Label").glob("*.txt"):
            df = pd.read_csv(filename, index_col=None, header=None, names=['dist_img', 'score'])
            dfs.append(df)

        df = pd.concat(dfs, axis=0, ignore_index=True)

        df["ref_img"] = df["dist_img"].apply(lambda x: f"Train_Ref/{x[:5] + x[-4:]}")
        df["dist_img"] = df["dist_img"].apply(lambda x: f"Train_Dist/{x}")

        self.scores = df["score"].to_numpy()
        self.df = df[["dist_img", 'ref_img', 'score']]
        self.root = root


DATASETS = {
    "tid2013": TID2013,
    "kadid10k": KADID10k,
    "pipal": PIPAL,
}


def eval_metric(loader: DataLoader, metric: Metric, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate metric on a given dataset.
    Args:
        loader: PyTorch dataloader that returns batch of distorted images, reference images and scores.
        metric: General metric that satisfies the Metric interface.
        device: Computation device.
    Returns:
        gt_scores: Ground truth values.
        metric_scores: Predicted values as torch.Tensors.
    """
    assert isinstance(loader, DataLoader), "Expect loader to be DataLoader class"
    assert isinstance(metric, Metric), f"Expected metric to be an instance of Metric, got {type(metric)} instead!"

    gt_scores = []
    metric_scores = []
    compute_function = determine_compute_function(metric_category=metric.category)

    for distorted_images, reference_images, scores in tqdm.tqdm(loader, ncols=50):
        distorted_images, reference_images = distorted_images.to(device), reference_images.to(device)
        gt_scores.append(scores.cpu())

        metric_score = compute_function(metric.functor, distorted_images, reference_images, device)
        if metric_score.dim() == 0:
            metric_score = metric_score.unsqueeze(0)

        metric_scores.append(metric_score.cpu())

    return torch.cat(gt_scores).numpy(), torch.cat(metric_scores).numpy()


def determine_compute_function(metric_category: str) -> Callable:
    return {
        'FR': compute_full_reference,
        'NR': compute_no_reference,
        'DB': compute_distribution_based
    }[metric_category]


def compute_full_reference(metric_functor: Callable, distorted_images: torch.Tensor,
                           reference_images: torch.Tensor, _) -> np.ndarray:
    return metric_functor(distorted_images, reference_images).cpu()


def compute_no_reference(metric_functor: Callable, distorted_images: torch.Tensor, _, __) -> np.ndarray:
    return metric_functor(distorted_images).cpu()


def compute_distribution_based(metric_functor: Callable, distorted_images: torch.Tensor,
                               reference_images: torch.Tensor, device: str) -> np.ndarray:
    feature_extractor = InceptionV3().to(device)
    distorted_features, reference_features = [], []

    # Create patches
    distorted_patches = crop_patches(distorted_images, size=96, stride=32)
    reference_patches = crop_patches(reference_images, size=96, stride=32)

    # Extract features from distorted images
    distorted_patch_loader = distorted_patches.view(-1, 10, *distorted_patches.shape[-3:])
    reference_patch_loader = reference_patches.view(-1, 10, *reference_patches.shape[-3:])
    for distorted in distorted_patch_loader:
        with torch.no_grad():
            distorted_features.append(feature_extractor(distorted)[0].squeeze())

    for reference in reference_patch_loader:
        with torch.no_grad():
            reference_features.append(feature_extractor(reference)[0].squeeze())

    distorted_features = torch.cat(distorted_features, dim=0)
    reference_features = torch.cat(reference_features, dim=0)

    return metric_functor(distorted_features, reference_features).cpu()


def crop_patches(images: torch.Tensor, size=64, stride=32):
    """Crop input images into smaller patches
    Args:
        images: Tensor of images with shape (batch x 3 x H x W)
        size: size of a square patch
        stride: Step between patches
    """
    patches = images.data.unfold(1, 3, 3).unfold(2, size, stride).unfold(3, size, stride)
    patches = patches.reshape(-1, 3, size, size)
    return patches


def main(dataset_name: str, path: Path, metrics: List[str], batch_size: int, device: str) -> None:
    # Init dataset and dataloader
    dataset = DATASETS[dataset_name](root=path)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    for metric_name in metrics:
        metric: Metric = METRICS[metric_name]
        gt_scores, metric_scores = eval_metric(loader, metric, device=device)
        print(f"{metric_name}: SRCC {abs(spearmanr(gt_scores, metric_scores)[0]):0.3f}",
              f"KRCC {abs(kendalltau(gt_scores, metric_scores)[0]):0.3f}")


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
