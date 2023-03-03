import os
import shutil
import torchvision.transforms as transforms
from PIL import Image

import torch
from piq import ssim, SSIMLoss
from piq import LPIPS
from piq import psnr

import torchvision.transforms as transforms

import json

SSIM_comparisons = dict()
LPIPS_comparisons = dict()
PSNR_comparison = dict()

ground_truth_PATH = 'datasets/polar_1024'

comparison_PATH = 'datasets/SynthDataset_Default'

transform = transforms.Compose(
    [transforms.Grayscale(1),
    transforms.ToTensor()]
)

for filename in os.listdir(ground_truth_PATH):
    gt_img_path = os.path.join(ground_truth_PATH, filename)
    c_img_path = os.path.join(comparison_PATH, filename)
    if os.path.isfile(c_img_path):

        gt = Image.open(gt_img_path)
        c = Image.open(c_img_path)

        gt, c = transform(gt).unsqueeze(0), transform(c).unsqueeze(0)
        # print(gt.shape)
        # print(c.shape)

        ssim_index: torch.Tensor = ssim(gt, c, data_range=255)
        SSIM_comparisons[filename] = ssim_index.item()

        LPIPS_comparisons[filename] = LPIPS()(gt, c).item()

        PSNR_comparison[filename] = psnr(gt, c, data_range=255).item()

results_path = "comparisons"

# Define the filename for the JSON file
filenames = [('SSIM.json', SSIM_comparisons), ('LPIPS.json', LPIPS_comparisons), ('PSNR.json', PSNR_comparison)]

#saves results to comparisons directory
for file, data in filenames:
    filepath = os.path.join(results_path, file)
    # Open the file in write mode and save the dictionary as JSON
    with open(filepath, "w") as f:
        json.dump(data, f)
