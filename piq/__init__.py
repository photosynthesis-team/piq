__version__ = "0.8.0"

from .ssim import ssim, SSIMLoss
from .ms_ssim import multi_scale_ssim, MultiScaleSSIMLoss
from .msid import MSID
from .fid import FID
from .kid import KID
from .pr import PR
from .tv import TVLoss, total_variation
from .gmsd import gmsd, multi_scale_gmsd, GMSDLoss, MultiScaleGMSDLoss
from .gs import GS
from .isc import IS, inception_score
from .vif import VIFLoss, vif_p
from .brisque import BRISQUELoss, brisque
from .perceptual import StyleLoss, ContentLoss, LPIPS, DISTS
from .psnr import psnr
from .fsim import fsim, FSIMLoss
from .vsi import vsi, VSILoss
from .mdsi import mdsi, MDSILoss
from .haarpsi import haarpsi, HaarPSILoss
from .srsim import srsim, SRSIMLoss
from .pieapp import PieAPP
from .dss import dss, DSSLoss
from .iw_ssim import information_weighted_ssim, InformationWeightedSSIMLoss
from .clip_iqa import CLIPIQA
