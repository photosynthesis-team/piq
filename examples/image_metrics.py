import torch
import piq
from skimage.io import imread


@torch.no_grad()
def main():
    # Read RGB image and it's noisy version
    prediction = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1) / 255.
    target = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1) / 255.
    
    if torch.cuda.is_available():
        # Move to GPU to make computaions faster
        prediction = prediction.cuda()
        target = target.cuda()

    # To compute BRISQUE score as a measure, use lower case function from the library
    brisque_index: torch.Tensor = piq.brisque(prediction, data_range=1., reduction='none')
    # In order to use BRISQUE as a loss function, use corresponding PyTorch module.
    # Note: the back propagation is not available using torch==1.5.0.
    # Update the environment with latest torch and torchvision.
    brisque_loss: torch.Tensor = piq.BRISQUELoss(data_range=1., reduction='none')(prediction)
    print(f"BRISQUE index: {brisque_index.item():0.4f}, loss: {brisque_loss.item():0.4f}")

    # To compute Content score as a loss function, use corresponding PyTorch module
    # By default VGG16 model is used, but any feature extractor model is supported.
    # Don't forget to adjust layers names accordingly. Features from different layers can be weighted differently.
    # Use weights parameter. See other options in class docstring.
    content_loss = piq.ContentLoss(
        feature_extractor="vgg16", layers=("relu3_3", ), reduction='none')(prediction, target)
    print(f"ContentLoss: {content_loss.item():0.4f}")

    # To compute DISTS as a loss function, use corresponding PyTorch module
    # By default input images are normalized with ImageNet statistics before forwarding through VGG16 model.
    # If there is no need to normalize the data, use mean=[0.0, 0.0, 0.0] and std=[1.0, 1.0, 1.0].
    dists_loss = piq.DISTS(reduction='none')(prediction, target)
    print(f"DISTS: {dists_loss.item():0.4f}")

    # To compute FSIM as a measure, use lower case function from the library
    fsim_index: torch.Tensor = piq.fsim(prediction, target, data_range=1., reduction='none')
    # In order to use FSIM as a loss function, use corresponding PyTorch module
    fsim_loss = piq.FSIMLoss(data_range=1., reduction='none')(prediction, target)
    print(f"FSIM index: {fsim_index.item():0.4f}, loss: {fsim_loss.item():0.4f}")

    # To compute GMSD as a measure, use lower case function from the library
    # This is port of MATLAB version from the authors of original paper.
    # In any case it should me minimized. Usually values of GMSD lie in [0, 0.35] interval.
    gmsd_index: torch.Tensor = piq.gmsd(prediction, target, data_range=1., reduction='none')
    # In order to use GMSD as a loss function, use corresponding PyTorch module:
    gmsd_loss: torch.Tensor = piq.GMSDLoss(data_range=1., reduction='none')(prediction, target)
    print(f"GMSD index: {gmsd_index.item():0.4f}, loss: {gmsd_loss.item():0.4f}")

    # To compute HaarPSI as a measure, use lower case function from the library
    # This is port of MATLAB version from the authors of original paper.
    haarpsi_index: torch.Tensor = piq.haarpsi(prediction, target, data_range=1., reduction='none')
    # In order to use HaarPSI as a loss function, use corresponding PyTorch module
    haarpsi_loss: torch.Tensor = piq.HaarPSILoss(data_range=1., reduction='none')(prediction, target)
    print(f"HaarPSI index: {haarpsi_index.item():0.4f}, loss: {haarpsi_loss.item():0.4f}")

    # To compute LPIPS as a loss function, use corresponding PyTorch module
    lpips_loss: torch.Tensor = piq.LPIPS(reduction='none')(prediction, target)
    print(f"LPIPS: {lpips_loss.item():0.4f}")

    # To compute MDSI as a measure, use lower case function from the library
    mdsi_index: torch.Tensor = piq.mdsi(prediction, target, data_range=1., reduction='none')
    # In order to use MDSI as a loss function, use corresponding PyTorch module
    mdsi_loss: torch.Tensor = piq.MDSILoss(data_range=1., reduction='none')(prediction, target)
    print(f"MDSI index: {mdsi_index.item():0.4f}, loss: {mdsi_loss.item():0.4f}")

    # To compute MS-SSIM index as a measure, use lower case function from the library:
    ms_ssim_index: torch.Tensor = piq.multi_scale_ssim(prediction, target, data_range=1.)
    # In order to use MS-SSIM as a loss function, use corresponding PyTorch module:
    ms_ssim_loss = piq.MultiScaleSSIMLoss(data_range=1., reduction='none')(prediction, target)
    print(f"MS-SSIM index: {ms_ssim_index.item():0.4f}, loss: {ms_ssim_loss.item():0.4f}")

    # To compute Multi-Scale GMSD as a measure, use lower case function from the library
    # It can be used both as a measure and as a loss function. In any case it should me minimized.
    # By defualt scale weights are initialized with values from the paper.
    # You can change them by passing a list of 4 variables to scale_weights argument during initialization
    # Note that input tensors should contain images with height and width equal 2 ** number_of_scales + 1 at least.
    ms_gmsd_index: torch.Tensor = piq.multi_scale_gmsd(
        prediction, target, data_range=1., chromatic=True, reduction='none')
    # In order to use Multi-Scale GMSD as a loss function, use corresponding PyTorch module
    ms_gmsd_loss: torch.Tensor = piq.MultiScaleGMSDLoss(
        chromatic=True, data_range=1., reduction='none')(prediction, target)
    print(f"MS-GMSDc index: {ms_gmsd_index.item():0.4f}, loss: {ms_gmsd_loss.item():0.4f}")

    # To compute PSNR as a measure, use lower case function from the library.
    psnr_index = piq.psnr(prediction, target, data_range=1., reduction='none')
    print(f"PSNR index: {psnr_index.item():0.4f}")
    
    # To compute PieAPP as a loss function, use corresponding PyTorch module:
    pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=32)(prediction, target)
    print(f"PieAPP loss: {pieapp_loss.item():0.4f}")

    # To compute SSIM index as a measure, use lower case function from the library:
    ssim_index = piq.ssim(prediction, target, data_range=1.)
    # In order to use SSIM as a loss function, use corresponding PyTorch module:
    ssim_loss: torch.Tensor = piq.SSIMLoss(data_range=1.)(prediction, target)
    print(f"SSIM index: {ssim_index.item():0.4f}, loss: {ssim_loss.item():0.4f}")

    # To compute Style score as a loss function, use corresponding PyTorch module:
    # By default VGG16 model is used, but any feature extractor model is supported.
    # Don't forget to adjust layers names accordingly. Features from different layers can be weighted differently.
    # Use weights parameter. See other options in class docstring.
    style_loss = piq.StyleLoss(feature_extractor="vgg16", layers=("relu3_3", ))(prediction, target)
    print(f"Style: {style_loss.item():0.4f}")

    # To compute TV as a measure, use lower case function from the library:
    tv_index: torch.Tensor = piq.total_variation(prediction)
    # In order to use TV as a loss function, use corresponding PyTorch module:
    tv_loss: torch.Tensor = piq.TVLoss(reduction='none')(prediction)
    print(f"TV index: {tv_index.item():0.4f}, loss: {tv_loss.item():0.4f}")

    # To compute VIF as a measure, use lower case function from the library:
    vif_index: torch.Tensor = piq.vif_p(prediction, target, data_range=1.)
    # In order to use VIF as a loss function, use corresponding PyTorch class:
    vif_loss: torch.Tensor = piq.VIFLoss(sigma_n_sq=2.0, data_range=1.)(prediction, target)
    print(f"VIFp index: {vif_index.item():0.4f}, loss: {vif_loss.item():0.4f}")

    # To compute VSI score as a measure, use lower case function from the library:
    vsi_index: torch.Tensor = piq.vsi(prediction, target, data_range=1.)
    # In order to use VSI as a loss function, use corresponding PyTorch module:
    vsi_loss: torch.Tensor = piq.VSILoss(data_range=1.)(prediction, target)
    print(f"VSI index: {vsi_index.item():0.4f}, loss: {vsi_loss.item():0.4f}")


if __name__ == '__main__':
    main()
