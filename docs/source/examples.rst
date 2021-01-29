Usage Examples
==============

Image-based metrics
^^^^^^^^^^^^^^^^^^^
The group of metrics (such as PSNR, SSIM, BRISQUE) takes image or images as input.
We have a functional interface, which returns a metric value, and a class interface, which allows us to use any metric
as a loss function.
::

    import torch
    from piq import ssim, SSIMLoss

    prediction = torch.rand(4, 3, 256, 256, requires_grad=True)
    target = torch.rand(4, 3, 256, 256)

    ssim_index: torch.Tensor = ssim(prediction, target, data_range=1.)

    loss = SSIMLoss(data_range=1.)
    output: torch.Tensor = loss(prediction, target)
    output.backward()

For a full list of examples, see `image metrics <https://github.com/photosynthesis-team/piq/blob/master/examples/image_metrics.py>`_ examples.

Feature-based metrics
^^^^^^^^^^^^^^^^^^^^^

The group of metrics (such as IS, FID, KID) takes a list of image features.
Image features can be extracted by some feature extractor network separately or by using the ``compute_feats`` method of a
class.

Note:
    ``compute_feats`` consumes a data loader of a predefined format.

::

    import torch
    from torch.utils.data import DataLoader
    from piq import FID

    first_dl, second_dl = DataLoader(), DataLoader()
    fid_metric = FID()
    first_feats = fid_metric.compute_feats(first_dl)
    second_feats = fid_metric.compute_feats(second_dl)
    fid: torch.Tensor = fid_metric(first_feats, second_feats)


If you already have image features, use the class interface for score computation:

::

    import torch
    from piq import FID

    prediction_feats = torch.rand(10000, 1024)
    target_feats = torch.rand(10000, 1024)
    msid_metric = MSID()
    msid: torch.Tensor = msid_metric(prediction_feats, target_feats)


For a full list of examples, see `feature metrics <https://github.com/photosynthesis-team/piq/blob/master/examples/feature_metrics.py>`_ examples.
