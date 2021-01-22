import torch
import piq


@torch.no_grad()
def main():
    prediction_features = torch.rand(2000, 128)
    target_features = torch.rand(2000, 128)

    if torch.cuda.is_available():
        # Move to GPU to make computaions faster
        prediction_features = prediction_features.cuda()
        target_features = target_features.cuda()

    # Use FID class to compute FID score from image features, pre-extracted from some feature extractor network
    fid: torch.Tensor = piq.FID()(prediction_features, target_features)
    print(f"FID: {fid:0.4f}")

    # If image features are not available, extract them using compute_feats of FID class.
    # Please note that compute_feats consumes a data loader of predefined format.

    # Use GS class to compute Geometry Score from image features, pre-extracted from some feature extractor network.
    # Computation is heavily CPU dependent, adjust num_workers parameter according to your system configuration.
    gs: torch.Tensor = piq.GS(
        sample_size=64, num_iters=100, i_max=100, num_workers=4)(prediction_features, target_features)
    print(f"GS: {gs:0.4f}")

    # Use inception_score function to compute IS from image features, pre-extracted from some feature extractor network.
    # Note, that we follow recomendations from paper "A Note on the Inception Score"
    isc_mean, _ = piq.inception_score(prediction_features, num_splits=10)
    # To compute difference between IS for 2 sets of image features, use IS class.
    isc: torch.Tensor = piq.IS(distance='l1')(prediction_features, target_features)
    print(f"IS: {isc_mean:0.4f}, difference: {isc:0.4f}")

    # Use KID class to compute KID score from image features, pre-extracted from some feature extractor network:
    kid: torch.Tensor = piq.KID()(prediction_features, target_features)
    print(f"KID: {kid:0.4f}")

    # Use MSID class to compute MSID score from image features, pre-extracted from some feature extractor network:
    msid: torch.Tensor = piq.MSID()(prediction_features, target_features)
    print(f"MSID: {msid:0.4f}")


if __name__ == '__main__':
    main()
