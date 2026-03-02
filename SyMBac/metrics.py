"""
Image similarity metrics for comparing synthetic and real microscopy images.

These metrics are designed to capture distributional/statistical similarity
rather than pixel-level correspondence, since cells in synthetic and real
images are not in corresponding positions.
"""

import numpy as np
from scipy.stats import wasserstein_distance, skew, kurtosis, entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.exposure import rescale_intensity


def intensity_histogram_emd(real, synth, n_bins=256):
    """
    Earth Mover's Distance (Wasserstein-1) between intensity histograms.

    Captures the shape of the full intensity distribution, not just mean/var.
    Robust to different cell positions.

    Parameters
    ----------
    real : 2D numpy array
        Real microscopy image (float, 0-1 range).
    synth : 2D numpy array
        Synthetic image (float, 0-1 range).
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        Wasserstein-1 distance between the two histograms.
    """
    real_flat = real.ravel()
    synth_flat = synth.ravel()
    return wasserstein_distance(real_flat, synth_flat)


def _radial_average_psd(image):
    """Compute radially-averaged power spectral density of a 2D image."""
    f2d = np.fft.fftshift(np.fft.fft2(image))
    psd2d = np.abs(f2d) ** 2

    cy, cx = np.array(psd2d.shape) // 2
    y, x = np.ogrid[:psd2d.shape[0], :psd2d.shape[1]]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cy, cx)
    radial_sum = np.bincount(r.ravel(), weights=psd2d.ravel(), minlength=max_r + 1)[:max_r + 1]
    radial_count = np.bincount(r.ravel(), minlength=max_r + 1)[:max_r + 1]
    radial_count[radial_count == 0] = 1
    return radial_sum / radial_count


def power_spectral_density_error(real, synth):
    """
    MSE between radially-averaged log power spectra.

    Captures frequency content — texture grain, edge sharpness, PSF halo
    ringing. Translation-invariant by construction.

    Parameters
    ----------
    real : 2D numpy array
        Real microscopy image.
    synth : 2D numpy array
        Synthetic image (same shape as real).

    Returns
    -------
    float
        Mean squared error between log-PSD curves.
    """
    psd_real = _radial_average_psd(real)
    psd_synth = _radial_average_psd(synth)
    min_len = min(len(psd_real), len(psd_synth))
    psd_real = psd_real[:min_len]
    psd_synth = psd_synth[:min_len]
    # Use log scale to avoid high-frequency domination
    log_real = np.log1p(psd_real)
    log_synth = np.log1p(psd_synth)
    return np.mean((log_real - log_synth) ** 2)


def glcm_feature_error(real, synth, distances=(1, 3), angles=(0, np.pi / 4, np.pi / 2)):
    """
    MSE between GLCM texture feature vectors.

    Computes Haralick texture features (contrast, homogeneity, energy,
    correlation) — position-invariant texture statistics.

    Parameters
    ----------
    real : 2D numpy array
        Real microscopy image (float, 0-1 range).
    synth : 2D numpy array
        Synthetic image (float, 0-1 range).
    distances : tuple of int
        Pixel distances for GLCM computation.
    angles : tuple of float
        Angles in radians for GLCM computation.

    Returns
    -------
    float
        Mean squared error between GLCM feature vectors.
    """
    props = ["contrast", "homogeneity", "energy", "correlation"]

    def _extract_glcm_features(image):
        img_uint8 = (rescale_intensity(image, out_range=(0, 1)) * 255).astype(np.uint8)
        glcm = graycomatrix(img_uint8, distances=list(distances), angles=list(angles), levels=256, symmetric=True,
                            normed=True)
        features = []
        for prop in props:
            features.append(graycoprops(glcm, prop).mean())
        return np.array(features)

    feat_real = _extract_glcm_features(real)
    feat_synth = _extract_glcm_features(synth)
    # Normalize by real features to make the error scale-invariant
    feat_real_safe = np.where(np.abs(feat_real) < 1e-10, 1e-10, feat_real)
    return np.mean(((feat_real - feat_synth) / feat_real_safe) ** 2)


def contrast_ratio_error(real_regions, synth_regions):
    """
    Error in inter-region contrast ratios.

    Compares cell/media and device/media contrast ratios, which are invariant
    to global brightness shifts.

    Parameters
    ----------
    real_regions : dict
        {"media": float, "cell": float, "device": float} — mean intensities.
    synth_regions : dict
        Same format as real_regions.

    Returns
    -------
    float
        Sum of squared relative differences in contrast ratios.
    """
    def _ratios(regions):
        media = regions["media"]
        if abs(media) < 1e-10:
            media = 1e-10
        return np.array([regions["cell"] / media, regions["device"] / media])

    real_ratios = _ratios(real_regions)
    synth_ratios = _ratios(synth_regions)
    real_ratios_safe = np.where(np.abs(real_ratios) < 1e-10, 1e-10, real_ratios)
    return np.mean(((real_ratios - synth_ratios) / real_ratios_safe) ** 2)


def higher_order_moments_error(real, synth):
    """
    Error in higher-order statistical moments (skewness, kurtosis, entropy).

    Goes beyond mean/variance to capture distribution shape.

    Parameters
    ----------
    real : 2D numpy array
        Real microscopy image.
    synth : 2D numpy array
        Synthetic image.

    Returns
    -------
    float
        Sum of squared relative differences in skewness, kurtosis, and entropy.
    """
    real_flat = real.ravel()
    synth_flat = synth.ravel()

    real_moments = np.array([
        skew(real_flat),
        kurtosis(real_flat),
    ])
    synth_moments = np.array([
        skew(synth_flat),
        kurtosis(synth_flat),
    ])

    # Entropy on histogram
    real_hist, _ = np.histogram(real_flat, bins=256, density=True)
    synth_hist, _ = np.histogram(synth_flat, bins=256, density=True)
    real_hist = real_hist + 1e-10
    synth_hist = synth_hist + 1e-10
    entropy_diff = (entropy(real_hist) - entropy(synth_hist)) ** 2

    moment_diffs = np.sum((real_moments - synth_moments) ** 2)
    return moment_diffs + entropy_diff


def composite_loss(real_images, synth_images, weights=None, real_regions=None, synth_regions=None):
    """
    Weighted combination of all image similarity metrics.

    Computes metrics over multiple image pairs and averages for noise
    reduction.

    Parameters
    ----------
    real_images : list of 2D numpy arrays
        Real microscopy images.
    synth_images : list of 2D numpy arrays
        Synthetic images (one per real image, or can be fewer — they'll
        be compared pairwise against randomly selected real images).
    weights : dict, optional
        Weights for each metric. Keys: "emd", "psd", "glcm", "contrast",
        "moments". Defaults to equal weighting.
    real_regions : dict, optional
        {"media": float, "cell": float, "device": float} mean intensities
        from the real image(s). Required for contrast_ratio_error.
    synth_regions : dict, optional
        Same format, from the synthetic image(s). Required for
        contrast_ratio_error.

    Returns
    -------
    float
        Weighted composite loss value.
    """
    if weights is None:
        weights = {
            "emd": 1.0,
            "psd": 1.0,
            "glcm": 1.0,
            "contrast": 1.0,
            "moments": 1.0,
        }

    n_comparisons = min(len(real_images), len(synth_images))

    emd_scores = []
    psd_scores = []
    glcm_scores = []
    moments_scores = []

    for i in range(n_comparisons):
        real = real_images[i]
        synth = synth_images[i]
        # Ensure same shape by cropping to minimum
        min_h = min(real.shape[0], synth.shape[0])
        min_w = min(real.shape[1], synth.shape[1])
        real = real[:min_h, :min_w]
        synth = synth[:min_h, :min_w]

        if weights.get("emd", 0) > 0:
            emd_scores.append(intensity_histogram_emd(real, synth))
        if weights.get("psd", 0) > 0:
            psd_scores.append(power_spectral_density_error(real, synth))
        if weights.get("glcm", 0) > 0:
            glcm_scores.append(glcm_feature_error(real, synth))
        if weights.get("moments", 0) > 0:
            moments_scores.append(higher_order_moments_error(real, synth))

    loss = 0.0
    if emd_scores:
        loss += weights["emd"] * np.mean(emd_scores)
    if psd_scores:
        loss += weights["psd"] * np.mean(psd_scores)
    if glcm_scores:
        loss += weights["glcm"] * np.mean(glcm_scores)
    if moments_scores:
        loss += weights["moments"] * np.mean(moments_scores)

    if real_regions is not None and synth_regions is not None and weights.get("contrast", 0) > 0:
        loss += weights["contrast"] * contrast_ratio_error(real_regions, synth_regions)

    return loss
