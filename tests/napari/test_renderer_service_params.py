import pytest

from SyMBac.napari.services.renderer_service import RendererService


def test_normalize_psf_params_casts_integer_fields():
    params = {
        "radius": 50.0,
        "wavelength": 0.75,
        "NA": 1.2,
        "n": 1.3,
        "apo_sigma": 20.0,
        "mode": "phase contrast",
        "condenser": "Ph3",
        "resize_amount": 3.0,
        "pix_mic_conv": 0.065,
    }

    out = RendererService._normalize_psf_params(params)
    assert isinstance(out["radius"], int)
    assert out["radius"] == 50
    assert isinstance(out["resize_amount"], int)
    assert out["resize_amount"] == 3


def test_normalize_psf_params_rejects_non_positive_radius():
    params = {
        "radius": 0,
        "wavelength": 0.75,
        "NA": 1.2,
        "n": 1.3,
        "apo_sigma": 20.0,
        "mode": "phase contrast",
        "condenser": "Ph3",
        "resize_amount": 3,
        "pix_mic_conv": 0.065,
    }
    with pytest.raises(ValueError, match="radius"):
        RendererService._normalize_psf_params(params)


def test_normalize_camera_params_validation():
    out = RendererService._normalize_camera_params(
        {
            "baseline": 100,
            "sensitivity": 2.9,
            "dark_noise": 8,
        }
    )
    assert out == {"baseline": 100.0, "sensitivity": 2.9, "dark_noise": 8.0}

    with pytest.raises(ValueError, match="sensitivity"):
        RendererService._normalize_camera_params(
            {
                "baseline": 100,
                "sensitivity": 0,
                "dark_noise": 8,
            }
        )
