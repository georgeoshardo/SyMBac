import numpy as np
import pytest

from SyMBac import PSF
from SyMBac.PSF import PSF_generator


def test_3d_psf_defaults_working_distance_before_calling_psfmodels(monkeypatch):
    received = {}

    def make_psf(**kwargs):
        received.update(kwargs)
        return np.ones((kwargs["z"], kwargs["nx"], kwargs["nx"]))

    monkeypatch.setattr(PSF.psfm, "make_psf", make_psf)
    generator = PSF_generator(
        radius=1,
        wavelength=0.6,
        NA=1.2,
        n=1.3,
        apo_sigma=1,
        mode="3d fluo",
        z_height=3,
        scale=0.1,
    )

    generator.calculate_PSF()

    assert received["ti0"] == 150.0


@pytest.mark.parametrize("working_distance", [0, -1, np.nan])
def test_3d_psf_rejects_invalid_working_distance(working_distance):
    with pytest.raises(ValueError, match="working_distance"):
        PSF_generator(
            radius=1,
            wavelength=0.6,
            NA=1.2,
            n=1.3,
            apo_sigma=1,
            mode="3d fluo",
            z_height=3,
            scale=0.1,
            working_distance=working_distance,
        )
