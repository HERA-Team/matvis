"""Compare matvis with pyuvsim visibilities."""

import numpy as np
import pytest
from pyuvdata import UVData
from pyuvsim import simsetup, uvsim
from pyuvsim.telescope import BeamList

from matvis import matvis_to_uvdata, simulate_vis
from matvis._test_utils import get_standard_sim_params, nants, perturbed_beam
from matvis.redundancy import antpairs_to_blocks


@pytest.fixture(scope="function")
def default_uvsim() -> UVData:
    """Pyuvsim output for interpolated polarized beam."""
    _, sky_model, beams, beam_dict, uvdata = get_standard_sim_params(
        use_analytic_beam=False, polarized=True, nsource=250
    )

    return uvsim.run_uvdata_uvsim(
        uvdata,
        beams,
        beam_dict=beam_dict,
        catalog=simsetup.SkyModelData(sky_model),
    )


@pytest.mark.parametrize(
    "use_analytic_beam", (True, False), ids=["analytic_beam", "uvbeam"]
)
@pytest.mark.parametrize("polarized", (True, False), ids=["polarized", "unpolarized"])
def test_compare_pyuvsim(polarized, use_analytic_beam):
    """Compare matvis and pyuvsim simulated visibilities."""
    print("Polarized=", polarized, "Analytic Beam =", use_analytic_beam)
    kw, sky_model, uvbeams, bmdict, uvdata = get_standard_sim_params(
        use_analytic_beam, polarized, nsource=250
    )

    vis_matvis = simulate_vis(precision=2, **kw)
    uvd_uvsim = uvsim.run_uvdata_uvsim(
        uvdata,
        uvbeams,
        beam_dict=bmdict,
        catalog=simsetup.SkyModelData(sky_model),
    )

    # ---------------------------------------------------------------------------
    # Compare
    # ---------------------------------------------------------------------------
    rtol = 2e-4 if use_analytic_beam else 0.01

    compare_sims(uvd_uvsim, matvis_uvdata(vis_matvis, kw, uvd_uvsim), rtol)


@pytest.mark.parametrize("min_chunks", (1, 2, 3))
@pytest.mark.parametrize("source_buffer", (1.0, 0.75))
def test_compare_pyuvsim_chunking(min_chunks, source_buffer, default_uvsim):
    """Test chunking and source buffer against pyuvsim."""
    kw, *_ = get_standard_sim_params(
        use_analytic_beam=False, polarized=True, nsource=250
    )

    vis_matvis = simulate_vis(
        precision=2, min_chunks=min_chunks, source_buffer=source_buffer, **kw
    )

    compare_sims(default_uvsim, matvis_uvdata(vis_matvis, kw, default_uvsim), rtol=0.01)


@pytest.mark.parametrize("perturbation", ["feed_phase", "rotated_feed"])
@pytest.mark.parametrize("matprod_method", ["CPUMatMul", "CPUVectorDot", "CPUMatBlock"])
def test_compare_pyuvsim_per_antenna_beams(perturbation: str, matprod_method: str):
    """Antennas with different beams match pyuvsim for every pair, in both orders.

    The GPU methods are checked against these CPU results in test_cpu_vs_gpu.py.
    """
    kw, sky_model, uvbeams, _, uvdata = get_standard_sim_params(
        use_analytic_beam=False, polarized=True
    )
    beam0 = uvbeams.beam_list[0]
    beam1 = beam0.clone(beam=perturbed_beam(beam0.beam, perturbation))
    beam_idx = np.arange(nants) % 2

    uvd_uvsim = uvsim.run_uvdata_uvsim(
        uvdata,
        BeamList([beam0, beam1]),
        beam_dict={str(ant): int(bidx) for ant, bidx in enumerate(beam_idx)},
        catalog=simsetup.SkyModelData(sky_model),
    )

    extra = {}
    if matprod_method.endswith("MatBlock"):
        # Blocks hold only i <= j, so the reversed pairs come from the
        # Hermitian-conjugate path.
        extra["antenna_blocks"] = antpairs_to_blocks(
            [(i, j) for i in range(nants) for j in range(i, nants)]
        )
    kw["beams"] = [beam0, beam1]
    vis_matvis = simulate_vis(
        precision=2,
        beam_idx=beam_idx,
        matprod_method=matprod_method,
        **extra,
        **kw,
    )

    # Interpolation and coordinate differences between the codes are ~1e-4 of
    # the peak; using the wrong antenna's beam conjugate is a ~10-100% error.
    compare_sims(
        uvd_uvsim,
        matvis_uvdata(vis_matvis, kw, uvd_uvsim),
        rtol=0,
        atol=1e-3 * np.abs(uvd_uvsim.data_array).max(),
    )


def test_perturbed_beam_rejects_unknown_perturbation(uvbeam):
    """Only the documented perturbations are accepted."""
    with pytest.raises(ValueError, match="unknown perturbation"):
        perturbed_beam(uvbeam, "not_a_perturbation")


def matvis_uvdata(vis: np.ndarray, kw: dict, uvd_uvsim: UVData) -> UVData:
    """Wrap matvis output in a UVData object, from the inputs that produced it.

    The channel width is metadata only; it is copied from the pyuvsim object
    because it cannot be inferred from a single frequency.
    """
    return matvis_to_uvdata(
        vis,
        channel_width=uvd_uvsim.channel_width,
        **{
            key: kw[key]
            for key in ("ants", "freqs", "times", "telescope_loc", "beams", "polarized")
        },
    )


def compare_sims(
    uvd_uvsim: UVData, uvd_matvis: UVData, rtol: float, atol: float = 5e-4
):
    """Compare every pyuvsim baseline, in both orders, for every matvis polarization.

    pyuvsim stores one order of each pair, and get_data returns the other order
    as its conjugate with the polarization swapped. The matvis UVData holds both
    orders, so it is compared one order at a time (get_data on a pair held in
    both orders returns both). Cross-polarizations are compared with rtol * 100.
    """
    forward = uvd_uvsim.get_antpairs()
    for antpairs in (forward, [(ant2, ant1) for ant1, ant2 in forward]):
        uvd = uvd_matvis.select(bls=antpairs, inplace=False)
        for ant1, ant2 in antpairs:
            for pol in uvd.get_pols():
                d_uvsim = uvd_uvsim.get_data((ant1, ant2, pol))
                d_matvis = uvd.get_data((ant1, ant2, pol))
                tol = rtol if pol[0] == pol[1] else rtol * 100
                err = (
                    f"baseline ({ant1}, {ant2}, {pol}): "
                    f"max |uvsim - matvis| = {np.abs(d_uvsim - d_matvis).max():.3e}, "
                    f"max |uvsim| = {np.abs(d_uvsim).max():.3e}"
                )
                for part in (np.real, np.imag):
                    np.testing.assert_allclose(
                        part(d_uvsim), part(d_matvis), rtol=tol, atol=atol, err_msg=err
                    )
