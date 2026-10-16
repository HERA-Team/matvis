"""Compare matvis with pyuvsim visibilities."""

import numpy as np
import pytest
from pyuvdata import UVData
from pyuvsim import simsetup, uvsim
from pyuvsim.telescope import BeamList

from matvis import matvis_to_uvdata, simulate_vis
from matvis._test_utils import get_standard_sim_params, nants, perturbed_beam
from matvis.redundancy import antpairs_to_blocks

# These fixture metadata warnings do not affect the Jones or visibility values.
pytestmark = [
    pytest.mark.filterwarnings("ignore:The mount_type parameter must be set"),
    pytest.mark.filterwarnings("ignore:No beam information:UserWarning"),
    pytest.mark.filterwarnings("ignore:mount_type, antenna_diameters:UserWarning"),
    pytest.mark.filterwarnings(
        "ignore:The default baseline conjugation convention has changed:UserWarning"
    ),
]


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


@pytest.mark.parametrize(
    "use_analytic_beam,xfail",
    [
        (True, False),
        pytest.param(
            False,
            True,
            marks=pytest.mark.xfail(
                reason=(
                    "pyuvsim swaps the Jones sky components and pyradiosky uses "
                    "C01=(U-iV)/2, whereas matvis uses native components and "
                    "C01=(U+iV)/2. Independently verified by the direct-oracle "
                    "and convention-converted comparison tests in this module."
                ),
                strict=True,
            ),
        ),
    ],
    ids=["analytic_beam", "uvbeam"],
)
def test_compare_pyuvsim_polarized_sky(use_analytic_beam, xfail):
    """Compare the matvis stokes path with pyuvsim for a fully polarized sky (Q,U,V≠0).

    The existing ``test_compare_pyuvsim`` only exercises Stokes I; this one
    drives non-trivial coherency rotation and XY/YX off-diagonals through
    the eigendecomposition path.
    """
    kw, sky_model, uvbeams, bmdict, uvdata = get_standard_sim_params(
        use_analytic_beam,
        polarized=True,
        use_polarized_sky=True,
        nsource=250,
    )

    vis_matvis = simulate_vis(precision=2, **kw)
    uvd_uvsim = uvsim.run_uvdata_uvsim(
        uvdata,
        uvbeams,
        beam_dict=bmdict,
        catalog=simsetup.SkyModelData(sky_model),
    )

    rtol = 2e-4 if use_analytic_beam else 0.01
    compare_sims(uvd_uvsim, matvis_uvdata(vis_matvis, kw, uvd_uvsim), rtol=rtol)


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


@pytest.mark.parametrize("swap_components", [False, True])
def test_polarized_uvbeam_direct_oracle(swap_components: bool):
    """Native and swapped Jones components each reproduce a direct local-basis RIME.

    Pyradiosky supplies the independent sky rotation. Neither this oracle nor
    pyuvsim's Jones construction uses matvis's decomposition or Z calculation.
    """
    from astropy.constants import c
    from pyuvsim.antenna import Antenna
    from pyuvsim.telescope import Telescope
    from pyuvsim.utils import altaz_to_zenithangle_azimuth

    kw, sky, beams, _, _ = get_standard_sim_params(
        False, True, nsource=15, ntime=2, use_polarized_sky=True
    )
    beams = BeamList(beams.beam_list, spline_interp_opts={"order": 3, "mode": "mirror"})
    telescope = Telescope("HERA", kw["telescope_loc"], beams)
    antenna = Antenna("0", 0, np.zeros(3), 0)
    expected = np.zeros((1, 2, 9, 2, 2), dtype=complex)
    positions = np.array(list(kw["ants"].values()))
    for ti, time in enumerate(kw["times"]):
        sky.update_positions(time, kw["telescope_loc"])
        alt, az = sky.alt_az[:, sky.above_horizon]
        za_beam, az_beam = altaz_to_zenithangle_azimuth(alt, az)
        response = (
            beams[0]
            .compute_response(
                az_array=az_beam,
                za_array=za_beam,
                freq_array=kw["freqs"],
                interpolation_function="az_za_map_coordinates",
                spline_opts={"order": 3, "mode": "mirror"},
            )[:, :, 0]
            .transpose(1, 0, 2)
        )
        external = antenna.get_beam_jones(
            telescope,
            np.array([alt, az]),
            kw["freqs"][0],
            interpolation_function="az_za_map_coordinates",
        )
        # Demonstrate the exact external convention difference, independently
        # of visibilities. Swapping sky axes changes both Q and V signs in C.
        np.testing.assert_allclose(external, response[:, ::-1], rtol=0, atol=1e-12)
        assert not np.allclose(external, response, rtol=1e-4, atol=1e-4)
        if swap_components:
            response = external
        # Pyradiosky uses C01=(U-iV)/2; this PR explicitly uses (U+iV)/2.
        # Its real basis rotation commutes with complex conjugation.
        local_c = sky.coherency_calc().to_value("Jy")[:, :, 0].conj()
        direction = np.array(
            [np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)]
        )
        phase = np.exp(2j * np.pi * kw["freqs"][0] / c.value * (positions @ direction))
        for i in range(3):
            for j in range(3):
                for s in range(len(alt)):
                    expected[0, ti, i * 3 + j] += (
                        response[:, :, s]
                        @ local_c[:, :, s]
                        @ response[:, :, s].conj().T
                        * phase[i, s].conj()
                        * phase[j, s]
                    )
    if swap_components:
        beam = kw["beams"][0].beam.copy()
        beam.data_array = beam.data_array[::-1].copy()
        kw["beams"] = [beam]
    actual = simulate_vis(**kw, precision=2, coord_method="CoordinateRotationAstropy")
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_compare_pyuvsim_polarized_converted_conventions():
    """Both independently demonstrated convention differences explain the mismatch."""
    kw, sky, beams, bmdict, uvdata = get_standard_sim_params(
        False, True, nsource=250, use_polarized_sky=True
    )
    beam = kw["beams"][0].beam.copy()
    beam.data_array = beam.data_array[::-1].copy()
    kw["beams"] = [beam]
    actual = simulate_vis(**kw, precision=2)
    # Convert the Stokes-V convention as well as the Jones sky-component order.
    sky.stokes[3] *= -1
    reference = uvsim.run_uvdata_uvsim(
        uvdata, beams, beam_dict=bmdict, catalog=simsetup.SkyModelData(sky)
    )
    compare_sims(
        reference,
        matvis_uvdata(actual, kw, reference),
        rtol=0,
        atol=1e-3 * np.abs(reference.data_array).max(),
    )
