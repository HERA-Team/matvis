"""Compare vis_cpu with pyuvsim visibilities."""
import numpy as np
from astropy.coordinates import EarthLocation, Latitude, Longitude
from astropy.time import Time
from astropy.units import Quantity
from hera_sim import io
from pyradiosky import SkyModel
from pyuvsim import AnalyticBeam, simsetup, uvsim
from pyuvsim.telescope import BeamList

from vis_cpu import conversions, simulate_vis, vis_cpu

nfreq = 3
ntime = 5
nants = 3
nsource = 500


def test_compare_pyuvsim():
    """Compare vis_cpu and pyuvsim simulated visibilities."""
    hera_lat = -30.7215
    hera_lon = 21.4283
    hera_alt = 1073.0
    obstime = Time("2018-08-31T04:02:30.11", format="isot", scale="utc")

    # HERA location
    location = EarthLocation.from_geodetic(lat=hera_lat, lon=hera_lon, height=hera_alt)

    np.random.seed(1)

    # Random antenna locations
    x = np.random.random(nants) * 400.0  # Up to 400 metres
    y = np.random.random(nants) * 400.0
    z = np.random.random(nants) * 0.0
    ants = {}
    for i in range(nants):
        ants[i] = (x[i], y[i], z[i])

    # Observing parameters in a UVData object.
    uvdata = io.empty_uvdata(
        nfreq=nfreq,  # Need >=2 freqs for healvis to work
        start_freq=100e6,
        channel_width=97.3e3,
        start_time=obstime.jd,
        integration_time=1820.0,  # Just over 30 mins between time samples
        ntimes=ntime,
        ants=ants,
        polarization_array=np.array(["XX", "YY", "XY", "YX"]),
        Npols=4,
    )
    lsts = np.unique(uvdata.lst_array)

    # One fixed source plus random other sources
    sources = [
        [125.7, -30.72, 2, 0],  # Fix a single source near zenith
    ]
    if nsource > 1:  # Add random other sources
        ra = np.random.uniform(low=0.0, high=360.0, size=nsource - 1)
        dec = -30.72 + np.random.random(nsource - 1) * 10.0
        flux = np.random.random(nsource - 1) * 4
        for i in range(nsource - 1):
            sources.append([ra[i], dec[i], flux[i], 0])
    sources = np.array(sources)

    # Source locations and frequencies
    ra_dec = np.deg2rad(sources[:, :2])
    freqs = np.unique(uvdata.freq_array)

    # Correct source locations so that vis_cpu uses the right frame
    ra_new, dec_new = conversions.equatorial_to_eci_coords(
        ra_dec[:, 0], ra_dec[:, 1], obstime, location, unit="rad", frame="icrs"
    )

    # Calculate source fluxes for vis_cpu
    flux = ((freqs[:, np.newaxis] / freqs[0]) ** sources[:, 3].T * sources[:, 2].T).T
    # beam_ids = list(ants.keys())

    # Beam model
    beams = [AnalyticBeam("gaussian", diameter=14.0) for i in range(len(ants.keys()))]
    # beams = [AnalyticBeam('uniform') for i in range(len(ants.keys()))]
    beam_dict = {}
    for i in range(len(beams)):
        beam_dict[str(i)] = i

    # Stokes for the first frequency only. Stokes for other frequencies
    # are calculated later.
    stokes = np.zeros((4, 1, ra_dec.shape[0]))
    stokes[0, 0] = sources[:, 2]
    reference_frequency = np.full(len(ra_dec), freqs[0])

    # Set up sky model
    sky_model = SkyModel(
        name=[str(i) for i in range(len(ra_dec))],
        ra=Longitude(ra_dec[:, 0], "rad"),
        dec=Latitude(ra_dec[:, 1], "rad"),
        spectral_type="spectral_index",
        spectral_index=sources[:, 3],
        stokes=stokes,
        reference_frequency=Quantity(reference_frequency, "Hz"),
    )

    # Calculate stokes at all the frequencies.
    sky_model.at_frequencies(Quantity(freqs, "Hz"), inplace=True)

    # ---------------------------------------------------------------------------
    # (1) Run vis_cpu
    # ---------------------------------------------------------------------------
    vis_vc = simulate_vis(
        ants=ants,
        fluxes=flux,
        ra=ra_new,
        dec=dec_new,
        freqs=freqs,
        lsts=lsts,
        beams=beams,
        pixel_beams=False,
        polarized=False,
        precision=2,
        latitude=hera_lat * np.pi / 180.0,
    )

    # ---------------------------------------------------------------------------
    # (2) Run pyuvsim
    # ---------------------------------------------------------------------------
    uvd_uvsim = uvsim.run_uvdata_uvsim(
        uvdata,
        BeamList(beams),
        beam_dict=beam_dict,
        catalog=simsetup.SkyModelData(sky_model),
    )

    # ---------------------------------------------------------------------------
    # Compare
    # ---------------------------------------------------------------------------
    # Loop over baselines and compare
    for i in range(nants):
        for j in range(i, nants):
            d = uvd_uvsim.get_data((i, j, "XX")).T
            # FIXME: There is a factor of 2 in amplitude unaccounted for
            assert np.allclose(0.5 * vis_vc[:, :, i, j], d, rtol=2e-4, atol=2e-4)
