"""Tests."""

import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, Latitude, Longitude
from astropy.time import Time
from astropy.units import Quantity
from dataclasses import replace
from pathlib import Path
from pyradiosky import SkyModel
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import GaussianBeam
from pyuvdata.beam_interface import BeamInterface
from pyuvdata.telescopes import get_telescope
from pyuvsim import simsetup
from pyuvsim.telescope import BeamList

from matvis import DATA_PATH

nfreq = 1
ntime = 5  # 20
nants = 3  # 4
nsource = 15  # 250
beam_file = DATA_PATH / "NF_HERA_Dipole_small.fits"


def get_standard_sim_params(
    use_analytic_beam: bool,
    polarized: bool,
    nants=nants,
    nfreq=nfreq,
    ntime=ntime,
    nsource=nsource,
    first_source_antizenith=False,
):
    """Create some standard random simulation parameters for use in tests."""
    hera = get_telescope("hera")
    obstime = Time("2018-08-31T04:02:30.11", format="isot", scale="utc")

    rng = np.random.default_rng(1)

    # Beam model
    if use_analytic_beam:
        n_freq = nfreq
        beam = BeamInterface(
            GaussianBeam(diameter=14.0), beam_type="efield" if polarized else "power"
        )
    else:
        n_freq = min(nfreq, 2)
        # This is a peak-normalized e-field beam file at 100 and 101 MHz,
        # downsampled to roughly 4 square-degree resolution.
        beam = UVBeam()
        beam.read_beamfits(beam_file)
        # Even though we sometimes want a power beam for matvis, we always need
        # an efield beam for pyuvsim, so we create an efield beam here, and let matvis
        # take care of conversion later.
        beam = BeamInterface(beam, beam_type="efield")

        # Now, the beam we have on file doesn't actually properly wrap around in azimuth.
        # This is fine -- the uvbeam.interp() handles the interpolation well. However, when
        # comparing to the GPU interpolation, which first has to interpolate to a regular
        # grid that ends right at 2pi, it's better to compare like for like, so we
        # interpolate to such a grid here.
        beam = beam.clone(
            beam=beam.beam.interp(
                az_array=np.linspace(0, 2 * np.pi, 181),
                za_array=np.linspace(0, np.pi / 2, 46),
                az_za_grid=True,
                new_object=True,
            )
        )

    # The UVSim beams must be efield all the time.
    uvsim_beams = BeamList([beam])

    beam_dict = {str(i): 0 for i in range(nants)}

    # Random antenna locations
    x = rng.uniform(size=nants) * 400.0  # Up to 400 metres
    y = rng.uniform(size=nants) * 400.0
    z = np.zeros(nants)
    ants = {i: (x[i], y[i], z[i]) for i in range(nants)}

    # Observing parameters in a UVData object
    uvdata = simsetup.initialize_uvdata_from_keywords(
        Nfreqs=n_freq,
        start_freq=100e6,
        channel_width=97.3e3,
        start_time=obstime.jd,
        integration_time=182.0,  # Just over 3 mins between time samples
        Ntimes=ntime,
        array_layout=ants,
        polarization_array=np.array(["XX", "YY", "XY", "YX"]),
        telescope_location=(
            float(hera.location.lat.deg),
            float(hera.location.lon.deg),
            float(hera.location.height.to_value("m")),
        ),
        telescope_name="HERA",
        phase_type="drift",
        vis_units="Jy",
        complete=True,
        write_files=False,
    )
    times = Time(np.unique(uvdata.time_array), format="jd")

    # One fixed source plus random other sources
    sources = [
        [
            300 if first_source_antizenith else 125.7,
            -30.72,
            2,
            0,
        ],  # Fix a single source near zenith
    ]
    if nsource > 1:  # Add random other sources
        ra = rng.uniform(low=0.0, high=360.0, size=nsource - 1)
        dec = -30.72 + rng.uniform(size=nsource - 1) * 10.0
        flux = rng.uniform(size=nsource - 1) * 4
        sources.extend([ra[i], dec[i], flux[i], 0] for i in range(nsource - 1))
    sources = np.array(sources)

    # Source locations and frequencies
    ra_dec = np.deg2rad(sources[:, :2])
    freqs = np.unique(uvdata.freq_array)

    # Calculate source fluxes for matvis
    flux = ((freqs[:, np.newaxis] / freqs[0]) ** sources[:, 3].T * sources[:, 2].T).T

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
        frame="icrs",
        spectral_type="spectral_index",
        spectral_index=sources[:, 3],
        stokes=stokes * un.Jy,
        reference_frequency=Quantity(reference_frequency, "Hz"),
    )

    # Calculate stokes at all the frequencies.
    sky_model.at_frequencies(Quantity(freqs, "Hz"), inplace=True)

    return (
        {
            "ants": ants,
            "fluxes": flux,
            "ra": sky_model.ra.rad,
            "dec": sky_model.dec.rad,
            "freqs": freqs,
            "times": times,
            "beams": [beam],
            "telescope_loc": uvdata.telescope.location,
            "polarized": polarized,
        },
        sky_model,
        uvsim_beams,
        beam_dict,
        uvdata,
    )
