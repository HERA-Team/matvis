"""CPU-based implementation of the visibility simulator."""

import numpy as np
from astropy.constants import c
from scipy.interpolate import RectBivariateSpline
from typing import Optional, Sequence
from . import conversions

        


def vis_cpu(
    antpos: np.ndarray,
    freq: float,
    I_sky: np.ndarray,
    ntimes : int,
    npix : int,
    beam_list: Sequence[np.ndarray],
    az_za_transforms : conversions.AzZaTransforms
):
    """
    Calculate visibility from an input intensity map and beam model.

    Provided as a standalone function.

    Parameters
    ----------
    antpos : array_like
        Antenna position array. Shape=(NANT, 3).
    freq : float
        Frequency to evaluate the visibilities at [GHz].
    eq2tops : array_like
        Set of 3x3 transformation matrices converting equatorial
        coordinates to topocentric at each
        hour angle (and declination) in the dataset.
        Shape=(NTIMES, 3, 3).
    crd_eq : array_like
        Equatorial coordinates of Healpix pixels, in Cartesian system.
        Shape=(3, NPIX).
    I_sky : array_like
        Intensity distribution on the sky,
        stored as array of Healpix pixels. Shape=(NPIX,).
    bm_cube : array_like, optional
        Pixelized beam maps for each antenna. Shape=(NANT, BM_PIX, BM_PIX).
    beam_list : list of UVBeam, optional
        If specified, evaluate primary beam values directly using UVBeam
        objects instead of using pixelized beam maps (`bm_cube` will be ignored
        if `beam_list` is not None).
    precision : int, optional
        Which precision level to use for floats and complex numbers.
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128

    Returns
    -------
    array_like
        Visibilities. Shape=(NTIMES, NANTS, NANTS).
    """

    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)."
    assert I_sky.ndim == 1 and I_sky.shape[0] == npix, "I_sky must have shape (NPIX,)."
    assert len(beam_list) == nant, "beam_list must have length nant"

    # Intensity distribution (sqrt) and antenna positions. Does not support
    # negative sky.
    Isqrt = np.sqrt(I_sky)

    ang_freq = 2 * np.pi * freq

    # Empty arrays: beam pattern, visibilities, delays, complex voltages.
    A_s = np.empty((nant, npix))
    vis = np.empty((ntimes, nant, nant), dtype=np.complex128)
    tau = np.empty((nant, npix))
    v = np.empty((nant, npix), dtype=np.complex128)

    # Loop over time samples
    for t in range(ntimes):
        # Primary beam pattern using direct interpolation of UVBeam object
        az, za, crd_top = az_za_transforms.transform(None, None, t)
        for i in range(nant):
            interp_beam = beam_list[i].interp(az, za, np.atleast_1d(freq))[0]
            A_s[i] = interp_beam[0, 0, 1]  # FIXME: assumes xx pol for now

        A_s = np.where(crd_top[2] > 0, A_s, 0)

        # Calculate delays, where tau = (b * s) / c
        np.dot(antpos, crd_top, out=tau)
        tau /= c.value

        # Component of complex phase factor for one antenna
        # (actually, b = (antpos1 - antpos2) * crd_top / c; need dot product
        # below to build full phase factor for a given baseline)
        np.exp(1.0j * (ang_freq * tau), out=v)

        # Complex voltages.
        v *= A_s * Isqrt

        # Compute visibilities using product of complex voltages (upper triangle).
        for i in range(len(antpos)):
            np.dot(v[i : i + 1].conj(), v[i:].T, out=vis[t, i : i + 1, i:])

    return vis


class VisCPU:
    def __init__(self, uvdata, beams, sky_freqs, point_source_pos, point_source_flux, **kwargs):
        self.beams = beams
        self.npix = point_source_pos.shape[0]
        self.times = np.unique(uvdata.get_times("XX"))
        self.point_source_flux = point_source_flux/2.0
        self.sky_freqs = sky_freqs
        self.uvdata = uvdata
        self.az_za_transforms = conversions.AzZaTransforms(
                            obstimes=self.times,
                            ra=point_source_pos[:, 0],
                            dec=point_source_pos[:, 1],
                            precompute=True,
                            use_central_time_values=False,
                            uvbeam_az_correction=True,
                            astropy=True
                        )
        # Get antpos for active antennas only
        #self.antpos = self.uvdata.get_ENU_antpos()[0].astype(self._real_dtype)
        ant_list = uvdata.get_ants() # ordered list of active ants
        self.antpos = []
        _antpos = uvdata.get_ENU_antpos()[0]
        for ant in ant_list:
            # uvdata.get_ENU_antpos() and uvdata.antenna_numbers have entries 
            # for all telescope antennas, even ones that aren't included in the 
            # data_array. This extracts only the data antennas.
            idx = np.where(ant == uvdata.antenna_numbers)
            self.antpos.append(_antpos[idx].flatten())
        self.antpos = np.array(self.antpos)
        
        assert point_source_flux.shape[0] == sky_freqs.size
        assert point_source_flux.shape[1] == self.npix
        
    def simulate(self):
        visfull = np.zeros_like(self.uvdata.data_array, dtype=np.complex128)
        for i in range(self.sky_freqs.size):
            vis = vis_cpu(self.antpos, self.sky_freqs[i], self.point_source_flux[i], 
                    len(self.times), self.npix, self.beams, self.az_za_transforms)
            indices = np.triu_indices(vis.shape[1])
            vis_upper_tri = vis[:, indices[0], indices[1]]

            visfull[:, 0, i, 0] = vis_upper_tri.flatten()
            
        self.uvdata.data_array += visfull
            
        
            


