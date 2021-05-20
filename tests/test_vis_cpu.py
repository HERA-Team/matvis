
import numpy as np
import pytest
from astropy.units import sday
from pyuvsim.analyticbeam import AnalyticBeam

from vis_cpu import vis_cpu, conversions

np.random.seed(0)
NTIMES = 10
BM_PIX = 31
NPIX = 12 * 16 ** 2
NFREQ = 5
NPTSRC = 20

ants={
        0: (0, 0, 0),
        1: (1, 1, 0)
    }


def test_vis_cpu():
    """
    Test basic operation of vis_cpu.
    """
    # Point source equatorial coords (radians)
    ra = np.linspace(0., np.pi, NPTSRC)
    dec = np.linspace(0., np.pi, NPTSRC)
    
    # Antenna x,y,z positions and frequency array
    antpos = np.array([ants[k] for k in ants.keys()])
    freq = np.linspace(100.e6, 120.e6, NFREQ) # Hz
    
    # SED for each point source
    fluxes = np.ones(NPTSRC)
    I_sky = fluxes[:,np.newaxis] * (freq[np.newaxis,:] / 100.e6)**-2.7
    
    # Point source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)
    
    # Get coordinate transforms as a function of LST
    lsts = np.linspace(0., 2.*np.pi, NTIMES)
    eq2tops = get_eq2tops(lsts, latitude=-30.7215*np.pi/180.) # HERA latitude
    
    # Create beam models
    beam = AnalyticBeam(type='gaussian', diameter=14.)
    beam_cube = conversions.uvbeam_to_lm(beam, freq, n_pix_lm=63, 
                                         polarized=False)
    
    # Run vis_cpu with pixel beams
    vis = vis_cpu(antpos, 
                  freq, 
                  eq2tops,
                  crd_eq,
                  I_sky,
                  bm_cube=beam_cube, 
                  precision=1,
                  polarized=False
                 )
    
    assert np.all(~np.isnan(vis)) # check that there are no NaN values
