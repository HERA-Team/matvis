
import numpy as np
import pytest
from astropy.units import sday
from pyuvsim.analyticbeam import AnalyticBeam

from vis_cpu import vis_cpu, conversions

np.random.seed(0)
NTIMES = 10
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
    print("****", I_sky.shape)
    
    # Point source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)
    print(">>>>", crd_eq.shape)
    
    # Get coordinate transforms as a function of LST
    lsts = np.linspace(0., 2.*np.pi, NTIMES)
    eq2tops = conversions.get_eq2tops(lsts, latitude=-30.7215*np.pi/180.) # HERA latitude
    print(">>>>", eq2tops.shape)
    
    # Create beam models
    beam = AnalyticBeam(type='gaussian', diameter=14.)
    beam_pix = conversions.uvbeam_to_lm(beam, freq, n_pix_lm=63, 
                                        polarized=False)
    beam_cube = np.array([beam_pix, beam_pix+1e-17])
    i = 0
    print("xxxx", beam_cube.shape, beam_cube[:,i,:,:].shape)
    
    # Run vis_cpu with pixel beams
    for i in range(freq.size):
        _vis = vis_cpu(antpos, 
                       freq[i], 
                       eq2tops,
                       crd_eq,
                       I_sky[:,i],
                       bm_cube=beam_cube[:,i,:,:], 
                       precision=1,
                       polarized=False
                      )
    
        assert np.all(~np.isnan(_vis)) # check that there are no NaN values
