"""Plotting convenience functions to help in analyzing vis_cpu output."""
import numpy as np
import pylab as plt

from . import conversions


def _source_az_za_beam(
    lst, crd_eq, beam, ref_freq=100.0e6, latitude=-30.7215 * np.pi / 180.0
):
    """
    Calculate the Az, ZA, and beam values of a set of sources at a given LST.

    Parameters
    ----------
    lst : float
        Local sidereal time, in radians.
    crd_eq : array_like
        Per-source Cartesian coordinate array.
    beam : UVBeam object
        Beam object. Used to calculate the value of the beam ('ee' polarization)
        for each source.
    ref_freq : float, optional
        Reference frequency to evaluate the beam at, in Hz.
    latitude : float, optional
        The latitude of the center of the array, in radians. The default is the
        HERA latitude = -30.7215 * pi / 180.

    Returns
    -------
    az, za : array_like
        Azimuth and zenith angle of each source, in radians.
    A : array_like
        Value of the beam (E-field, not power, unless the beam object contains
        only the power beam) for each source.
    """
    # Get coordinate transforms as a function of LST
    eq2top = conversions.eci_to_enu_matrix(lst, latitude)

    # Get source az, za (note the azimuth convention used by UVBeam)
    tx, ty, tz = np.dot(eq2top, crd_eq)
    az, za = conversions.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")

    # Get beam values
    interp_beam = beam.interp(az, za, np.atleast_1d(ref_freq))[0]
    A_s = interp_beam[0, 0, 1, 0]  # (2, 1, 2, 1, Nptsrc)

    # Horizon cut
    A_s = np.where(tz > 0, A_s, np.nan)

    return az, za, A_s


def animate_source_map(
    ra,
    dec,
    lsts,
    beam,
    interval=200,
    ref_freq=100.0e6,
    latitude=-30.7215 * np.pi / 180.0,
):
    """
    Create an animated map of sources as a function of LST, Az, and ZA.

    The sources are colored by the beam value. Note that the ``IPython``
    package is required by this function.

    NOTE: If you get an error about the ``ffmpeg`` encoder not being installed,
    you may need to change the path setting in ``matplotlib``:
    ``plt.rcParams['animation.ffmpeg_path'] = '/path/to/ffmpeg'``.

    Parameters
    ----------
    ra, dec : array_like
        RA and Dec coordinates of sources, in radians.
    lsts : array_like
        Array of LSTs to plot, in radians.
    beam : UVBeam object
        Beam object, used to color the point sources.
    interval : int, optional
        Interval between frames, in ms.
    ref_freq : float, optional
        Reference frequency to evaluate the beam at, in Hz.
    latitude : float, optional
        The latitude of the center of the array, in radians. The default is the
        HERA latitude = -30.7215 * pi / 180.

    Returns
    -------
    anim : matplotlib HTML animation
        Animation object HTML, for display in a Jupyter notebook.
    """
    from IPython.display import HTML
    from matplotlib import animation, rc

    # Point source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)

    # Calculate source positions for all LSTs
    y = np.array(
        [
            _source_az_za_beam(
                lst, crd_eq, beam=beam, ref_freq=ref_freq, latitude=latitude
            )
            for lst in lsts
        ]
    )
    all_az = y[:, 0, :]
    all_za = y[:, 1, :] * 180.0 / np.pi
    all_As = y[:, 2, :]

    # Plot initial positions
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    scatter = ax.scatter(all_az[0], all_za[0], c=all_As[0], s=3.0, cmap="cool")
    ax.set_xlabel("az")
    ax.set_ylabel("za")
    fig.set_size_inches((8.0, 8.0))

    def animate(i):
        # Set scatter plot values/colours
        scatter.set_offsets(np.column_stack([all_az[i], all_za[i]]))  # x,y
        scatter.set_array(all_As[i])  # colour

        ax.set_title("LST = %4.4f" % (lsts[i]))
        return (scatter,)

    def init():
        # Set scatter plot values/colours
        scatter.set_offsets(np.column_stack([all_az[0], all_za[0]]))  # x,y
        scatter.set_array(all_As[0])  # colour

        ax.set_title("LST = %4.4f" % (lsts[0]))
        return (scatter,)

    # Make animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=lsts.size, interval=interval, blit=True
    )
    return HTML(anim.to_html5_video())
