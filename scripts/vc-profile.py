"""Profile the code with a simple scalable sky model.

Running the script will write a summary of the timings of various main blocks of code.
It will also save these results in pickle format to a file annotated with the inputs
(eg. nants, ntimes, nsources, nfreqs).
"""
import click
import inspect
import linecache
import numpy as np
import os
import pickle
from astropy.coordinates import EarthLocation
from astropy.time import Time
from line_profiler import LineProfiler
from pathlib import Path
from pyuvdata import UVBeam
from pyuvsim import AnalyticBeam, simsetup

from vis_cpu import HAVE_GPU, conversions, simulate_vis, vis_cpu, vis_gpu

beam_file = Path(__file__).parent.parent / "tests" / "data/NF_HERA_Dipole_small.fits"


profiler = LineProfiler()

main = click.Group()


@main.command()
@click.option("-A/-I", "--analytic-beam/--interpolated-beam", default=True)
@click.option("-f", "--nfreq", default=1)
@click.option(
    "-t",
    "--ntimes",
    default=1,
)
@click.option(
    "-a",
    "--nants",
    default=1,
)
@click.option(
    "-a",
    "--nbeams",
    default=1,
)
@click.option(
    "-s",
    "--nsource",
    default=1,
)
@click.option(
    "-g/-c",
    "--gpu/--cpu",
    default=False,
)
@click.option("-o", "--outdir", default=".")
@click.option("--double-precision/--single-precision", default=True)
def run(
    analytic_beam, nfreq, ntimes, nants, nbeams, nsource, gpu, double_precision, outdir
):
    """Run the script."""
    (
        ants,
        flux,
        ra,
        dec,
        freqs,
        lsts,
        cpu_beams,
        hera_lat,
        beam_idx,
    ) = get_standard_sim_params(analytic_beam, nfreq, ntimes, nants, nsource, nbeams)

    if gpu:
        profiler.add_function(vis_gpu)
    else:
        profiler.add_function(vis_cpu)

    profiler.runcall(
        simulate_vis,
        ants=ants,
        fluxes=flux,
        ra=ra,
        dec=dec,
        freqs=freqs,
        lsts=lsts,
        beams=cpu_beams,
        polarized=True,
        precision=2 if double_precision else 1,
        latitude=hera_lat * np.pi / 180.0,
        use_gpu=gpu,
        beam_idx=beam_idx,
    )

    outdir = Path(outdir).expanduser().absolute()

    str_id = f"A{analytic_beam}_nf{nfreq}_nt{ntimes}_na{nants}_ns{nsource}_nb{nbeams}_g{gpu}_pr{2 if double_precision else 1}"
    profiler.dump_stats(f"{outdir}/stats-{str_id}.pkl")

    with open(f"{outdir}/full-stats-{str_id}.txt", "w") as fl:
        profiler.print_stats(stream=fl, stripzeros=True)

    thing_stats = get_summary_stats(profiler, gpu)

    for thing, (hits, time, time_per_hit, percent, nlines) in thing_stats.items():
        print(
            f"{thing:>19}: {hits:>4} hits, {time:.2f} seconds, {time_per_hit:.2f} sec/hit, {percent:3.2f}%, {nlines} lines"
        )

    with open(f"{outdir}/summary-stats-{str_id}.pkl", "wb") as fl:
        pickle.dump(thing_stats, fl)


def get_summary_stats(profiler, gpu):
    """Convert a line-by-line set of stats into a summary of major components."""
    lstats = profiler.get_stats()

    (fn, lineno, name), timings = sorted(lstats.timings.items())[0]
    d, total_time = get_stats_and_lines(fn, lineno, timings)

    # specify contents of lines where important things happen
    vis_cpu_lines = {
        "eq2top": ("np.dot(eq2top",),
        "beam_interp": ("_evaluate_beam_cpu(",),
        "get_tau": ("np.dot(antpos",),
        "get_antenna_vis": ("ang_freq * tau", "v = A_s"),
        "get_baseline_vis": ("vis[t, :",),
    }

    vis_gpu_lines = {
        "eq2top": ("# compute crdtop",),
        "beam_interp": ("do_beam_interpolation(",),
        "get_tau": ("# compute tau",),
        "get_antenna_vis": ("meas_eq(",),
        "get_baseline_vis": ("vis_inner_product(",),
    }

    ids = vis_gpu_lines if gpu else vis_cpu_lines

    thing_stats = {"total": (1, total_time, total_time / 1, 100, len(d))}
    for thing, lines in ids.items():
        init_line = None
        assoc_lines = []
        for dd in d:
            if lines[0] in dd:
                init_line = dd
                assoc_lines.append(init_line)

            if len(lines) > 1 and init_line:
                assoc_lines.append(dd)

                if lines[1] in dd:
                    break
            elif len(lines) == 1 and init_line:
                break

        if not assoc_lines:
            raise RuntimeError(
                f"Could not find any lines for {thing} satisfying '{lines}'. Possible lines: {' | '.join(list(d.keys()))}"
            )

        # save (hits, time, time/hits, percent, nlines)
        thing_stats[thing] = (
            d[assoc_lines[0]][0],
            sum(d[ln][1] for ln in assoc_lines),
            sum(d[ln][2] for ln in assoc_lines),
            sum(d[ln][3] for ln in assoc_lines),
            len(assoc_lines),
        )

    return thing_stats


def get_stats_and_lines(filename, start_lineno, timings):
    """Match up timing stats with line content of the code."""
    d = {}
    total_time = 0.0
    linenos = []
    for lineno, nhits, time in timings:
        total_time += time
        linenos.append(lineno)

    if not os.path.exists(filename):
        raise ValueError(f"Could not find file: {filename}")

    linecache.clearcache()
    all_lines = linecache.getlines(filename)
    sublines = inspect.getblock(all_lines[start_lineno - 1 :])
    all_linenos = list(range(start_lineno, start_lineno + len(sublines)))

    for lineno, nhits, time in timings:
        percent = 100 * time / total_time
        idx = all_linenos.index(lineno)

        d[sublines[idx].rstrip("\n").rstrip("\r")] = (
            nhits,
            time / 1e6,
            float(time) / nhits / 1e6,
            percent,
            lineno,
        )

    return d, total_time / 1e6


def get_standard_sim_params(
    use_analytic_beam: bool, nfreq, ntime, nants, nsource, nbeams
):
    """Create some standard random simulation parameters for use in profiling.

    Will create a sky with uniformly distributed point sources (half below the horizon).

    """
    hera_lat = -30.7215
    hera_lon = 21.4283
    hera_alt = 1073.0
    obstime = Time("2018-08-31T04:02:30.11", format="isot", scale="utc")

    # HERA location
    location = EarthLocation.from_geodetic(lat=hera_lat, lon=hera_lon, height=hera_alt)

    # Set the seed so that different runs take about the same time.
    np.random.seed(1)

    # Beam model
    if use_analytic_beam:
        beam = AnalyticBeam("gaussian", diameter=14.0)
    else:
        # This is a peak-normalized e-field beam file at 100 and 101 MHz,
        # downsampled to roughly 4 square-degree resolution.
        beam = UVBeam()
        beam.read_beamfits(beam_file)

    beams = [beam] * nbeams

    # Random antenna locations
    x = np.random.random(nants) * 400.0  # Up to 400 metres
    y = np.random.random(nants) * 400.0
    z = np.random.random(nants) * 0.0
    ants = {i: (x[i], y[i], z[i]) for i in range(nants)}

    # This will make the beam_idx like [0,1,2,3,3,3,3,3,3,3] where nbeams=4 and the
    # array is nants long.
    beam_idx = np.array(list(range(nbeams)) + [nbeams - 1] * (nants - nbeams))

    # Observing parameters in a UVData object
    uvdata = simsetup.initialize_uvdata_from_keywords(
        Nfreqs=nfreq,
        start_freq=100e6,
        channel_width=97.3e3,
        start_time=obstime.jd,
        integration_time=182.0,  # Just over 3 mins between time samples
        Ntimes=ntime,
        array_layout=ants,
        polarization_array=np.array(["XX", "YY", "XY", "YX"]),
        telescope_location=(hera_lat, hera_lon, hera_alt),
        telescope_name="test_array",
        phase_type="drift",
        vis_units="Jy",
        complete=True,
        write_files=False,
    )
    lsts = np.unique(uvdata.lst_array)

    ra = np.random.uniform(low=0.0, high=2 * np.pi, size=nsource)
    dec = np.arccos(1 - 2 * np.random.uniform(size=nsource)) - np.pi / 2
    flux0 = np.random.random(nsource) * 4
    spec_indx = np.random.normal(0.8, scale=0.05, size=nsource)

    # Source locations and frequencies
    freqs = np.unique(uvdata.freq_array)

    # Correct source locations so that vis_cpu uses the right frame
    ra, dec = conversions.equatorial_to_eci_coords(
        ra, dec, obstime, location, unit="rad", frame="icrs"
    )

    # Calculate source fluxes for vis_cpu
    flux = ((freqs[:, np.newaxis] / freqs[0]) ** spec_indx.T * flux0.T).T

    return (
        ants,
        flux,
        ra,
        dec,
        freqs,
        lsts,
        beams,
        hera_lat,
        beam_idx,
    )


if __name__ == "__main__":
    run()
