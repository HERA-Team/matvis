=======
matvis
=======
.. image:: https://github.com/hera-team/ matvis/workflows/Tests/badge.svg
    :target: https://github.com/hera-team/ matvis
.. image:: https://badge.fury.io/py/vis-cpu.svg
    :target: https://badge.fury.io/py/vis-cpu
.. image:: https://codecov.io/gh/hera-team/ matvis/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/hera-team/ matvis
.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff


Fast matrix-based visibility simulator capable of running on CPU and GPU.


Description
===========

``matvis`` is a fast Python matrix-based interferometric visibility simulator with both
CPU and GPU implementations.

It is applicable to wide field-of-view instruments such as the Hydrogen Epoch of
Reionization Array (HERA) and the Square Kilometre Array (SKA), as it does not make
any approximations of the visibility integral (such as the flat-sky approximation).
The only approximation made is that the sky is a collection of point sources, which
is valid for sky models that intrinsically consist of point-sources, but is an
approximation for diffuse sky models.

An example wrapper for the main ``matvis`` simulator function is provided in this
package (``matvis.simulate_vis()``).

Features
--------

* Matrix-based algorithm is fast and scales well to large numbers of antennas.
* Supports both CPU and GPU implementations as drop-in replacements for each other.
* Supports both dense and sparse sky models.
* Includes a wrapper for simulating multiple frequencies and setting up the simulation.
* No approximations of the visibility integral (such as the flat-sky approximation).
* Arbitrary primary beams per-antenna using the ``pyuvdata.UVBeam`` class.
* Full-Stokes skies, including signed coherency models, on CPU and GPU.
* Dense, selected-pair, and redundant block matrix products.

Limitations
-----------

* Diffuse sky models must be pixelised, which may not be the best basis-function for
  some sky models.


Full-Stokes sky input
=====================

For a polarized sky, pass ``stokes`` instead of ``fluxes``, with shape
``(4, Nsource, Nfreq)`` ordered I, Q, U, V::

    vis = matvis.simulate_vis(
        ants=ants, ra=ra, dec=dec, freqs=freqs, times=times,
        beams=efield_beams, telescope_loc=telescope_loc, stokes=stokes,
        precision=2,
    )

Exactly one of ``fluxes`` and ``stokes`` must be given. Passing ``stokes``
computes all feed pairs (``polarized=True``), so the beams must be E-field beams.
Sources whose coherency has negative eigenvalues, as in Epoch of Reionization
models with negative intensities, are supported; set
``raise_on_negative_flux=True`` to reject them instead. Polarized skies work on
CPU and GPU and with every ``matprod_method``.


Installation
============
``pip install matvis``.

If you want to use the GPU functions, install
with ``pip install matvis[gpu]``.

Developers
==========
Run ``pre-commit install`` before working on this code.

Read the Docs
=============
https://matvis.readthedocs.io/en/latest/
