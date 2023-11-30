=======
matvis
=======
.. image:: https://github.com/hera-team/ matvis/workflows/Tests/badge.svg
    :target: https://github.com/hera-team/ matvis
.. image:: https://badge.fury.io/py/vis-cpu.svg
    :target: https://badge.fury.io/py/vis-cpu
.. image:: https://codecov.io/gh/hera-team/ matvis/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/hera-team/ matvis
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


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

Limitations
-----------

* Currently no support for polarized sky models.
* Currently no way of taking advantage of baseline redundancy to speed up simulations.
* Diffuse sky models must be pixelised, which may not be the best basis-function for
  some sky models.


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
