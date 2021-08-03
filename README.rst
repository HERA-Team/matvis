=======
vis_cpu
=======
.. image:: https://github.com/hera-team/vis_cpu/workflows/Tests/badge.svg
    :target: https://github.com/hera-team/vis_cpu
.. image:: https://badge.fury.io/py/vis-cpu.svg
    :target: https://badge.fury.io/py/vis-cpu
.. image:: https://codecov.io/gh/hera-team/vis_cpu/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/hera-team/vis_cpu
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


Fast visibility simulator capable of running on CPU and GPU.


Description
===========

``vis_cpu`` is a Python/numpy-based simulator for interferometer visibilities.
It models the sky as an ensemble of point sources, each with their own frequency
spectrum. Diffuse emission can be modelled by treating (e.g.) each pixel of a Healpix
map as a separate source. The code is capable of modelling polarized visibilities
and primary beams, but currently only a Stokes I sky model.

``vis_cpu`` includes a separate ``pycuda``-based implementation called ``vis_gpu``.
This is intended to keep feature parity with the ``vis_cpu`` code to the greatest
extent possible.

An example wrapper for the main ``vis_cpu`` simulator function is provided in this
package (``vis_cpu.wrapper.simulate_vis()``).

Installation
============
Merely do ``pip install vis_cpu``. If you want to use the GPU functions, install
with ``pip install vis_cpu[gpu]``.

Developers
==========
Run ``pre-commit install`` before working on this code.

Read the Docs
=============
https://vis-cpu.readthedocs.io/en/latest/
