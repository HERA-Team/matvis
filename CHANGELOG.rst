=========
Changelog
=========

Version 0.2.3
=============

Fixed
-----

- Fix issue with spurious beam normalization when a pixel beam
  interpolation grid is generated from a UVBeam object
- Fix bug where the imaginary part of complex pixel beams was
  being discarded
- Fix bug that was causing polarized calculations to fail with
  ``simulate_vis``
- CI paths fixed so coverage reports are linked properly

Added
-----

- New units tests

Version 0.2.2
=============

Fixed
-----

- Fix issue with complex primary beams being cast to real

Version 0.2.1
=============

Fixed
-----

- Make IPython import optional.

Version 0.2.0
=============

Changed
-------

- ``lm_to_az_za`` --> ``enu_to_az_za`` and added ``orientation`` parameter. Significant
  increase in documentation of this and related coordinate functions.
- Refactoring of construction of spline within main CPU routine to its own function:
  ``construct_pixel_beam_spline``.

Added
-----

- ``eci_to_enu_matrix`` function
- ``enu_to_eci_matrix`` function
- ``point_source_crd_eq`` function
- ``equatorial_to_eci_coords`` function
- ``uvbeam_to_lm`` function
- New ``plotting`` module with ``animate_source_map`` function.
- Ability to do **polarization**! (Only in ``vis_cpu`` for now, not GPU).
- New ``wrapper`` module with ``simulate_vis`` function that makes it easier to simulate
  over an array of frequencies and source positions in standard RA/DEC (i.e. it does
  the frequency loop, and calculates the rotation matrices for you). It is an *example*
  wrapper for the core engine.
- Many more unit tests.

Version 0.1.2
=============

Fixed
-----

- Installation of gpu extras fixed.

Version 0.1.1
=============

Fixed
-----

- Fix import logic for GPU.

Version 0.1.0
=============

- Port out of hera_sim.
