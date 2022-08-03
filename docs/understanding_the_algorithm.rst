.. understanding_the_algorithm::

=========================
The ``vis_cpu`` algorithm
=========================

What Is ``vis_cpu``?
====================

``vis_cpu`` is a package for simulating radio interferometer observations.
That is, it simulates observations of the radio-frequency intensity of the sky by
*baselines* (i.e. correlated pairs of antennas). This is useful for validating analysis
pipelines, or understanding observational systematics on theoretical predictions.

The basic high-level idea of ``vis_cpu`` is that you give it a few ingredients: a model
of the sky in "normal" (image) space, a model of the sensitivity of each antenna to
different directions and frequencies, and a set of antenna positions, then ``vis_cpu``
will simulate what the array of antennas should observe (if no noise or other systematics
are present -- these can typically be added later if required).
