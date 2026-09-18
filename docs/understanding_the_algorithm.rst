=======================================
Understanding the ``matvis`` algorithm
=======================================

What Is ``matvis``?
====================

``matvis`` is a package for simulating radio interferometer observations.
That is, it simulates observations of the radio-frequency intensity of the sky by
*baselines* (i.e. correlated pairs of antennas). This is useful for validating analysis
pipelines, or understanding observational systematics on theoretical predictions.

The basic high-level idea of ``matvis`` is that you give it a few ingredients: a model
of the sky in "normal" (image) space, a model of the sensitivity of each antenna to
different directions and frequencies, and a set of antenna positions, then ``matvis``
will simulate what the array of antennas should observe (if no noise or other systematics
are present -- these can typically be added later if required).

There are many codes that do this same basic task, for example,
`pyuvsim <https://github.com/RadioAstronomySoftwareGroup/pyuvsim>`_. Each has its own
approximations and performance considerations. The ``matvis`` package does two unique
things:

    1. It splits up the calculation in a novel way, using an antenna-based approach
       instead of a baseline-based approach. This makes some of its calculations scale
       as :math:`N_{\rm ant}` instead of :math:`N_{\rm ant}^2`. The last step of the
       algorithm, which is unavoidably :math:`N_{\rm ant}^2`, is a simple matrix product,
       which is extremely well-tuned on most modern systems through software like BLAS.
    2. The algorithm lends itself to implementation on GPUs, since the dominant parts
       of the algorithm are bilinear interpolation and a matrix product, both of which
       are fantastically fast on GPUs. Therefore, the ``matvis`` *algorithm* is seen
       as distinct from its *implementation*, and the ``matvis`` package defines *two*
       implementations: ``matvis.cpu`` and ``matvis.gpu``, which have the same API.

The ``matvis`` Framework
=========================

The visibility observed on a baseline formed by antennas *i* and *j* at frequency :math:`\nu` is

.. math:: V_{ij} = \int_{\rm sky} \mathcal{A}_i \mathcal{C} \mathcal{A}_j^\dagger \exp(-2\pi \nu i \vec{b}_{ij} \hat{n}/c) d^2 \Omega,

where :math:`\mathcal{A}_i` is the complex, polarized beam of antenna *i*,
:math:`\mathcal{C}` is the "coherency matrix" which is essentially the polarized sky model,
:math:`\vec{b}_{ij}` is the vector pointing from antenna *i* to antenna *j*, *c* is
the speed of light,
and :math:`\hat{n}` is the unit vector in the direction of the sky.
The integral is over all angles in the sky, and both the beam and sky model are
angle-dependent.

From here, all visibility simulators must make at least one approximation: we do not
have analytic forms for either the sky or beam model, as a function of angle.
Thus, we cannot perform the integral with arbitrary precision. Instead, we must make a
choice about which *discrete basis* the sky model should be represented in so that the
integral can be discretized. A very natural basis is a "pixelization", i.e. a choice of
a set of points on the sky to which we assign the total intensity for a local region around
them. This is in fact a perfect representation if the sky completely consisted of
unresolved point-sources. It is imperfect if the sky is diffuse, and the integration then
effectively becomes a Riemann sum over the sphere. Other discretization choices are possible,
for example spherical harmonics. However, we use the simple pixelization representation
in ``matvis``. Note that the user is responsible for performing this discretization:
``matvis`` is agnostic to the specific choice of pixel positions, except that it gives
equal weight to each pixel (thus, for a diffuse sky, it is assumed that each pixel's
value is the total intensity within regions of *fixed surface area*). One choice of
explicit pixelization that is consistent with these assumptions is the HEALpix pixelization.

In ``matvis``, we also make the following assumptions/approximations (these aren't
fundamental to the algorithm, and may be updated at a later date):

    1. The sky is unpolarized
    2. The ground provides a perfectly conducting ground plane and is perfectly flat
       out to the horizon (i.e., we see everything up to the horizon, and nothing at all
       beyond it).
    3. The Earth rotates as a rigid body along a single axis. This makes updating of
       sky coordinates over time much faster, at the expense of a little bit of accuracy,
       if a long time is simulated.

Now, let the discrete pixels of the sky model (or discrete sources, if the sky model is
composed of such) *in topocentric coordinates* (i.e. sin-projected l, m)
be :math:`\vec{X}(t)`, and their flux-density by *I*.

Then, with all these approximations in place, we can rewrite our visibility equation for
baseline *ij* and feed-pair *pq* as:

.. math:: V^{pq}_{ij}(t) = \sum_n \vec{A}^p_i(\vec{X}_n(t)) \cdot \vec{A}^q_i(\vec{X}_n(t)) I_n \exp(-2\pi i \nu \vec{X}_n \cdot \vec{b}_{ij}/c).

This is the equation that ``matvis`` calculates.

The ``matvis`` Algorithm
=========================

Having the above mathematical framework, we can understand the steps of the ``matvis``
algorithm. Firstly, we realize that the above equation is performed at a single frequency.
Thus, frequency forms our outer-most loop. Our second loop is over times.

We ask the user to give us the following:

    1. A set of antenna locations, :math:`D` in Cartesian East-North-Up coordinates as a
       :math:`N_{\rm ant} \times 3` matrix.
    2. A beam model, :math:`A_i(\nu, \vec{\theta})` for each antenna that may be
       evaluated (or interpolated) to any particular set of topocentric coordinates.
       In general, the beam should be defined for each component of the electric field (ax)
       and for each feed of the antenna (feed).
    3. A set of sky model pixel/source locations in Cartesian equatorial coordinates (ECI).
       This is a coordinate system in which the positions are fixed with respect to
       distance stars (i.e. do not depend on the Earth's rotation). Explicitly, in terms
       of RA/DEC, each source has the unit-vector
       (cos(RA) cos(Dec), sin(RA) cos(Dec), sin(Dec)). Let the :math:`3 \times N_{\rm src}`
       matrix of these sources be called :math:`X_{\rm eq}`.
    4. A length :math:`N_{\rm src}` vector of source intensities, :math:`I`.

Then, for a particular frequency and time, the ``matvis`` algorithm is:

    1. Compute a 3x3 rotation matrix, :math:`R_t`, that rotates the equatorial locations
       of the pixels/sources into the topocentric frame. This depends only on the latitude
       of the telescope and the hour-angle at the particular time.
    2. Rotate the pixels/sources into topocentric frame: :math:`X = R_t X_{\rm eq}`,
       where :math:`X` is a :math:`3 \times N_{\rm src}` matrix.
    3. Mask all sources that are below the horizon (i.e. :math:`X_2 < 0`), leaving
       :math:`X` as a :math:`3 \times N'_{\rm src}` matrix.
    4. Interpolate the beam model onto the topocentric coordinates, i.e. produce the
       :math:`N_{\rm feed}N_{\rm ant} \times N_{\rm ax}N'_{\rm src}` matrix
       :math:`A_{ij, kl} = A_{ijk}(X_l)`.
    5. Compute the antenna-based exponent:
       :math:`\tau = -2 \pi i \nu D \cdot X / c`, where
       :math:`\tau` is a :math:`N_{\rm ant}\times N_{\rm src}` matrix.
    6. Compute the :math:`N_{\rm feed}N_{\rm ant} \times N_{\rm ax}N'_{\rm src}`
       "pseudo"-visibility of an antenna:
       :math:`Z_{ij, kl} = \sqrt{I}_l A_{ij, kl} \exp(\tau_{jl})`.
    7. Compute the :math:`N_{\rm feed} N_{\rm ant} \times N_{\rm feed} N_{\rm ant}`
       visibility: :math:`V = Z Z^*`.

Reducing the Cost of Step 7: Block-Decomposed Products
========================================================

Step 7's matrix product is unavoidably :math:`N_{\rm ant}^2`, and dominates the total
runtime for interferometers of a realistic size (see :doc:`performance`). For arrays
where the requested baselines (``antpairs``) are concentrated within a modest number of
antenna groupings -- most commonly a compact, highly-redundant core, or an array with a
compact core plus a handful of more remote outrigger antennas -- ``matvis`` can compute
this product as a set of smaller sub-matrix products instead of the full
:math:`N_{\rm ant} \times N_{\rm ant}` product, via the ``CPUMatBlock``/``GPUMatBlock``
``matprod_method`` options.

**When to use it.** Pass an ``antenna_blocks`` argument to :func:`~matvis.simulate_vis`
(a list of ``(row_antenna_idx, col_antenna_idx)`` integer-array tuples) along with
``matprod_method="CPUMatBlock"`` (or ``"GPUMatBlock"``), and ``matvis`` computes one
smaller matrix product per block instead of the single full one, gathering just the
requested ``antpairs`` out of each block. This computes *exactly* the same visibilities
as the default ``MatMul`` method -- unlike the sky-coarsening approximation discussed for
some future work, this is not an approximation, just a different way of arranging the
same exact computation -- so there is no accuracy trade-off to consider, only a
performance one:

- If your antenna pairs of interest cluster into a handful of spatial groups (e.g. "this
  set of antennas is the compact core, this smaller set is the outriggers"), using
  blocks that respect that grouping can substantially reduce the number of FLOPs
  computed relative to the full product, since pairs *between* distant groups that
  aren't requested are simply never computed.
- Even without any grouping structure, tiling the full antenna set into fixed-size
  blocks bounds the peak memory of a single matrix-product call, which can be useful
  for very large arrays regardless of any redundancy consideration.
- If your blocks don't reflect any real structure in the requested ``antpairs``, this
  mechanism buys you little over the default ``MatMul`` (and adds bookkeeping overhead
  in ``setup()``), so it isn't a good default choice for every case.

``matvis`` deliberately does not try to work out good blocks for you -- that is a
judgment call about your specific array's geometry and which baselines you actually
want, and :mod:`matvis.redundancy` offers a few different starting points depending on
how much you already know about your array:

- If you already know a sensible grouping of your antennas (e.g. an exact core/outrigger
  split from your array-layout generator), pass those group labels to
  :func:`~matvis.redundancy.blocks_from_groups`.
- If you don't have a grouping in hand but your array has a compact core plus a small
  number of more remote antennas (as is typical of HERA-like layouts),
  :func:`~matvis.redundancy.radial_groups` gives a reasonable default: it bins antennas
  into concentric shells by distance from the array center, which can be fed straight
  into :func:`~matvis.redundancy.blocks_from_groups`.
- If you want every antenna pair computed (no dedup at all), but with bounded per-block
  memory, use :func:`~matvis.redundancy.tile_antennas`.
- If you specifically want to only compute one representative baseline out of each
  redundant group, :func:`~matvis.redundancy.find_redundant_antpairs` finds those
  representative pairs, and :func:`~matvis.redundancy.antpairs_to_blocks` turns them
  into singleton blocks (though for that particular case, the existing ``VectorDot``
  method, run directly against the deduplicated ``antpairs``, does the same job just as
  well).
