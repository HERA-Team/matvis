"""Coordinate rotation methods."""

import numpy as np
from astropy.coordinates import AltAz
from astropy.coordinates.erfa_astrom import erfa_astrom

from ..coordinates import point_source_crd_eq
from ..core.coords import CoordinateRotation

# Schwarzschild radius of the Sun (au) */
ERFA_SRS = 1.97412574336e-8


class CoordinateRotationAstropy(CoordinateRotation):
    """Perform coordinate rotation with astropy directly."""

    def rotate(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute the topocentric coordinates from the given time and telescope loc."""
        frame = AltAz(obstime=self.times[t], location=self.telescope_loc)
        altaz = self.skycoords.transform_to(frame)
        el, az = self.xp.asarray(altaz.alt.rad), self.xp.asarray(altaz.az.rad)

        # Astropy has Az oriented East of North, i.e. Az(N) = 0 deg, Az(E) = +90 deg.
        self.all_coords_topo[0] = self.xp.cos(el) * self.xp.sin(az)
        self.all_coords_topo[1] = self.xp.cos(el) * self.xp.cos(az)
        self.all_coords_topo[2] = self.xp.sin(el)

        if self.gpu:
            self.xp.cuda.Device().synchronize()


class CoordinateRotationERFA(CoordinateRotation):
    """Perform coordinate rotation with functions pulled from ERFA.

    This class does almost exactly the same thing as Astropy, but calls the underlying
    ERFA functions directly. This allows us to remove some unnecessary computations for
    our particular use-case (specifically, refraction and extraneous to-and-froing
    between spherical and cartesian representations).

    All functions were essentially transcribed and pythonized directly from the
    excellent ERFA C-library.
    """

    def setup(self):
        """Standard setup, as well as storing the cartesian representation of ECI."""
        super().setup()
        self._eci = self.xp.asarray(
            point_source_crd_eq(self.skycoords.ra, self.skycoords.dec)
        )

    def _atioq(self, xyz: np.ndarray, astrom):
        # cirs to hadec rot
        ce = np.cos(astrom["eral"])
        se = np.sin(astrom["eral"])

        c2h = self.xp.array([[ce, se, 0], [-se, ce, 0], [0, 0, 1]])

        # Polar motion.
        sx = np.sin(astrom["xpl"])
        cx = np.cos(astrom["xpl"])
        sy = np.sin(astrom["ypl"])
        cy = np.cos(astrom["ypl"])
        pm = self.xp.array(
            [[cx, 0, sx], [sx * sy, cy, -cx * sy], [-sx * cy, sy, cx * cy]]
        )

        # hadec to enu
        enu = self.xp.array(
            [
                [0, 1, 0],
                [-astrom["sphi"], 0, astrom["cphi"]],
                [astrom["cphi"], 0, astrom["sphi"]],
            ]
        )

        rot = enu.dot(pm.dot(c2h))

        return self.xp.matmul(rot, xyz, out=xyz)

    def _ld(self, p: np.ndarray, e: np.ndarray, em: float, dlim: float):
        """
        Apply light deflection by a solar-system body.

        Parameters
        ----------
        p : np.ndarray (3, nsrc)
            direction from observer to source (unit vector)
        e : np.ndarray (3,)
            direction from body to observer (unit vector)
        em
            distance from body to observer (au)
        dlim
            deflection limiter (Note 4)
        """
        # p,q are shape (3, Nsrc)

        # /* q . (q + e). */
        qpe = (p.T + e).T
        qdqpe = self.xp.sum(p * qpe, axis=0)
        self.xp.clip(qdqpe, a_min=dlim, a_max=None, out=qdqpe)

        # /* 2 x G x bm / ( em x c^2 x ( q . (q + e) ) ). */
        w = ERFA_SRS / em / qdqpe

        # p x (e x q). */
        eq = self.xp.cross(e.T, p.T)
        peq = self.xp.cross(p.T, eq).T

        # Apply the deflection. */
        p += w[None] * peq

    def _ab(self, pnat: np.ndarray, v: np.ndarray, s: float, bm1: float):
        """
        Apply aberration to transform natural direction into proper direction.

        Parameters
        ----------
        pnat : (3, nsrc) array
            natural direction to the source (unit vector)
        v : (3,) vector
            observer barycentric velocity in units of c
        s
            distance between the Sun and the observer (au)
        bm1
            ``sqrt(1-|v|^2)``: reciprocal of Lorenz factor

        """
        # pnat is shape (3, nsrc)

        pdv = self.xp.dot(v, pnat)  # shape (nsrc,)
        w1 = 1.0 + pdv / (1.0 + bm1)  # shape (nsrc,)
        w2 = ERFA_SRS / s  # float
        pnat *= bm1
        pnat += self.xp.outer(v, w1) + w2 * (v - (pdv * pnat).T).T
        r = self.xp.linalg.norm(pnat, axis=0)
        pnat /= r

    def _bpn(self, p, astrom):
        """Apply bias-precession-nutation."""
        self.xp.matmul(self.xp.asarray(astrom["bpn"]), p, out=p)

    def _atciqz(self, eci: np.ndarray, astrom: dict):
        """
        A slightly modified version of the ERFA function ``eraAtciqz``, from astropy.

        Parameters
        ----------
        eci
            Astrometric ICRS positions of sources in cartesian representation
        astrom : eraASTROM array
            ERFA astrometry context, as produced by, e.g. ``eraApci13`` or ``eraApcs13``

        Returns
        -------
        ri : float or `~numpy.ndarray`
            Right Ascension in radians
        di : float or `~numpy.ndarray`
            Declination in radians
        """
        # Light deflection by the Sun, giving BCRS natural direction.
        self._ld(eci, self.xp.asarray(astrom["eh"]), astrom["em"], 1e-6)

        # Aberration, giving GCRS proper direction.
        self._ab(eci, self.xp.asarray(astrom["v"]), astrom["em"], astrom["bm1"])

        # Bias-precession-nutation, giving CIRS proper direction.
        # Has no effect if matrix is identity matrix, in which case gives GCRS ppr.
        self._bpn(eci, astrom)

    def _apco(self, observed_frame):
        return erfa_astrom.get().apco(observed_frame)

    def _get_obsf(self, obstime, location):
        return AltAz(obstime=obstime, location=location)

    def rotate(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Rotate the coordinates into the observed frame."""
        obsf = self._get_obsf(self.times[t], self.telescope_loc)
        astrom = self._apco(obsf)

        # Copy the eci coordinates, because these routines modify them in-place
        self.all_coords_topo[:] = self._eci[:]

        # convert to topocentric CIRS
        self._atciqz(self.all_coords_topo, astrom)

        # now perform observed conversion
        self._atioq(self.all_coords_topo, astrom)

        if self.gpu:
            self.xp.cuda.Device().synchronize()
