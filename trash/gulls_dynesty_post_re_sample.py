import os
import sys
import dynesty
import numpy as np
import multiprocessing as mp
from astropy import units as u
import astropy.coordinates as astrocoords
from VBBinaryLensing import VBBinaryLensing
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.time import Time
from scipy.interpolate import interp1d
from astroquery.jplhorizons import Horizons
import pandas as pd
import pickle
import corner
import matplotlib.pyplot as plt
from dynesty import plotting as dyplot

# Here is where parallax mags are computed
# bozzaPllxOMLCGen.cpp

# Here is where chi2 calculated
# pllxLightcurveFitter.cpp

# debug event: 331
# N = 6757
# chi2 = 6762 - binary
# chi2 = 6857 - single
# delta chi2 = 95

# limb darkening coefficients !!!
# 0.36 in all bands - linear limb darkening
# W146, Z087, K213

# probably dynesty would make more sense for posterior sampling because it will
# have z values.

# rE in AU (Einstein radius)
# Event->rE = rEsun * sqrt(
#     Lenses->data[ln][MASS] * Sources->data[sn][DIST] * (1 - x) * x
# )

# thetaE in mas (angular Einstein radius)
# Event->thE = Event->rE/Lenses->data[ln][DIST];

# relative ls proper motion - lens motion relative to the source in mas/yr
# calculate the heliocentric relative proper motion
# pmgal[0] = Lenses->data[ln][MUL]-Sources->data[sn][MUL];
# pmgal[1] = Lenses->data[ln][MUB]-Sources->data[sn][MUB];

# work out its absolute value
# Event->murel = qAdd(pmgal[0],pmgal[1]);  # add in quadrature

# calculate the parallax
# Event->piE = (1-x)/Event->rE;

# fractional lens source distance
# x = Lenses->data[ln][DIST]/Sources->data[sn][DIST];

# relative transverse velocity in kms-1
# Event->vt = Event->murel * Lenses->data[ln][DIST] * AU/1000 / SECINYR;

# tE in days
# Event->tE_h = DAYINYR * Event->thE / Event->murel;

# source size (rho) in Einstein radii
# Event->rs = (
#     Sources->data[sn][RADIUS] * Rsun / Sources->data[sn][DIST]
# ) / Event->thE;
# radius (Rsun) -> AU / Ds (kpc) -> mas / thetaE (mas) = ratio

# rate weighting
# Event->raww = Event->thE * Event->murel;


class Orbit:

    def __init__(
        self,
        obs_location="SEMB-L2",
        start_date="2018-08-10",
        end_date="2023-04-30",
        refplane="ecliptic",
        n_epochs=None,
        origin="500@10",
        date_format="iso",
    ):
        """Position file from JPL Horizons"""

        self.start_time = Time(start_date, format=date_format)
        self.end_time = Time(end_date, format=date_format)

        if n_epochs is None:
            self.n_epochs = int(
                self.end_time.jd - self.start_time.jd + 1
            )  # 1 epoch per day
        else:
            self.n_epochs = n_epochs

        self.obs_location = obs_location
        self.origin = origin
        self.refplane = refplane

        self.epochs, self.positions, self.velocities = (
            self.fetch_horizons_data()
        )

    def fetch_horizons_data(self):
        times = np.linspace(
            self.start_time.jd, self.end_time.jd, self.n_epochs
        )
        times = Time(times, format="jd")

        positions_list = []
        velocities_list = []

        # Split the times into chunks to avoid hitting the API limits
        chunk_size = 75  # Adjust this based on the API limits
        for i in range(0, len(times), chunk_size):
            chunk_times = times[i : i + chunk_size]
            q = Horizons(
                id=self.obs_location,
                location=self.origin,
                epochs=chunk_times.jd,
            )
            data = q.vectors(refplane=self.refplane)

            positions_list.append(
                CartesianRepresentation(data["x"], data["y"], data["z"])
            )
            velocities_list.append(
                CartesianDifferential(data["vx"], data["vy"], data["vz"])
            )

        # Combine the chunks into single arrays
        positions = CartesianRepresentation(
            np.concatenate([pos.x for pos in positions_list]),
            np.concatenate([pos.y for pos in positions_list]),
            np.concatenate([pos.z for pos in positions_list]),
        )
        velocities = CartesianDifferential(
            np.concatenate([vel.d_x for vel in velocities_list]),
            np.concatenate([vel.d_y for vel in velocities_list]),
            np.concatenate([vel.d_z for vel in velocities_list]),
        )

        return times, positions, velocities

    def get_pos(self, t):
        """Get the observatory position at time ``t`` by interpolating the
        position file."""
        x_interp = interp1d(self.epochs.jd, self.positions.x.to(u.au).value)
        y_interp = interp1d(self.epochs.jd, self.positions.y.to(u.au).value)
        z_interp = interp1d(self.epochs.jd, self.positions.z.to(u.au).value)
        x = x_interp(t)
        y = y_interp(t)
        z = z_interp(t)
        xyz = np.vstack((x, y, z))
        return xyz

    def get_vel(self, t):
        """Get the observatory velocity at time ``t`` by interpolating the
        position file."""
        vx_interp = interp1d(
            self.epochs.jd, self.velocities.d_x.to(u.au / u.day).value
        )
        vy_interp = interp1d(
            self.epochs.jd, self.velocities.d_y.to(u.au / u.day).value
        )
        vz_interp = interp1d(
            self.epochs.jd, self.velocities.d_z.to(u.au / u.day).value
        )
        vx = vx_interp(t)
        vy = vy_interp(t)
        vz = vz_interp(t)
        vxyz = np.vstack((vx, vy, vz)) * (u.au / u.day)
        return vxyz


class Parallax:

    def __init__(self, ra, dec, orbit, t_ref):
        self.ra = ra * u.deg
        self.dec = dec * u.deg
        self.event_coords = astrocoords.SkyCoord(
            ra=self.ra, dec=self.dec, frame="icrs"
        )
        self.orbit = orbit
        self.set_ref_frame(t_ref)
        self.rotate_view()

        # Event->pllx[obsidx].reset();

    # Event->pllx[obsidx].set_reference(
    #     Paramfile->simulation_zerotime + tref,
    #     &World[0].orbit,
    # )
    # Event->pllx[obsidx].set_orbit(&World[obsidx].orbit)
    # Event->pllx[obsidx].set_lb(Event->l, Event->b)
    # Event->pllx[obsidx].set_pm_lb(
    #     Lenses->data[ln][MUL] - Sources->data[sn][MUL],
    #     Lenses->data[ln][MUB] - Sources->data[sn][MUB],
    # )
    # Event->pllx[obsidx].set_piE(Event->piE)
    # Event->pllx[obsidx].set_tE_h(Event->tE_h)  # tE in the heliocentric frame

    def set_ref_frame(self, t_ref):
        self.t_ref = t_ref  # bjd
        self.xref = self.orbit.get_pos(t_ref)
        self.vref = self.orbit.get_vel(t_ref)

    def update_piE_NE(self, piEN, piEE):
        """update the parallax parameters"""
        self.piEN = piEN
        self.piEE = piEE

        self.piE = np.array([piEN, piEE])

    def dperp():
        """calculate the perpendicular distance to the source"""
        pass

    def rotate_view(self):
        """rotates from x, y, z to n, e, d."""

        # unit vector pointing to the source
        self.rad = np.array(
            [
                np.cos(self.ra.rad) * np.cos(self.dec.rad),
                np.sin(self.ra.rad) * np.cos(self.dec.rad),
                np.sin(self.dec.rad),
            ]
        )

        # north vector in x, y, z
        north = np.array([0, 0, 1])

        # unit vector ointing east in the lens plane
        e_unit = np.cross(north, self.rad) / np.linalg.norm(
            np.cross(north, self.rad)
        )

        # unit vector pointing north in the lens plane
        n_unit = np.cross(self.rad, e_unit)

        # rotation matrix
        self.rot_matrix = np.array([n_unit, e_unit, self.rad])

        # rotate the reference values into the source pointing frame
        self.xref = np.dot(self.rot_matrix, self.xref)
        self.vref = np.dot(self.rot_matrix, self.vref)

    def get_pos(self, t):
        """Get the position of the observatory at time ``t`` in the n, e, d
        frame."""
        xyz = self.orbit.get_pos(t)
        ned = np.dot(self.rot_matrix, xyz)
        return ned

    def parallax_shift(self, t):
        """Calculate the parallax shift.
        Gould, A., 2004, ApJ, 606, 319.

        Let ``s(t)`` be the Earth-to-Sun vector in units of AU in the
        heliocentric frame. Let ``tp`` be some fixed time, in practice very
        close to the time ``t0`` of the peak of the event as seen from the
        Earth, and
        evaluate the derivative of ``s(t)`` at this time,

        \[\Delta s(t) = s(t) - s(t_ref) = (t-t_ref) v_ref\]

        observations are toward an event at some given celestial coordinates
        and define nˆ and eˆ as the unit vectors pointing north and east.

        \[(dtau, dbeta) = (
        s_n(t)\pi_{EN} + s_e(t)\pi_{EE},
        -s_n(t)\pi_{EE} + s_e(t)\pi_{EN})\]
        """
        x = self.get_pos(
            t
        )  # xyz heliocentric L2 position of the observatory at time t,
        # in the ecliptic plane?
        x = np.dot(
            self.rot_matrix, x
        )  # rotates the L2 position vector into the source-pointing
        # coordinate system

        NEshift = [0, 0]
        # L2-Sun shift with n^ e^ defined when looking at the event
        NEshift[0] = (
            x[0] - self.xref[0] - (t - self.tref) * self.vref[0]
        )  # array the same size as t. n^ shift component
        NEshift[1] = (
            x[1] - self.xref[1] - (t - self.tref) * self.vref[1]
        )  # e^ shift component
        # I think these assume that vref (x, y) are a good enough
        # approximation for the velocity of the observatory, throughout
        # the event; i.e. the transverse velocity is assumed to be
        # constant. I'm not sure if this is a good assumption.
        # I dunno. I'm not following Andy here.

        # Michael's code (ground-based observer)
        # q_n = -S_n_arr[0] - vn0 * (ts_in - t_peak)   # v is Earth's
        # perpendicular velocity
        # q_e = -S_e_arr[0] - ve0 * (ts_in - t_peak)
        # I think he has north and east in the opposite direction to Matt
        # delta_tau = q_n*pi_EN + q_e*pi_EE
        # delta_beta = -q_n*pi_EE + q_e*pi_EN
        # ^ this method was introducting machine error

        # relative lens source proper motion angle in n^ e^ frame
        phi_pi = np.arctan2(self.piEE, self.piEN)

        cs = np.cos(phi_pi)
        sn = np.sin(phi_pi)

        # Convert the shift in the observer plane to a shift in the source
        # position
        tushift = np.zeros(2)
        tushift[0] = -self.piE * (
            NEshift[0] * cs + NEshift[1] * sn
        )  # Delta_tau - shift in the relative-motion direction
        tushift[1] = -self.piE * (
            -NEshift[0] * sn + NEshift[1] * cs
        )  # Delta_beta - shift perpendicular to the lens-source motion

        return tushift


# 2 pi radians / period
# phase is an angle in radians
# phase0 is the phase at t0
# assume a circular orbit and a small mass ratio m << M


class Event:

    def __init__(self, parallax, orbit, data, truths):
        t_ref = truths[5]  # the real t0
        self.parallax = parallax
        self.parallax.set_ref_frame(t_ref)
        self.data = data
        self.true_params = truths
        self.orbit = orbit

    def set_params(self, params):
        self.params = params
        piEN = params[7]
        piEE = params[8]
        self.parallax.update_piE_NE(self, piEN, piEE)

    def projected_separation(self, i, period, t, phase_offset=0.0, a=1.0):
        """Calculate the projected separation of the binary lens.

        Parameters
        ----------
        i : float or array_like
            Inclination of the orbit in radians.
        period : float or array_like
            Orbital period.  If ``t`` is given in units of ``tE`` then
            ``period`` must be as well.
        t : float or array_like
            Time relative to ``t0``.
        phase_offset : float, optional
            Initial orbital phase in radians. Default is ``0.0``.
        a : float, optional
            Semimajor axis in units of ``theta_E``. Default is ``1.0``.

        Returns
        -------
        s : float or ndarray
            Projected separation of the two components.
        x : float or ndarray
            X-coordinate of the secondary relative to the primary.
        y : float or ndarray
            Y-coordinate of the secondary relative to the primary.
        """

        phase = 2 * np.pi * (t) / period + phase_offset
        x = a * np.cos(phase)
        y = a * np.sin(phase) * np.cos(i)
        z = a * np.sin(phase) * np.sin(i)

        # if an inclination of 0 means birds eye view, then use x, y
        s = np.sqrt(x**2 + y**2)

        # if an inclination of 0 means edge on view, then use x, z
        # s = np.sqrt(x**2 + z**2)

        return s, x, y

    def get_magnification(self, t):
        """Calculate the magnification"""

        q = self.params[1].copy()
        s = self.params[0].copy()

        i = self.params[9].copy()  # inclination of the orbit
        phase0 = self.params[10].copy()  # phase at time t0
        period = self.params[11].copy()  # planet orbital period

        s0, _, _ = self.projected_separation(i, period, 0.0, phase0)
        a = s / s0  # semimajor axis in uits of thetaE

        a1 = q / (1.0 + q) * a
        a2 = a - a1

        # Notes:
        # m1a1 = m2a2  (a is 'distance' from COM)
        # a1 + a2 = a
        # m1/m2 = q     (q<1, and m2<m1)
        # a2 = a1/q
        # a = a1 + a1/q = a1(1+1/q) = a1(q+1)/q
        # a1 = aq/(1+q)

        alpha = self.params[4].copy()  # in radians
        cosa = np.cos(alpha)
        sina = np.sin(alpha)

        t0 = self.params[5].copy()  # time of event peak in BJD
        tE = self.params[6].copy()  # Einstein crossing time/event timescale
        tau = (t - t0) / tE  # event time in units of tE

        # Orbital motion - dsdt
        s1, x1, y1 = self.projected_separation(
            i, period / tE, tau, phase0 + np.pi, a1
        )  # star
        s2, x2, y2 = self.projected_separation(
            i, period / tE, tau, phase0, a2
        )  # planet
        ss = np.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2
        )
        # i don't know that this is strictly necessary since they are always
        # opposite
        print("debug: s: ", ss, s1 + s2)  # these should be equal, I think

        # umin = min(umin, np.sqrt(tt**2, uu**2))

        # Orbital motion - dalphadt
        rot = np.arctan2(y2, x2)
        cosrot = np.cos(-rot)
        sinrot = np.sin(-rot)

        u0 = self.params[3].copy()  # impact parameter

        # 'source' trajectory with parallax
        Delta_tau, Delta_beta = self.parallax.parallax_shift(
            t
        )  # offset vectors due to parallax
        # t needs to be in HJD
        tt = tau + Delta_tau
        uu = u0 + Delta_beta

        xsin = tt * cosa - uu * sina
        ysin = tt * sina + uu * cosa

        xsrot = xsin * cosrot - ysin * sinrot
        ysrot = xsin * sinrot + ysin * cosrot

        rho = self.params[2].copy()  # source radius in units of thetaE

        eps = 1.0e-3
        gamma = 0.36
        vbbl = VBBinaryLensing()
        A = vbbl.BinaryMagDark(ss, q, xsrot, ysrot, rho, gamma, eps)
        # s is a vector to account for OM
        # xsrot and ysrot have been shifted and rotated to account for parallax
        # and OM
        # gamma is the limb darkening coefficient, which is fixed for the
        # gulls run
        # eps is the precision of the numerical integration

        return A


# Fitting Functions
# ------------------


def get_fluxes(model: np.ndarray, f: np.ndarray, sig2: np.ndarray):
    """Solves for the flux parameters for a givin model using least squares.

    Parameters:
    -----------
    model: model magnification curve
    f: observed flux values
    sig2: flux errors.

    Returns:
    --------
    FS: source flux
    FB: blend flux.
    """
    # A
    A11 = np.sum(model**2 / sig2)
    Adiag = np.sum(model / sig2)
    A22 = np.sum(1.0 / sig2)
    A = np.array([[A11, Adiag], [Adiag, A22]])

    # C
    C1 = np.sum((f * model) / sig2)
    C2 = np.sum(f / sig2)
    C = np.array([C1, C2]).T

    # B
    B = np.linalg.solve(A, C)
    FS = float(B[0])
    FB = float(B[1])

    return FS, FB


def get_chi2(event, params):

    event.set_params(params)

    # sim_time, obs_rel_flux, obs_rel_flux_err, true_rel_flux,
    # true_rel_flux_err, observatory_code, saturation_flag, _, pllx_shift_t,
    # pllx_shift_u, BJD, x_s, y_s, x_l1, y_l1, x_l2, y_l2 = data

    t = event.data[0]  # BJD
    f = event.data[1]  # obs_rel_flux
    f_err = event.data[2]  # obs_rel_flux_err

    A = event.get_magnification(t)
    fs, fb = get_fluxes(A, f, f_err**2)

    chi2 = ((f - A * fs + fb) / f_err) ** 2

    return chi2


def lnlike(theta, event):

    chi2 = get_chi2(event, theta)
    return -0.5 * chi2


def lnprior(theta, event, bound_penalty=False):
    s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period = theta
    sinphase = np.sin(phase)
    s0 = event.projected_separation(i, period / tE, 0.0, phase)
    if (
        tE > 0.0
        and q <= 1.0
        and period / tE > 4
        and sinphase < 0.9
        and sinphase >= 0.00
        and i <= np.pi / 2
        and i >= 0
        and s0 > 0.0
    ):

        if bound_penalty:  # i'm not using this. I need to redo the calculation
            # calculate beta and requiere the orbits conserve energy
            # G_au = 1.0  # gravitational constant in AU^3 /(M1 * years^2)
            # /(2pi)^2
            # m1 = 1  # mass of the first object
            # ßm2 = m1*q  # mass of the second object
            # I1 = a1**2  # *m1
            # I2 = q * a2**2  # *m1
            I = q * a**2  # fixed m1 frame
            period = period / 365.25  # convert period to years
            w = 1.0 / period  # /2pi

            # calculate the gravitational potential energy
            # m1*m2 = m1*(m1*q) = m1^2*q
            # /m1: m1*q
            # /(2pi^)2
            Eg = q / a

            # calculate the rotational kinetic energy
            # /m2
            # /(2pi^)2
            Erot = 0.5 * (I1 + I2) * w**2

            bound_penatly = (Eg - Erot) ** 2
        else:
            bound_penatly = 0.0

        return 0.0 + bound_penatly

    return -np.inf


def lnprob(theta, event):
    # prior
    lp = lnprior(theta, event)
    if not np.isfinite(lp):
        return -np.inf

    # likelihood
    ll = lnlike(theta, event)
    if not np.isfinite(ll):
        return -np.inf

    # prob
    return lp + ll


def prior_transform(u, truths):
    """Transform the unit cube to parameter space.
    Nested sampling has firm boundaries on the prior space."""
    (
        logs,
        logq,
        logrho,
        u0,
        alpha,
        t0,
        tE,
        piEE,
        piEN,
        i,
        sinphase,
        logperiod,
    ) = u

    logs_true = np.log(truths["params"][0])
    logq_true = np.log(truths["params"][1])
    logrho_true = np.log(truths["params"][2])
    u0_true = truths["params"][3]
    alpha_true = truths["params"][4]
    t0_true = truths["params"][5]
    tE_true = truths["params"][6]
    piEE_true = truths["params"][7]
    piEN_true = truths["params"][8]
    i_true = truths["params"][9]
    sinphase_true = np.sin(truths["params"][10])
    logperiod_true = np.log(truths["params"][11])

    # I think these ranges need to stay the same for the logz values to be
    # comparable
    # check how big these uncertainties normally are and adjust the ranges
    # accordingly
    logs_range = 0.5
    logq_range = 1.0  # this one might be poorly constrained
    logrho_range = 1.0  # this one might also be poorly constrained
    u0_range = 0.2
    alpha_range = 0.1
    t0_range = 0.5
    tE_range = 1.0
    piEE_range = 5.0  # these might not be big enough - I don't know how well
    # constrain piE is
    piEN_range = 5.0
    i_range = 0.1  # I have no clue how much to constrain these last three
    sinphase_range = 0.1  # I'm going to have to run some tests
    logperiod_range = 0.1

    logs = logs_true + logs_range * (logs - 0.5)
    logq = logq_true + logq_range * (logq - 0.5)
    logrho = logrho_true + logrho_range * (logrho - 0.5)
    u0 = u0_true + u0_range * (u0 - 0.5)
    alpha = alpha_true + alpha_range * (alpha - 0.5)
    t0 = t0_true + t0_range * (t0 - 0.5)
    tE = tE_true + tE_range * (tE - 0.5)
    piEE = piEE_true + piEE_range * (piEE - 0.5)
    piEN = piEN_true + piEN_range * (piEN - 0.5)
    i = i_true + i_range * (i - 0.5)
    sinphase = sinphase_true + sinphase_range * (sinphase - 0.5)
    logperiod = logperiod_true + logperiod_range * (logperiod - 0.5)

    s = 10.0**logs  # log uniform samples
    q = 10.0**logq
    rho = 10.0**logrho
    phase = np.arcsin(sinphase)  # sin uniform samples
    period = 10.0**logperiod

    return s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period


def new_event(path, sort="alphanumeric"):
    """get the data and true params for the next event"""

    files = os.listdir(path)
    files = sorted(files)

    if path[-1] != "/":
        path = path + "/"

    if not os.path.exists(
        path + "run_list.txt"
    ):  # if the run list doesn't exist, create it
        run_list = np.array([])
        np.savetxt(path + "run_list.txt", run_list, fmt="%s")

    if not os.path.exists(
        path + "complete.txt"
    ):  # if the complete list doesn't exist, create it
        complete_list = np.array([])
        np.savetxt(path + "complete.txt", complete_list, fmt="%s")

    for file in files:
        if "csv" in file:
            master_file = path + file

    if sort == "alphanumeric":

        for file in files:
            run_list = np.loadtxt(path + "run_list.npy")
            if (file not in runlist) and ("csv" not in file):
                runlist = np.vstack(file)
                np.savetxt(path + "runlist.txt", runlist, fmt="%s")
                lc_file_name = file.split(".")[0]
                event_identifiers = lc_file_name.split("_")
                EventID = event_identifiers[-1]
                SubRun = event_identifiers[-3]
                Field = event_identifiers[-2]
                data_file = path + file
                data = load_data(
                    data_file
                )  # bjd, flux, flux_err, tshift, ushift
                event_name = f"{EventID}_{SubRun}_{Field}"
                truths = get_params(
                    master_file, EventID, SubRun, Field
                )  # make sure to turn all the degress to radians
                break

    """if ".txt" in sort:
        files = np.loadtxt(sort)
        for i in range(len(files)):
            if os.path.exists('runlist.npy'):
                runlist = np.loadtxt('runlist.npy')
            else:
                runlist = np.array([])
            if files[i] not in runlist:
                runlist = np.vstack(files[i])
                np.savetxt('runlist.txt', runlist, fmt='%s')
                data = mm.MulensData(file_name='data/' + files[i])
                true_params = np.loadtxt('true_params/' + files[i].split('.')[0] + '.txt')
                break"""

    if (
        file == truths["lcname"]
    ):  # check that the data file and true params match
        print("Data file and true params match")
        return event_name, truths, data
    else:
        print("Data file and true params do not match")
        sys.exit()


def load_data(data_file):
    """load the data file.

    Notes:
    ------
    The lightcurve columns are:
        [0] Simulation_time
        [1] measured_relative_flux
        [2] measured_relative_flux_error
        [3] true_relative_flux
        [4] true_relative_flux_error
        [5] observatory_code
        [6] saturation_flag
        [7] best_single_lens_fit
        [8] parallax_shift_t
        [9] parallax_shift_u
        [10] BJD
        [11] source_x
        [12] source_y
        [13] lens1_x
        [14] lens1_y
        [15] lens2_x
        [16] lens2_y

    Magnitudes can be computed using:
        \[m = m_{source} + 2.5 log f_s - 2.5 log{F}\]
    where $F=fs*\mu + (1-fs)$ is the relative flux (in the file), $\mu$ is the magnification, and
        \[\sigma_m = 2.5/ln{10} \sigma_F/F.\]

    These are listed in the header information in lines #fs and #Obssrcmag with order matching the observatory code order.
    The observatory codes correspond to 0=W146, 1=Z087, 2=K213

    Bugs/issues/caveats:
    The output file columns list a limb darkening parameter of Gamma=0, it is actually Gamma=0.36 (in all filters)
    The orbit for the Z087 observatory appears to be different to the W146 and K213 observatory
    Dev is working on producing the ephemerides, but for single observatory parallax, using interpolated versions of the ones
    available for the data challenge will probably be accurate enough, or an Earth ephemeris with the semimajor axis (but
    not period) increased by 0.01 AU
    Lenses with masses smaller than the isochrone grid limits (I believe 0.1 MSun, will have filler values for magnitudes
    and lens stellar properties).
    There may be some spurious detections in the list where the single lens fit failed. Please let dev know if you find any
    of these events so that we can improve the single lens fitter."""

    data = pd.read_csv(data_file, header=0, delimiter=" ")
    data = data.to_numpy()

    return data


def get_params(master_file, EventID, SubRun, Field):
    """get the true params for the event"""

    master = pd.read_csv(master_file, header=0, delimiter=",")
    master_array = master.to_numpy()

    for i in range(len(master)):
        if (
            master_array[i][0] == int(EventID)
            and master_array[i][1] == int(SubRun)
            and master_array[i][2] == int(Field)
        ):
            row = i * 1
            break

    truths = master.iloc[row]
    truths = truths.to_dict()
    s = truths["s"]
    q = truths["q"]
    rho = truths["rho"]
    u0 = truths["u0"]
    alpha = truths["alpha"] * np.pi / 180
    t0 = truths["t0"]
    tE = truths["tE"]
    piEE = truths["piEE"]
    piEN = truths["piEN"]
    i = truths["i"] * np.pi / 180
    phase = truths["phase"] * np.pi / 180
    period = truths["period"]
    truths["params"] = [
        s,
        q,
        rho,
        u0,
        alpha,
        t0,
        tE,
        piEE,
        piEN,
        i,
        phase,
        period,
    ]

    return truths


def corner_post(res, event_name, path, truths):
    labels = [
        "s",
        "q",
        "rho",
        "u0",
        "alpha",
        "t0",
        "tE",
        "piEE",
        "piEN",
        "i",
        "phase",
        "period",
    ]
    fig = corner.corner(res.samples, labels=labels, truths=truths["params"])
    plt.title(event_name)

    fig.savefig(path + "posteriors/" + event_name + "_corner.png")
    plt.close(fig)


def runplot(res, event_name, path):
    fig, _ = dyplot.runplot(res)
    plt.title(event_name)

    fig.savefig(path + "posteriors/" + event_name + "_runplot.png")
    plt.close(fig)


def traceplot(res, event_name, path, truths):
    labels = [
        "s",
        "q",
        "rho",
        "u0",
        "alpha",
        "t0",
        "tE",
        "piEE",
        "piEN",
        "i",
        "phase",
        "period",
    ]
    fig, _ = dyplot.traceplot(
        res,
        truths=np.array(truths["params"]),
        truth_color="black",
        show_titles=True,
        trace_cmap="viridis",
        connect=True,
        connect_highlight=range(5),
        labels=labels,
    )
    plt.title(event_name)

    fig.savefig(path + "posteriors/" + event_name + "_traceplot.png")
    plt.close(fig)


if __name__ == "__main__":
    nevents = int(sys.argv[1])

    path = sys.argv[2]  # put some error handling in

    if len(sys.argv) == 3:
        sort = sys.argv[3]
    else:
        sort = "alphanumeric"
    ndim = 12

    orbit = Orbit()

    for i in range(nevents):

        if not os.path.exists(
            path + "posteriors/"
        ):  # make a directory for the posteriors
            os.mkdir(path + "posteriors/")

        event_name, truths, data = new_event(path, sort)
        parallax = Parallax(truths["ra"], truths["dec"], orbit)
        event = Event(parallax, orbit, data, truths["params"])

        sampler = dynesty.NestedSampler(
            lnprob,
            prior_transform,
            ndim,
            nlive=200,
            sample="rwalk",
            bound="multi",
            pool=mp.Pool(mp.cpu_count()),
            args=[event],
        )  # 'rwalk' is best for 10 < ndim < 20
        sampler.run_nested(
            maxiter=1000,
            checkpoint_file=path + "posteriors/" + event_name + ".save",
        )

        # Save the sampler as a pickle file
        with open(
            path + "posteriors/" + event_name + "_sampler.pkl", "wb"
        ) as f:
            pickle.dump(sampler, f)

        res = sampler.results

        # print for logs
        print("Event", i, "(", event_name, ") is done")
        print(res.summary())

        # Save plots
        corner_post(res, event_name, path, truths)
        runplot(res, event_name, path)
        traceplot(res, event_name, path, truths)

        samples = res.samples
        np.save(
            path + "posteriors/" + event_name + "_post_samples.npy", samples
        )

        # Done with the event
        complete_list = np.loadtxt(path + "complete.txt")
        complete_list = np.vstack(event_name)
        np.savetxt(path + "complete.txt", complete_list, fmt="%s")
        sampler.reset()

# ephermeris
# 2018-08-10
# 2023-04-30
# get emphemeris file from JPL Horizons for SEMB-L2
