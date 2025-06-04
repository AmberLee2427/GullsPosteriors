"""Event model including parallax and orbital motion."""

# import multiprocessing
# multiprocessing.set_start_method('fork', force=True)
import numpy as np

from Parallax import Parallax
from Orbit import Orbit


class Event:
    """Microlensing event combining data, parallax and orbital effects."""

    from ._VBM import magnification

    def __init__(
        self,
        parallax: Parallax,
        orbit: Orbit,
        data: dict,
        truths: dict,
        t_start: float,
        t_ref: float,
        eps=1e-4,
        gamma=0.36,
        LOM_enabled=True,
    ):
        """Instantiate a microlensing event model.

        Parameters
        ----------
        parallax : Parallax
            Object providing parallax calculations.
        orbit : Orbit
            Ephemerides used to locate the observatory.
        data : dict
            Light curve data keyed by observatory code.
        truths : dict
            Ground truth parameters of the event.  Must contain a ``params``
            array used to initialise the model.
        t_start : float
            Simulation start time in Barycentric Julian Date.
        t_ref : float
            Reference time defining the orientation of the parallax frame.
        eps : float, optional
            Numerical precision for the magnification calculation.
        gamma : float, optional
            Linear limb darkening coefficient.
        LOM_enabled : bool, optional
            If ``True`` include lens orbital motion parameters in the model.

        Attributes
        ----------
        orbit : Orbit
            Copy of the provided orbit object.
        parallax : Parallax
            Parallax helper configured with ``t_ref``.
        data : dict
            Stored light curve data.
        t_ref : float
            Reference epoch for parallax calculations.
        sim_time0 : float
            Simulation start time in BJD.
        truths : dict
            Truth parameters for the event.
        true_params : array_like
            Parameter array extracted from ``truths``.
        params : array_like
            Current working parameter array.
        eps : float
            Precision used by :mod:`_VBM` for integrations.
        gamma : float
            Linear limb darkening coefficient.
        mag_obj : object or None
            Backend magnification calculator instance.
        LOM_enabled : bool
            Flag controlling orbital motion physics.
        traj_base_tau, traj_parallax_tau, traj_base_beta, traj_parallax_beta
            Dictionaries used to store trajectory components for each
            observatory.
        traj_base_u1, traj_base_u2, traj_parallax_u1, traj_parallax_u2
            More trajectory diagnostics keyed by observatory.
        traj_parallax_dalpha_u1, traj_parallax_dalpha_u2
            Rotated trajectory coordinates.
        ss, tau, dalpha : dict
            Storage for separation, scaled time and rotation of the source
            trajectory.
        """

        self.orbit = orbit
        self.parallax = parallax
        self.parallax.set_ref_frame(t_ref)
        self.data = data
        self.t_ref = t_ref * 1.0
        self.sim_time0 = t_start * 1.0
        # truths = self.tref2t0(truths)
        self.truths = truths
        self.true_params = truths["params"]
        self.params = (
            self.true_params.copy()
        )  # don't want to edit true params when we edit params
        # print('debug Event.__init__: sim_time0: ', self.sim_time0)
        # print('debug Event.__init__: t_ref: ', self.t_ref)
        self.eps = eps
        self.gamma = gamma
        self.mag_obj = None

        self.LOM_enabled = LOM_enabled  # NEW: store the flag
        self.traj_base_tau = {}
        self.traj_parallax_tau = {}
        self.traj_base_beta = {}
        self.traj_parallax_beta = {}
        self.traj_base_u1 = {}
        self.traj_base_u2 = {}
        self.traj_parallax_u1 = {}
        self.traj_parallax_u2 = {}
        self.traj_parallax_dalpha_u1 = {}
        self.traj_parallax_dalpha_u2 = {}

        self.ss = {}
        self.tau = {}
        self.dalpha = {}

    def set_params(self, params):
        """Update the event parameters.

        Parameters
        ----------
        params : array_like
            Array of model parameters in the order ``[s, q, rho, u0, alpha,``
            ``t0, tE, piEN, piEE, i, phase, period]``.  The North and East
            components of the microlensing parallax are located at indices 7
            (``piEN``) and 8 (``piEE``).

        Notes
        -----
        The parameter array is stored on the instance and the parallax
        components are forwarded to :meth:`Parallax.update_piE_NE` to keep the
        parallax object in sync with the current model parameters.
        """

        self.params = params
        piEN = params[7]
        piEE = params[8]
        self.parallax.update_piE_NE(piEN, piEE)

    def projected_separation(
        self, i, period, t, phase_offset=0.0, t_start=None, a=1.0
    ):
        """Calculate the projected separation of the binary lens.

        Parameters
        ----------
        i : float or array_like
            Inclination of the orbit in radians.
        period : float or array_like
            Orbital period. If ``t`` is measured in units of ``tE`` then
            ``period`` must be as well.
        t : float or array_like
            Time of evaluation.  If ``t`` is given in units of ``tE`` it should
            be measured relative to ``t0``.
        phase_offset : float, optional
            Phase at ``t_start`` in radians. Default is ``0.0``.
        t_start : float, optional
            Reference start time for the orbit. Defaults to ``self.t_ref``.
        a : float, optional
            Semimajor axis in units of ``theta_E``. Default is ``1.0``.

        Returns
        -------
        s : float or ndarray
            Projected separation of the two components in the plane of the
            sky.
        x : float or ndarray
            X-coordinate of the secondary relative to the primary.
        y : float or ndarray
            Y-coordinate of the secondary relative to the primary.
        """

        if t_start is None:
            t_start = self.t_ref

        phase = 2 * np.pi * (t - t_start) / period + phase_offset
        # phase = phase % (2 * np.pi)
        x = a * np.cos(phase)
        y = a * np.sin(phase) * np.cos(i)
        z = a * np.sin(phase) * np.sin(i)

        # if an inclination of 0 means birds eye view, then use x, y
        s = np.sqrt(x**2 + y**2)

        # if an inclination of 0 means edge on view, then use x, z
        # s = np.sqrt(x**2 + z**2)

        return s, x, y

    def tref2t0(self, truths):
        """convert the ref time from tcroin to t0 for the lightcurve params"""

        tcroin = truths["tcroin"]
        t0 = truths["params"][5].copy()  # t0lens1 (random)'''

        # Convert to COM origin

        # convert to t0 reference time
        alpha = truths["params"][4].copy()  # 'alpha' * pi / 180 (random)
        s = truths["params"][0].copy()  # 'Planet_s'
        q = truths["params"][1].copy()  # 'q'
        period = truths["params"][11].copy()  # 'Planet_orbperiod' * 365.25
        tE = truths["params"][6].copy()  # tE_ref
        phase0 = truths["params"][
            10
        ].copy()  # 'Planet_phase' * pi / 180 at tcroin
        t_start = self.sim_time0
        i = truths["params"][9].copy()  # 'Planet_inclination' * pi / 180

        a = truths["Planet_semimajoraxis"]
        # truths['xCoM'] = s * q/(1.0+q)

        # redefining s is always at time t0

        # O at L1 to CoM
        delta_x0_com = s * q / (1.0 + q)
        truths["xCoM"] = delta_x0_com
        # print('\ndebug Event.tref2t0: delta_x0_com: ', delta_x0_com)

        # I think alpha is defined at t0, always

        # phase ofset at time t0
        phase0_t0_L1 = phase0 + (t0 - self.t_ref) * 2.0 * np.pi / period
        # print('\ndebug Event.tref2t0: phase0_t0_L1: ', phase0_t0_L1)

        u0 = truths["params"][
            3
        ].copy()  # 'u0lens1' closest approach to L1 (random)
        # at t0 with L1 at the origin
        piEE = truths["params"][7].copy()  # 'piEE'
        piEN = truths["params"][8].copy()  # 'piEN'
        rho = truths["params"][2].copy()  # 'rho'

        """ the scale in time is wrong here
        x0 = t0 * np.cos(alpha) - u0 * np.sin(alpha)
        x0_com = x0 - delta_x0_com
        y0 = t0 * np.sin(alpha) + u0 * np.cos(alpha)
        y0_com = y0 - delta_x0_com * np.tan(alpha)
        delta_t0_com = np.sqrt(
            delta_x0_com**2 + (delta_x0_com * np.tan(alpha)) ** 2
        )
        t0_com = t0 - delta_t0_com
        delta_u0_com = delta_t0_com * np.tan(alpha)
        u0_com = u0 - delta_u0_com #"""

        # u0, t0, and source position at t0 (x0, y0)
        x0 = -u0 * np.sin(alpha)  # because t0 = 0 in the tau scale
        y0 = u0 * np.cos(alpha)
        x0_com = x0 - delta_x0_com
        y0_com = y0 - delta_x0_com * np.tan(alpha)
        delta_tau_com = np.sqrt(
            delta_x0_com**2 + (delta_x0_com * np.tan(alpha)) ** 2
        )
        t0_com = t0 - delta_tau_com * tE
        # print('\ndebug Event.tref2t0: t0_com: ', t0_com)
        delta_u0_com = delta_tau_com * np.tan(alpha)
        u0_com = u0 - delta_u0_com
        # print('debug Event.tref2t0: u0_com: ', u0_com)
        # this is ignoring orital motion

        _, sx_ref, sy_ref = self.projected_separation(
            i, period, self.t_ref, phase_offset=phase0, t_start=self.t_ref
        )
        s_, sx_t0, sy_t0 = self.projected_separation(
            i, period, t0, phase_offset=phase0, t_start=self.t_ref
        )
        _, sx_t0_com, sy_t0_com = self.projected_separation(
            i, period, t0_com, phase_offset=phase0, t_start=self.t_ref
        )

        a = s / s_
        rot_ref = np.arctan2(
            sy_ref, sx_ref
        )  # again, the size of Sx and Sy are nonsense, but the ratio is valid
        rot_t0_L1 = np.arctan2(sy_t0, sx_t0)
        rot_t0_com = np.arctan2(sy_t0_com, sx_t0_com)
        # print('\ndebug Event.tref2t0: rot_ref: ', rot_ref)
        # entirely due to the phase offset
        # print('debug Event.tref2t0: rot_t0_L1: ', rot_t0_L1)
        # print('debug Event.tref2t0: rot_t0_com: ', rot_t0_com)
        rot = rot_t0_L1 - rot_ref
        rot_t0_com = rot_t0_com - rot_ref
        # print('debug Event.tref2t0: rot_t0_com: ', rot_t0_com)
        # print('debug Event.tref2t0: rot: ', rot)

        #  have no clue if this is right and don't wan to check itß
        x0_com_rot = x0_com * np.cos(rot_t0_com) - y0_com * np.sin(rot_t0_com)
        y0_com_rot = x0_com * np.sin(rot_t0_com) + y0_com * np.cos(rot_t0_com)
        # print('\ndebug Event.tref2t0: alpha_gulls: ', alpha)
        alpha_t0_com = np.arctan2(
            y0_com_rot, x0_com_rot
        )  # alpha changes because t0 changes
        # print('debug Event.tref2t0: alpha_t0_com: ', alpha_t0_com)
        phase0_t0_com = phase0 + (t0_com - self.t_ref) * 2.0 * np.pi / period

        s_com, _, _ = self.projected_separation(
            i, period, t0_com, phase_offset=phase0, t_start=self.t_ref
        )

        truths["params_t0_L1"] = [
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
            phase0_t0_L1,
            period,
        ]
        # ß
        truths["params_t0_COM"] = [
            s_com,
            q,
            rho,
            u0_com,
            alpha_t0_com,
            t0_com,
            tE,
            piEE,
            piEN,
            i,
            phase0_t0_com,
            period,
        ]
        # print(
        #     'debug Event.tref2t0: truths[params_t0_L1]: ',
        #     truths['params_t0_L1'],
        # )
        # print(
        #     'debug Event.tref2t0: truths[params_t0_COM]: ',
        #     truths['params_t0_COM'],
        # )

        return truths

    def croin(
        self, t0: float, u0: float, s: float, q: float, alpha: float, tE: float
    ):
        """Return the epoch when the source crosses the lens centre of mass.

        Parameters
        ----------
        t0 : float
            Time of closest approach to the primary lens in Barycentric Julian
            Date.
        u0 : float
            Impact parameter relative to the primary lens.
        s : float
            Lens separation at ``t0`` in units of ``theta_E``.
        q : float
            Mass ratio of the binary lens ``m2/m1``.
        alpha : float
            Source trajectory angle in radians.
        tE : float
            Einstein crossing time of the event.

        Returns
        -------
        float
            Computed ``tcroin`` in Barycentric Julian Date.
        """

        sina = np.sin(alpha)
        cosa = np.cos(alpha)

        if s < 0.1:
            tc = t0
            uc = u0
            rc = np.nan

        else:
            yc = 0.0

            if s < 1.0:
                xc = (s - 1.0 / s) * (1 - q) / (1 + q)
                rc = 2.0 * np.sqrt(q) / (s * np.sqrt(1.0 + s * s))
            else:
                xc = (s - 1.0 / (s * (1.0 + q))) - q * s / (1.0 + q)
                rc = 2.0 * np.sqrt(q)

            xc += s * q / (1 + q)  # shift to CoM
            rc *= 2.0 + np.min([45 * s * s, 80 / s / s])

            tc = t0 + tE * (xc * cosa + yc * sina)  # yc = 0
            uc = u0 + (xc * sina - yc * cosa) / rc  # yc = 0

        print("debug Event.croin: uc:", uc, self.truths["ucroin"])
        print("debug Event.croin: tc:", tc, self.truths["tcroin"])
        print("debug Event.croin: rc:", tc, self.truths["rcroin"])

        self.truths["ucroin_calc"] = uc
        self.truths["tcroin_calc"] = tc
        self.truths["rcroin_calc"] = rc

        return tc

    def get_magnification(self, t, obs=None, pp=None):
        """Return the magnification for a set of epochs.

        Parameters
        ----------
        t : float or array_like
            Times of evaluation in Barycentric Julian Date.
        obs : str, optional
            Observatory code used when storing trajectory diagnostics. If
            ``None`` no observatory specific information is kept.
        pp : array_like, optional
            Parameter vector overriding ``self.params`` when provided.

        Returns
        -------
        ndarray
            Magnification values corresponding to ``t``.
        """

        if pp is None:
            p = self.params.copy()
        else:
            p = pp.copy()

        q = p[1]  # mass ratio - not dependent on reference frame
        t0 = p[5]  # time of event peak in BJD
        # t_start = self.sim_time0  # I shouldn't need this
        tE = p[6]  # Einstein crossing time/event timescale
        t_ref = self.t_ref.copy()  # reference time for the parallax and LOM
        # tau_start = (t_start-t0)/tE
        # tcroin = self.truths['tcroin']
        u0 = p[3]  # impact parameter relative to L1
        alpha = p[4]  # angle of the source trajectory
        s = p[0]  # angular lens seperation at time t0

        tau = (t - t0) / tE  # event time in units of tE (tt) relative to t0
        # where t0 is the time of closest approach to L1

        # --- NEW: Conditional LOM physics ---
        if self.LOM_enabled:
            # Orbital parameters are unpacked only if needed
            i = p[9]  # inclination of the orbit
            phase0 = p[10]  # phase at time tcroin
            period = p[11]  # planet orbital period

            # Semimajor axis in units of thetaE
            s_ref, _, _ = self.projected_separation(
                i, period, 0.0, phase_offset=phase0, t_start=0.0
            )
            a = s / (s_ref)  # angular semimajor axis in uits of thetaE
            # print('debug Event.get_magnification: a/rE:',
            #      self.truths['Planet_semimajoraxis']/self.truths['rE'], a)

            # COM frame
            a1 = q / (1.0 + q) * a
            a2 = a - a1

            s1 = q / (1.0 + q) * s  # L1 to CoM offset
            s2 = s_ref - s1
            self.lens1_0 = np.array([-s1, 0.0])  # lens 1 position at time tref
            self.lens2_0 = np.array([s2, 0.0])  # lens 2 position at time tref

            # Notes:
            # m1a1 = m2a2  (a is 'distance' from COM)
            # a1 + a2 = a
            # m1/m2 = q     (q<1, and m2<m1)
            # a2 = a1/q
            # a = a1 + a1/q = a1(1+1/q) = a1(q+1)/q
            # a1 = aq/(1+q)

            # Orbital motion - dsdt - COM frame
            # print('debug Event.get_magnification: i: ', i)
            # print('debug Event.get_magnification: period: ', period)
            # print('debug Event.get_magnification: period(tau): ', period/tE)
            # print('debug Event.get_magnification: tau: ', tau)
            # print('debug Event.get_magnification: phase0: ', phase0)
            # print(
            #     'debug Event.get_magnification: phase0 - 180: ',
            #     phase0 + np.pi,
            # )
            # print('debug Event.get_magnification: a1: ', a1)
            # print('debug Event.get_magnification: a2: ', a2)

            # phase is always defined at the fixed time t_ref
            # this coordinate system is defined at time t_ref with a COM origin
            # let ss be the array of angular lens seperations at each epoch
            # ss does not have a reference frame
            ss1, x1, y1 = self.projected_separation(
                i, period, t, phase_offset=phase0 + np.pi, a=a1, t_start=t_ref
            )  # star - L1
            ss2, x2, y2 = self.projected_separation(
                i, period, t, phase_offset=phase0, a=a2, t_start=t_ref
            )  # planet - L2
            ss = np.sqrt(
                (x2 - x1) ** 2 + (y2 - y1) ** 2
            )
            # i don't know that this is strictly necessary since they are
            # always opposite
            self.ss[obs] = ss  # saving these for bug testing
            # print('\ndebug Event.get_magnification: s: \n',
            #      ss, ss.shape, '\n',
            #      ss1+ss2, (ss1+ss2).shape
            #      )  # these should be equal, I think

            # Orbital motion - dalphadt
            rot = np.arctan2(y2, x2)
            # positions of the planet in the COM-origin, planet-rotation frame
            # what is the reference time for this? - currently tref
            _, x0, y0 = self.projected_separation(
                i, period, 0.0, t_start=0.0, phase_offset=phase0, a=a
            )
            rot0 = np.arctan2(y0, x0)  # at reference time
            # the x, y positions are nonsense without a/a1/a2, but their ratios
            # are valid for the rotation either way
            self.dalpha[obs] = rot - rot0  # saving for debugging
            # print('debug Event.get_magnification: rot: ', rot)
            # print('debug Event.get_magnification: rot0: ', rot0)
            # print(
            #     'debug Event.get_magnification: dalpha: ',
            #     self.dalpha[obs],
            # )

        else:
            # If LOM is disabled, separation is constant and there is no
            # rotation
            ss = np.ones_like(t) * s  # Separation is just 's'
            self.ss[obs] = ss
            self.dalpha[obs] = np.zeros_like(t)  # No orbital rotation

            # --- ADD THESE LINES ---
            # Define static lens positions for plotting
            s1 = q / (1.0 + q) * s
            s2 = s - s1
            self.lens1_0 = np.array([-s1, 0.0])
            self.lens2_0 = np.array([s2, 0.0])

        # --- End of conditional LOM physics ---

        # umin = min(umin, np.sqrt(tt**2, uu**2))
        # gulls uses this value for weights stuff
        self.tau[obs] = tau

        # 'source' trajectory with parallax
        piEE = p[7]
        piEN = p[8]
        self.parallax.update_piE_NE(piEN, piEE)
        Delta_tau, Delta_beta = self.parallax.parallax_shift(
            t, obs
        )  # offset vectors due to parallax
        # t needs to be in HJD

        # print('\ndebug Event.get_magnification: Delta_tau: \n',
        #      Delta_tau, Delta_tau.shape
        #      )
        # print('\ndebug Event.get_magnification: Delta_beta: \n',
        #      Delta_beta, Delta_beta.shape
        #      )

        tt = tau + Delta_tau  # scaled time with parallax
        self.traj_base_tau[obs] = tau
        self.traj_parallax_tau[obs] = tt
        # uu = np.sqrt(u0**2 + tau**2)

        beta = np.ones_like(tt) * u0  # L1 impact parameter without parallax
        uu = beta + Delta_beta  # L1 impact parameter with parallax
        self.traj_base_beta[obs] = beta
        self.traj_parallax_beta[obs] = uu

        cosalpha = np.cos(
            alpha
        )  # have to do this rotation before the CoM shift
        sinalpha = np.sin(
            alpha
        )  # because s, u0, t0, and alpha are defined relative
        # to L1 closeset approach

        delta_x0_com = s * q / (1.0 + q)
        self.truths["xCoM"] = delta_x0_com

        # - s1 moves to CoM
        xsin = (
            tt * cosalpha - uu * sinalpha - self.truths["xCoM"]
        )  # subtracting L1-COM distance
        # at time t0
        self.traj_parallax_u1[obs] = xsin  # saving for debugging
        self.traj_base_u1[obs] = (
            tau * cosalpha - beta * sinalpha - delta_x0_com
        )  # sans parallax, for debug

        ysin = (
            tt * sinalpha + uu * cosalpha
        )  # no y-shift because lens axis is fixed
        self.traj_parallax_u2[obs] = ysin  # saving for debugging
        self.traj_base_u2[obs] = (
            tau * sinalpha + beta * cosalpha
        )  # sans parallax, for debug

        cosrot = np.cos(-self.dalpha[obs])  # gulls has -dalpha
        sinrot = np.sin(-self.dalpha[obs])

        # orbital motion rotation
        xsrot = xsin * cosrot - ysin * sinrot
        ysrot = xsin * sinrot + ysin * cosrot
        self.traj_parallax_dalpha_u1[obs] = xsrot
        self.traj_parallax_dalpha_u2[obs] = ysrot

        rho = self.params[2].copy()  # source radius in units of thetaE

        # print('\ndebug Event.get_magnification: q: \n',
        #      q, q.shape
        #      )
        # print('\ndebug Event.get_magnification: xsrot: \n',
        #      xsrot, xsrot.shape
        #      )
        # print('\ndebug Event.get_magnification: ysrot: \n',
        #      ysrot, ysrot.shape
        #      )

        # print('Debug Event.get_magnification: ss', ss, type(ss), ss.shape)
        # print(
        #     'Debug Event.get_magnification: ss1',
        #     ss1,
        #     type(ss1),
        #     ss1.shape,
        # )
        # print(
        #     'Debug Event.get_magnification: ss2',
        #     ss2,
        #     type(ss2),
        #     ss2.shape,
        # )
        # print('Debug Event.get_magnification: x1', x1, type(x1), x1.shape)
        # print('Debug Event.get_magnification: x2', x2, type(x2), x2.shape)
        # print('Debug Event.get_magnification: y1', y1, type(y1), y1.shape)
        # print('Debug Event.get_magnification: y2', y2, type(y2), y2.shape)
        # print('Debug Event.get_magnification: period', period)
        # print('Debug Event.get_magnification: i', i)
        # print('Debug Event.get_magnification: phase0', phase0)
        # print('Debug Event.get_magnification: tref', t_ref)
        # print('Debug Event.get_magnification: t', t)

        A = self.magnification(
            ss, q, xsrot, ysrot, rho, eps=self.eps, gamma=self.gamma
        )

        # vbbl has CoM O

        # ss is an array that accounts for changing s due to OM
        # xsrot and ysrot have been shifted and rotated to account for parallax
        # and OM
        # gamma is the limb darkening coefficient, which is fixed for the
        # gulls run
        # eps is the precision of the numerical integration
        # vbbl.BinaryMagDark only takes floats

        return A
