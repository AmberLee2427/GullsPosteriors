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
import warnings
import pickle

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

# probably dynesty would make more sense for posterior sampling because it will have z values.

# rE in AU (Einstein radius)
# Event->rE = rEsun * sqrt(Lenses->data[ln][MASS] * Sources->data[sn][DIST] * (1-x) * x);

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
# Event->rs = (Sources->data[sn][RADIUS] * Rsun / Sources->data[sn][DIST]) / Event->thE;
# radius (Rsun) -> AU / Ds (kpc) -> mas / thetaE (mas) = ratio

# rate weighting
# Event->raww = Event->thE * Event->murel;

class Orbit:

    def __init__(self, 
                 obs_location='SEMB-L2', 
                 start_date='2018-08-10', 
                 end_date='2023-05-30', 
                 refplane='ecliptic', 
                 n_epochs=None,
                 origin='500@10',
                 date_format='iso'):
        '''Position file from JPL Horizons'''

        self.start_time = Time(start_date, format=date_format)
        self.end_time = Time(end_date, format=date_format)

        if n_epochs is None:
            self.n_epochs = int(self.end_time.jd - self.start_time.jd + 1)  # 1 epoch per day
        else:
            self.n_epochs = n_epochs

        self.obs_location = obs_location
        self.origin = origin
        self.refplane = refplane

        self.epochs, self.positions, self.velocities = self.fetch_horizons_data()

    def fetch_horizons_data(self):
        times = np.linspace(self.start_time.jd, self.end_time.jd, self.n_epochs)
        times = Time(times, format='jd')

        positions_list = []
        velocities_list = []

        # Split the times into chunks to avoid hitting the API limits
        chunk_size = 75  # Adjust this based on the API limits
        for i in range(0, len(times), chunk_size):
            chunk_times = times[i:i + chunk_size]
            q = Horizons(id=self.obs_location, location=self.origin, epochs=chunk_times.jd)
            data = q.vectors(refplane=self.refplane)

            positions_list.append(CartesianRepresentation(data['x'], data['y'], data['z']))
            velocities_list.append(CartesianDifferential(data['vx'], data['vy'], data['vz']))

        # Combine the chunks into single arrays
        positions = CartesianRepresentation(
            np.concatenate([pos.x for pos in positions_list]),
            np.concatenate([pos.y for pos in positions_list]),
            np.concatenate([pos.z for pos in positions_list])
        )
        velocities = CartesianDifferential(
            np.concatenate([vel.d_x for vel in velocities_list]),
            np.concatenate([vel.d_y for vel in velocities_list]),
            np.concatenate([vel.d_z for vel in velocities_list])
        )

        return times, positions, velocities

    def get_pos(self, t):
        '''get the position of the observatory at time t by interpolating the position file'''
        t = Time(t, format='jd')
        x_interp = interp1d(self.epochs.jd, self.positions.x.to(u.au).value)
        y_interp = interp1d(self.epochs.jd, self.positions.y.to(u.au).value)
        z_interp = interp1d(self.epochs.jd, self.positions.z.to(u.au).value)
        x = x_interp(t.jd)
        y = y_interp(t.jd)
        z = z_interp(t.jd)
        xyz = np.vstack((x, y, z))
        return xyz
    
    def get_vel(self, t):
        '''get the velocity of the observatory at time t by interpolating the position file'''
        vx_interp = interp1d(self.epochs.jd, self.velocities.d_x.to(u.au / u.day).value)
        vy_interp = interp1d(self.epochs.jd, self.velocities.d_y.to(u.au / u.day).value)
        vz_interp = interp1d(self.epochs.jd, self.velocities.d_z.to(u.au / u.day).value)
        vx = vx_interp(t)
        vy = vy_interp(t)
        vz = vz_interp(t)
        vxyz = np.vstack((vx, vy, vz)) * (u.au / u.day)
        return vxyz
    

class Parallax:

    def __init__(self, 
                 ra, 
                 dec, 
                 orbit, 
                 t_ref, 
                 tu=None, 
                 piE=None, 
                 epochs=None
                 ):
        self.ra = ra * u.deg
        self.dec = dec * u.deg
        self.event_coords = astrocoords.SkyCoord(ra=self.ra, dec=self.dec, frame='icrs')
        self.orbit = orbit
        self.set_ref_frame(t_ref)
        self.rotate_view()

        if tu is not None:
            self.tu = tu
            self.epochs = epochs
            ne = {}
            for obs in tu.keys():
                ne[obs] = self.tu2ne(tu[obs], piE)
                #print('observatory key:', obs)
                #print('ne', ne[obs], ne[obs].shape)
                #print('tu', tu[obs], tu[obs].shape)
            self.ne = ne
        else:
            self.tu = None
            self.ne = None
            
        # Event->pllx[obsidx].reset();
	    # Event->pllx[obsidx].set_reference(Paramfile->simulation_zerotime+tref,&World[0].orbit);
	    # Event->pllx[obsidx].set_orbit(&World[obsidx].orbit);
	    # Event->pllx[obsidx].set_lb(Event->l, Event->b);
	    # Event->pllx[obsidx].set_pm_lb(Lenses->data[ln][MUL]-Sources->data[sn][MUL],Lenses->data[ln][MUB]-Sources->data[sn][MUB]);
	    # Event->pllx[obsidx].set_piE(Event->piE);
	    # Event->pllx[obsidx].set_tE_h(Event->tE_h);  // tE in the heliocentric reference rame

    def set_ref_frame(self, t_ref):
        self.tref = t_ref  # bjd
        self.xref = self.orbit.get_pos(t_ref)
        self.vref = self.orbit.get_vel(t_ref)

    def update_piE_NE(self, piEN, piEE):
        '''update the parallax parameters'''
        self.piEN = piEN
        self.piEE = piEE
        
        self.piE = np.array([piEN, piEE])

    def dperp():
        '''calculate the perpendicular distance to the source'''
        pass

    def rotate_view(self):
        '''roates from x, y, z, to n, e, d.'''

        # unit vector pointing to the source
        self.rad = np.array([np.cos(self.ra.to(u.rad).value) * np.cos(self.dec.to(u.rad).value), 
                             np.sin(self.ra.to(u.rad).value) * np.cos(self.dec.to(u.rad).value), 
                             np.sin(self.dec.to(u.rad).value)])
        
        # north vector in x, y, z
        north = np.array([0, 0, 1])
        
        # unit vector ointing east in the lens plane
        e_unit = np.cross(north, self.rad)/np.linalg.norm(np.cross(north, self.rad))

        # unit vector pointing north in the lens plane
        n_unit = np.cross(self.rad, e_unit)

        # rotation matrix
        self.rot_matrix = np.array([n_unit, e_unit, self.rad])

        # rotate the reference values into the source pointing frame
        self.xref = np.dot(self.rot_matrix, self.xref)
        self.vref = np.dot(self.rot_matrix, self.vref)

    def counter_rotate(self, v, phi):
        '''counter rotate the vector v by phi'''
        cosp = np.cos(-phi)
        sinp = np.sin(-phi)
        return np.array([v[:,0]*cosp - v[:,1]*sinp, v[:,0]*sinp + v[:,1]*cosp])

    def tu2ne(self, tu, piE):
        '''convert the parallax shift in t, u to n, e'''
        piEN = piE[0]
        piEE = piE[1]
        phi_pi = np.arctan2(piEE, piEN)
        tu_ = -tu / np.linalg.norm(piE)
        en = self.counter_rotate(tu_, phi_pi)
        #print('en', en, en.shape)
        #print('e', en[0], en[0].shape)
        ne = np.array([en[1], en[0]]).T
        return ne

    def get_pos(self, t):
        '''get the position of the observatory at time t in the n, e, d frame'''
        xyz = self.orbit.get_pos(t)
        ned = np.dot(self.rot_matrix, xyz)
        return ned

    def parallax_shift(self, t, obs=None):
        r'''calculate the parallax shift.

        Notes
        -----
        Gould, A., 2004, ApJ, 606, 319

        Let s(t) be the Earth-to-Sun vector in units of AU in the heliocentric frame. 
        Let tp be some fixed time, in practice a time very close to the time t0 of 
        the peak of the event as seen from the Earth, and evaluate the derivative of 
        s(t) at this time,

        ..math::
        \Delta s(t) = s(t) - s(t_ref) = (t-t_ref) v_ref$$

        observations are toward an event at some given celestial coordinates 
        and define nˆ and eˆ as the unit vectors pointing north and east.

        ..math::
        (dtau, dbeta) = (s_n(t)\pi_{EN} + s_e(t)\pi_{EE}, -s_n(t)\pi_{EE} + s_e(t)\pi_{EN})$$
        '''
        NE = {}
        tt = {}

        if type(t) == np.ndarray and obs is None:  # all observatories, but one t set
            keys = self.tu.keys()
            for obs in keys:
                tt[obs] = t

        elif type(t) == np.ndarray and obs is not None:  # single observatory
            tt[obs] = t
            keys = [obs]

        else:  # all observatories, all t sets
            keys = t.keys()
            tt = t  # t is a dictionary

        t = tt

        for obs in keys:
            tt = t[obs]


            NEshift = [0, 0]

            if self.tu is None:
                x = self.get_pos(tt)  # xyz heliocentric L2 position of the observatory at time t,
                                    # in the ecliptic plane?
                x = np.dot(self.rot_matrix, x)  # rotates the L2 position vector into the source-pointing
                                                # coordinate system

                #L2-Sun shift with n^ e^ defined when looking at the event
                NEshift[0] = x[0] - self.xref[0] - (tt-self.tref)*self.vref[0].value  # array the same size as t. n^ shift component
                NEshift[1] = x[1] - self.xref[1] - (tt-self.tref)*self.vref[1].value  # e^ shift component
                                                                                    # I think these assume that vref (x, y) are a good enough 
                                                                                    # approximation for the velocity of the observatory, throughout
                                                                                    # the event; i.e. the transverse velocity is assumed to be 
                                                                                    # constant. I'm not sure if this is a good assumption.
                                                                                    # I dunno. I'm not following Andy here.
                NE[obs] = np.array(NEshift)

            else:

                ne = self.ne[obs]  # lc ne shift at lc epochs
                print(tt.shape, self.epochs[obs].shape, ne.shape)
                NEshift[0] = np.interp(tt, self.epochs[obs], ne[:,0]) # n shift at t
                NEshift[1] = np.interp(tt, self.epochs[obs], ne[:,1]) # e shift at t

                NE[obs] = np.array(NEshift)

        # Michael's code (ground-based observer)
        #q_n = -S_n_arr[0] - vn0 * (ts_in - t_peak)   v is Earth's perpendicular velocity
	    #q_e = -S_e_arr[0] - ve0 * (ts_in - t_peak)   I think he has north and east in the opposite direction to Matt
        #delta_tau = q_n*pi_EN + q_e*pi_EE
		#delta_beta = -q_n*pi_EE + q_e*pi_EN
        # ^ this method was introducting machine error

        # relative lens source proper motion angle in n^ e^ frame
        phi_pi = np.arctan2(self.piEE,self.piEN)

        cs = np.cos(phi_pi)
        sn= np.sin(phi_pi)

        #Convert the shift in the observer plane to a shift in the source position
        tu = {}
        for obs in NE.keys():
            tushift = np.zeros([2, len(NE[obs][0])])
            tushift[0] = -np.linalg.norm(self.piE) * ( NE[obs][0]*cs + NE[obs][1]*sn)  # Delta_tau - shift in the relative-motion direction
            tushift[1] = -np.linalg.norm(self.piE) * (-NE[obs][0]*sn + NE[obs][1]*cs)  # Delta_beta - shift perpendicular to the lens-source motion

            tu[obs] = tushift

        # ideally, this will cause a break if the t logic is wrong
        if len(tu.keys()) == 1:  # unpack from the dictionary if there is only one observatorty
            obs = list(tu.keys())[0]
            tu = tu[obs]

        return tu

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
        self.params = params
        piEN = params[7]
        piEE = params[8]
        self.parallax.update_piE_NE(piEN, piEE)

    def projected_seperation(self, i, period, t, phase_offset=0.0, a=1.0):
        '''Calculate the projected seperation of the binary'''

        # i is the inclination of the orbit in radians
        # period is the period of the orbit (if t is tau then period is in units of tE)
        # t is the time relative to t0 (t')
        # phase_offset is the phase at t0
        # a is the semi-major axis of the orbit

        phase = 2 * np.pi * (t) / period + phase_offset
        phase = phase % (2 * np.pi)
        x = a * np.cos(-phase)
        y = a * np.sin(-phase) * np.cos(i)
        z = a * np.sin(-phase) * np.sin(i)

        # if an inclination of 0 means birds eye view, then use x, y
        s = np.sqrt(x**2 + y**2)

        # if an inclination of 0 means edge on view, then use x, z
        #s = np.sqrt(x**2 + z**2)

        return s, x, y 
    
    def get_magnification(self, t, obs=None):
        '''Calculate the magnification'''

        q = self.params[1].copy()
        s = self.params[0].copy()

        i = self.params[9].copy()  # inclination of the orbit
        phase0 = self.params[10].copy()  # phase at time t0
        period = self.params[11].copy()  # planet orbital period

        s0, x0, y0 = self.projected_seperation(i, period, 0.0, phase0) # star-centric ref frame
        a = s/s0  # semimajor axis in uits of thetaE

        # COM frame
        a1 = q/(1.0+q) * a
        a2 = a - a1

        s1 = q/(1.0+q) * s
        s2 = s - s1

        self.lens1_0 = np.array([-s1, 0.0])
        self.lens2_0 = np.array([s2, 0.0])

        # Notes:
        # m1a1 = m2a2  (a is 'distance' from COM) 
        # a1 + a2 = a
        # m1/m2 = q     (q<1, and m2<m1)
        # a2 = a1/q
        # a = a1 + a1/q = a1(1+1/q) = a1(q+1)/q
        # a1 = aq/(1+q)

        t0 = self.params[5].copy()  # time of event peak in BJD
        tE = self.params[6].copy()  # Einstein crossing time/event timescale
        tau = (t - t0) / tE  # event time in units of tE

        # Orbital motion - dsdt - COM frame
        print('debug Event.get_magnification: i: ', i)
        print('debug Event.get_magnification: period: ', period)
        print('debug Event.get_magnification: period(tau): ', period/tE)
        print('debug Event.get_magnification: tau: ', tau)
        print('debug Event.get_magnification: phase0: ', phase0)
        print('debug Event.get_magnification: phase0 - 180: ', phase0+np.pi)
        print('debug Event.get_magnification: a1: ', a1)
        print('debug Event.get_magnification: a2: ', a2)
        ss1, x1, y1 = self.projected_seperation(i, period/tE, tau, phase0+np.pi, a1)  # star
        ss2, x2, y2 = self.projected_seperation(i, period/tE, tau, phase0, a2)  # planet
        ss = np.sqrt((x2 - x1)**2+(y2 - y1)**2)  # i don't know that this is strictly necessary since they are always opposite
        self.ss[obs] = ss
        self.tau[obs] = tau
        #print('\ndebug Event.get_magnification: s: \n', 
        #      ss, ss.shape, '\n', 
        #      ss1+ss2, (ss1+ss2).shape
        #      )  # these should be equal, I think

        #umin = min(umin, np.sqrt(tt**2, uu**2))

        u0 = self.params[3].copy()  # impact parameter

        # 'source' trajectory with parallax
        piEE = self.params[7].copy()
        piEN = self.params[8].copy()
        self.parallax.update_piE_NE(piEN, piEE)
        Delta_tau, Delta_beta = self.parallax.parallax_shift(t, obs)  # offset vectors due to parallax
                                                                      # t needs to be in HJD
        
        #print('\ndebug Event.get_magnification: Delta_tau: \n', 
        #      Delta_tau, Delta_tau.shape
        #      )
        #print('\ndebug Event.get_magnification: Delta_beta: \n', 
        #      Delta_beta, Delta_beta.shape
        #      )

        tt = tau + Delta_tau  # scaled time with parallax
        self.traj_base_tau[obs] = tau
        self.traj_parallax_tau[obs] = tt
        #uu = np.sqrt(u0**2 + tau**2)

        beta = np.ones_like(tt) * u0  # impact parameter without parallax
        uu = beta + Delta_beta  # impact parameter with parallax
        self.traj_base_beta[obs] = beta
        self.traj_parallax_beta[obs] = uu

        alpha = self.params[4].copy() # in radians
        cosalpha = np.cos(alpha)
        sinalpha = np.sin(alpha)

        xsin = tt * cosalpha - uu * sinalpha
        self.traj_parallax_u1[obs] = xsin
        self.traj_base_u1[obs] = tau * cosalpha - beta * sinalpha

        ysin = tt * sinalpha + uu * cosalpha
        self.traj_parallax_u2[obs] = ysin
        self.traj_base_u2[obs] = tau * sinalpha + beta * cosalpha

        # Orbital motion - dalphadt
        rot = np.arctan2(y2, x2)  # position of the planet in the COM , planet rotation frame. 
                                  # y2, x2 are arrays
        rot0 = np.arctan2(y0, x0)  # y0, x0 are floats (at time
        self.dalpha[obs] = rot - rot0
        print('debug Event.get_magnification: rot: ', rot)
        print('debug Event.get_magnification: rot0: ', rot0)
        print('debug Event.get_magnification: dalpha: ', self.dalpha[obs])
        cosrot = np.cos(self.dalpha[obs])
        sinrot = np.sin(self.dalpha[obs])

        # orbital motion rotation
        xsrot = xsin * cosrot - ysin * sinrot
        ysrot = xsin * sinrot + ysin * cosrot
        self.traj_parallax_dalpha_u1[obs] = xsrot
        self.traj_parallax_dalpha_u2[obs] = ysrot

        rho = self.params[2].copy()  # source radius in units of thetaE

        #print('\ndebug Event.get_magnification: q: \n', 
        #      q, q.shape
        #      )
        #print('\ndebug Event.get_magnification: xsrot: \n', 
        #      xsrot, xsrot.shape
        #      )
        #print('\ndebug Event.get_magnification: ysrot: \n',
        #      ysrot, ysrot.shape
        #      )

        eps = 1.0e-5  # precision of the numerical integration
        gamma = 0.36
        vbbl = VBBinaryLensing()
        vbbl.a1 = gamma

        A = np.zeros_like(ss)
        
        for i, s in enumerate(ss):
            #print('\ndebug Event.get_magnification: i: ', i)
            A[i] = vbbl.BinaryMagDark(s, q, xsrot[i], ysrot[i], rho, eps)
        
        # ss is a vector to account for OM
        # xsrot and ysrot have been shifted and rotated to account for parallax and OM
        # gamma is the limb darkening coefficient, which is fixed for the gulls run
        # eps is the precision of the numerical integration
        # vbbl.BinaryMagDark only takes floats

        return A

class Fit:

    def __init__(self) -> None:
        pass

    def get_fluxes(self, model:np.ndarray, f:np.ndarray, sig2:np.ndarray):
        '''Solves for the flux parameters for a givin model using least squares.
        
        Parameters:
        -----------
        model: model magnification curve
        f: observed flux values
        sig2: flux errors.
        
        Returns:
        --------
        FS: source flux
        FB: blend flux.
        '''
        #A
        A11 = np.sum(model**2 / sig2)
        Adiag = np.sum(model / sig2) 
        A22 = np.sum(1.0 / sig2)
        A = np.array([[A11,Adiag], [Adiag, A22]])
        
        #C
        C1 = np.sum((f * model) / sig2)
        C2 = np.sum(f / sig2)
        C = np.array([C1, C2]).T
        
        #B
        B = np.linalg.solve(A,C)
        FS = float(B[0])
        FB = float(B[1])
        
        return FS, FB 

    def get_chi2(self, event, params):

        event.set_params(params)
        chi2sum = 0.0
        chi2 = {}
        
        for obs in event.data.keys():  # looping through observatories
            t = event.data[obs][0]  # BJD
            f = event.data[obs][1]  # obs_rel_flux
            f_err = event.data[obs][2]  # obs_rel_flux_err

            A = event.get_magnification(t, obs)
            fs, fb = self.get_fluxes(A, f, f_err**2)

            chi2[obs] = ((f - A*fs+fb) / f_err) ** 2

            chi2sum += np.sum(chi2[obs])

        return chi2, chi2sum

    def lnlike(self, theta, event):  

        _, chi2 = self.get_chi2(self, event, theta)
        return -0.5 * chi2

    def lnprior(self, theta, event, bound_penalty=False):
        s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period = theta
        sinphase = np.sin(phase)
        s0 = event.projected_seperation(i, period/tE, 0.0, phase)
        if tE > 0.0 and q <= 1.0 and period/tE > 4 and sinphase < 0.9 and sinphase >= 0.00 and i <= np.pi/2 and i >= 0 and s0 > 0.0:
        
            if bound_penalty:   # i'm not using this. I need to redo the calculation
                # calculate beta and requiere the orbits conserve energy
                #G_au = 1.0  # gravitational constant in AU^3 / (M1 * years^2)  / (2pi)^2
                #m1 = 1  # mass of the first object
                #ßm2 = m1*q  # mass of the second object
                #I1 = a1**2  # *m1
                #I2 = q * a2**2  # *m1
                #I = q * a**2  # fixed m1 frame
                #period = period/365.25 # convert period to years
                #w = 1.0/ period  # /2pi

                # calculate the gravitational potential energy
                # m1*m2 = m1*(m1*q) = m1^2*q
                # /m1: m1*q
                # /(2pi^)2
                #Eg = q / a

                # calculate the rotational kinetic energy
                # /m2
                # /(2pi^)2
                #Erot = 0.5 * (I1+I2) * w**2

                #bound_penatly = (Eg - Erot)**2
                bound_penatly = 0.0
            else:
                bound_penatly = 0.0

            return 0.0 + bound_penatly


        return -np.inf

    def lnprob(self, theta, event):
        # prior
        lp = self.lnprior(theta, event)
        if not np.isfinite(lp):
            return -np.inf
        
        # likelihood
        ll = self.lnlike(theta, event)
        if not np.isfinite(ll):
            return -np.inf
        
        # prob
        return lp + ll

    def prior_transform(self, u, truths):
        """Transform unit cube to the parameter space. Nested sampling has firm boundaries on the prior space."""
        logs, logq, logrho, u0, alpha, t0, tE, piEE, piEN, i, sinphase, logperiod = u

        logs_true = np.log(truths['params'][0])
        logq_true = np.log(truths['params'][1])
        logrho_true = np.log(truths['params'][2])
        u0_true = truths['params'][3]
        alpha_true = truths['params'][4]
        t0_true = truths['params'][5]
        tE_true = truths['params'][6]
        piEE_true = truths['params'][7]
        piEN_true = truths['params'][8]
        i_true = truths['params'][9]
        sinphase_true = np.sin(truths['params'][10])
        logperiod_true = np.log(truths['params'][11])

        # I think these ranges need to stay the same for the logz values to be comparable
        # check how big these uncertainties normally are and adjust the ranges accordingly
        logs_range = 0.5
        logq_range = 1.0  # this one might be poorly constrained
        logrho_range = 1.0  # this one might also be poorly constrained
        u0_range = 0.2
        alpha_range = 0.1
        t0_range = 0.5
        tE_range = 1.0
        piEE_range = 5.0  # these might not be big enough - I don't know how well constrain piE is
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

    def corner_post(self, res, event_name, path, truths):
        labels = ['s', 'q', 'rho', 'u0', 'alpha', 't0', 'tE', 'piEE', 'piEN', 'i', 'phase', 'period']
        fig = corner.corner(res.samples, labels=labels, truths=truths['params'])
        plt.title(event_name)

        fig.savefig(path+'posteriors/'+event_name+'_corner.png')
        plt.close(fig)

    def runplot(self, res, event_name, path):
        fig, _ = dyplot.runplot(res)
        plt.title(event_name)

        fig.savefig(path+'posteriors/'+event_name+'_runplot.png')
        plt.close(fig)

    def traceplot(self, res, event_name, path, truths):
        labels = ['s', 'q', 'rho', 'u0', 'alpha', 't0', 'tE', 'piEE', 'piEN', 'i', 'phase', 'period']
        fig, _ = dyplot.traceplot(res, 
                                    truths=np.array(truths['params']),
                                    truth_color='black', 
                                    show_titles=True,
                                    trace_cmap='viridis', 
                                    connect=True,
                                    connect_highlight=range(5), 
                                    labels=labels)
        plt.title(event_name)

        fig.savefig(path+'posteriors/'+event_name+'_traceplot.png')
        plt.close(fig)

class Data:

    def __init__(self):
        pass

    def new_event(self, path, sort='alphanumeric'):
        '''get the data and true params for the next event'''

        files = os.listdir(path)
        files = sorted(files)

        if path[-1] != '/':
            path = path + '/'

        if not os.path.exists(path+'run_list.txt'):  # if the run list doesn't exist, create it
            run_list = np.array([])
            np.savetxt(path+'run_list.txt', run_list, fmt='%s')

        if not os.path.exists(path+'complete.txt'):  # if the complete list doesn't exist, create it
            complete_list = np.array([])
            np.savetxt(path+'complete.txt', complete_list, fmt='%s')

        for file in files:
            if 'csv' in file:
                master_file = path + file

        if sort == 'alphanumeric':

            for file in files:

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    run_list = np.loadtxt(path+'run_list.txt', dtype=str)

                if (file not in run_list) and ('det.lc' in file):

                    print('Already ran:', run_list)
                    run_list = np.hstack([run_list, file])
                    print('Running:', file, type(file))
                    np.savetxt(path+'run_list.txt', run_list, fmt='%s')

                    lc_file_name = file.split('.')[0]
                    event_identifiers = lc_file_name.split('_')
                    EventID = event_identifiers[-1]
                    SubRun = event_identifiers[-3]  # the order of these is fucked up
                    Field = event_identifiers[-2]  # and this one. -A 2024-11-11 resample

                    data_file = path + file
                    
                    data = self.load_data(data_file)  # bjd, flux, flux_err, tshift, ushift
                    
                    event_name = f'{Field}_{SubRun}_{EventID}'
                    print('event_name = ', event_name)
                    
                    truths = self.get_params(master_file, EventID, SubRun, Field)  # make sure to turn all the degress to radians
                    break

        '''if ".txt" in sort:
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
                    break'''

        print()
        # this is fucking dumb, but the 'lcname's is the master file do not match the actual lc file names
        if file == truths['lcname']:  # check that the data file and true params match
            print('Data file and true params \'lcname\' match')
            sys.exit()
            #return event_name, truths, data
        else:
            print('Data file and true params \'lcname\' do not match')
            print(file, '!=', truths['lcname'])
            if len(file) != len(truths['lcname']):
                print('length:', len(file), '!=\n', len(truths['lcname']))
            return event_name, truths, data

    def load_data(self, data_file):
        r'''load the data file.
        
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
    
        ..math:
            m = m_{source} + 2.5 log f_s - 2.5 log{F}

        where :math:`F=fs*\mu + (1-fs)` is the relative flux (in the file), :math:`\mu` is the magnification, and
        
        ..math:
            \sigma_m = 2.5/ln{10} \sigma_F/F.

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
        of these events so that we can improve the single lens fitter.'''

        header = ['Simulation_time', 
                  'measured_relative_flux', 
                  'measured_relative_flux_error', 
                  'true_relative_flux', 
                  'true_relative_flux_error', 
                  'observatory_code', 
                  'saturation_flag', 
                  'best_single_lens_fit', 
                  'parallax_shift_t', 
                  'parallax_shift_u', 
                  'BJD', 
                  'source_x', 
                  'source_y', 
                  'lens1_x', 
                  'lens1_y', 
                  'lens2_x', 
                  'lens2_y',
                  'A',
                  'B',
                  'C'
                  ]

        data = pd.read_csv(data_file, sep=r'\s+', skiprows=12, names=header)  # delim_whitespace=True is the same as sep=r'\s+', but older.
                                                                              # The 'r' in sep=r'\s+' means raw string, which is not necessary.
                                                                              # Otherwise you get annoying warnings.
        
        # simulation time to BJD
        print(data['BJD'][0])
        print(data['Simulation_time'][0])

        self.sim_time0 = data['BJD'][0] - data['Simulation_time'][0]

        data = data[['BJD', 
                    'measured_relative_flux',
                    'measured_relative_flux_error',
                    'parallax_shift_t',
                    'parallax_shift_u', 'observatory_code'
                    ]]
        
        data_dict = {}
        for code in data['observatory_code'].unique():
            data_obs = data[data['observatory_code'] == code][['BJD', 
                                          'measured_relative_flux', 
                                          'measured_relative_flux_error', 
                                          'parallax_shift_t', 
                                          'parallax_shift_u']].reset_index(drop=True)
            data_dict[code] = data_obs.to_numpy().T

        return data_dict

    def get_params(self, master_file, EventID, SubRun, Field):
        '''get the true params for the event'''
        EventID = int(EventID)
        SubRun = int(SubRun)
        Field = int(Field)

        master = pd.read_csv(master_file, header=0, delimiter=',')
        #print(master.head())

        truths = master[(master['EventID'] == int(EventID)) & 
                        (master['SubRun'] == int(SubRun)) & 
                        (master['Field'] == int(Field))
                        ].iloc[0]

        print(self.sim_time0)
        print(truths)
        
        s = truths['Planet_s']
        q = truths['Planet_q']
        rho = truths['rho']
        u0 = truths['u0lens1']
        alpha = truths['alpha']*np.pi/180 # convert to radians
        truths['t0lens1'] = truths['t0lens1'] + self.sim_time0  # convert to BJD
        t0 = truths['t0lens1'] 
        tcroin = truths['tcroin'] + self.sim_time0  # convert to BJDs
        tE = truths['tE_ref']
        piEE = truths['piEE']
        piEN = truths['piEN']
        i = truths['Planet_inclination']*np.pi/180  # convert to radians
        phase = truths['Planet_orbphase']*np.pi/180  # convert to radians # centre on tcroin
        period = truths['Planet_period']*365.25  # convert to days
        phase_change = truths['tcroin'] / period
        phase = phase + phase_change  # centre on t0
        phase = phase % (2.0*np.pi)  # make sure it's between 0 and 2pi
        truths['params'] = [s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period]

        truths['tcroin'] = tcroin

        return truths
    

if __name__ == '__main__':
    nevents = int(sys.argv[1])

    path = sys.argv[2]  # put some error handling in

    if len(sys.argv) == 4:
        sort = sys.argv[3]
    else:
        sort = 'alphanumeric'
    ndim = 12

    orbit = Orbit()

    if not os.path.exists(path + 'posteriors/'):  # make a directory for the posteriors
            os.mkdir(path + 'posteriors/')

    for i in range (nevents):

        data_structure = Data()
        event_name, truths, data = data_structure.new_event(path, sort)
        
        print('\n\n\n\n\n\nevent_name = ', event_name)
        print('---------------------------------------')
        print('truths = ', truths)

        piE = np.array([truths['piEN'], truths['piEE']])

        tu_data = {}
        epochs = {}
        t_data = {}

        for obs in data.keys():
            tu_data[obs] = data[obs][3:5,:].T
            epochs[obs] = data[obs][0,:]
        
        parallax = Parallax(truths['ra_deg'], truths['dec_deg'], 
                            orbit, truths['tcroin'],
                            tu_data, piE, epochs)
        parallax.update_piE_NE(truths['piEN'], truths['piEE'])

        event = Event(parallax, orbit, data, truths['params'])
        
        fit = Fit()
        chi2_ew, chi2 = fit.get_chi2(event, truths['params'])
        with open(path+'posteriors/'+event_name+'_chi2_elements.pkl', 'wb') as f:
            pickle.dump(chi2_ew, f)
        np.savetxt(path+'posteriors/'+event_name+'_chi2.txt', np.array([chi2]))

        # VBBL
        #------------------------------------------------
        vbbl = VBBinaryLensing()
        vbbl.a1 = 0.36
        #------------------------------------------------


        # Plot lightcurve and "truths" model
        #------------------------------------------------
        A = {}  # t is data epochs
        A_lin = 0  # linearly spaced time
        n = 10000
        fb = {}
        t0 = event.true_params[5]
        tE = event.true_params[6]
        tt = np.linspace(t0-5.0*tE, t0+5.0*tE, n)
        fs = {}

        plt.figure()

        rho = event.true_params[2]

        for obs in event.data.keys():

            # data
            t = event.data[obs][0]  # BJD
            f = event.data[obs][1]  # obs_rel_flux
            f_err = event.data[obs][2]  # obs_rel_flux_err

            # calculating the model
            A[obs] = event.get_magnification(t, obs)
            fs[obs], fb[obs] = fit.get_fluxes(A[obs], f, f_err**2)

            # plotting the data
            plt.plot(t, (f-fb[obs])/fs[obs], '.', label=str(obs)+' data', alpha=0.5, zorder=0)
            
        # plotting the model
        A_lin = event.get_magnification(tt, obs)
        plt.plot(tt, A_lin, '-', label='model', zorder=1)

        plt.xlabel('BJD')
        plt.xlim(t0-0.5*tE, t0+0.5*tE)
        plt.ylabel('Magnification')
        plt.title('s=%.2f, q=%.2f, rho=%.2f, u0=%.2f, alpha=%.2f, t0=%.2f, tE=%.2f, \npiEE=%.2f, piEN=%.2f, i=%.2f, phase=%.2f, period=%.2f' %tuple(event.true_params))
        plt.legend()
        plt.savefig(path+'posteriors/'+event_name+'_truths_lightcurve.png')
        plt.close()
        #------------------------------------------------

        # caustic "truths"
        #------------------------------------------------
        # plot lenses
        s = event.true_params[0]
        q = event.true_params[1]
        t0 = event.true_params[5]
        tE = event.true_params[6]
        print('s, q', s, q)

        n = 201
        u1 = np.linspace(-2, 2, n)
        u2 = np.linspace(-2, 2, n)
        Amap = np.zeros([n, n])
        rho = event.true_params[2]

        '''
        for i in range(n):
            for j in range(n):
                Amap[j,i] = vbbl.BinaryMagDark(s, q, u1[i], u2[j],  rho, 1.0e-5)
                #'''

        solutions = vbbl.PlotCrit(s, q) # Returns _sols object containing n crit. curves followed by n caustic curves

        def iterate_from(item):
            while item is not None:
                yield item
                item = item.next

        curves = []
        for curve in iterate_from(solutions.first):
            for point in iterate_from(curve.first):
                curves.append((point.x1, point.x2))
                
        critical_curves = np.array(curves[:int(len(curves)/2)])
        caustic_curves = np.array(curves[int(len(curves)/2):])

        plt.figure()

        print('\nplotting lenses\n')
        plt.plot(event.lens1_0[0], 
                event.lens1_0[1], 
                'o', ms=10, color='red'
                )
        plt.plot(event.lens2_0[0], 
                event.lens2_0[1], 
                'o', ms=10**q, color='blue'
                )

        print('\nselecting important epochs\n')
        t = event.data[0][0]  # BJD
        points = np.where(np.logical_and(t > (t0-5.0*tE), t < (t0+5.0*tE)))

        # plot trajectory
        print('\nplotting standard trajectory\n')
        plt.plot(event.traj_base_u1[0][points], 
                event.traj_base_u2[0][points], 
                ':', color='black'
                )
        print('\nadding parallax purturbation\n')
        plt.plot(event.traj_parallax_u1[0][points], 
                event.traj_parallax_u2[0][points], 
                '--', color='black'
                )
        print('\nadding LOM purturbation\n')
        plt.plot(event.traj_parallax_dalpha_u1[0][points], 
                event.traj_parallax_dalpha_u2[0][points], 
                '-', color='black'
                )
        
        print('\nskipping contours\n')
        '''
        plt.contour(u1, u2, np.log10(Amap), 
                    levels=50, 
                    linewidths=0.5, 
                    colors='black', 
                    zorder=0
                    )#'''

        print('\nplotting caustics\n')
        plt.plot(caustic_curves[:,0], 
                caustic_curves[:,1], 
                '.', color='blue', ms=0.2, 
                zorder=1
                )

        print('\nplotting criticals\n')
        plt.plot(critical_curves[:,0], 
                critical_curves[:,1], 
                '.', color='grey', ms=0.2, alpha=0.5, 
                zorder=1
                )

        print('\naesthetics\n')
        plt.grid()
        plt.axis('equal')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.axis('equal')

        plt.savefig(path+'posteriors/'+event_name+'_truths_caustic.png')
        plt.close()
        #------------------------------------------------

        '''
        sampler = dynesty.NestedSampler(
                                        fit.lnprob, 
                                        fit.prior_transform, 
                                        ndim, 
                                        nlive=200, 
                                        sample='rwalk', 
                                        bound='multi', 
                                        pool=mp.Pool(mp.cpu_count()),
                                        args=[event]
                                        )  # 'rwalk' is best for 10 < ndim < 20
        sampler.run_nested(maxiter=1000, checkpoint_file=path+'posteriors/'+event_name+'.save')

        # Save the sampler as a pickle file
        with open(path+'posteriors/'+event_name+'_sampler.pkl', 'wb') as f:
            pickle.dump(sampler, f)

        res = sampler.results

        # print for logs
        print('Event', i, '(', event_name, ') is done')
        print(res.summary())

        # Save plots
        Fit.corner_post(res, event_name, path, truths)
        Fit.runplot(res, event_name, path)
        Fit.traceplot(res, event_name, path, truths)

        samples = res.samples
        np.save(path+'posteriors/'+event_name+'_post_samples.npy', samples)
        #'''
        # Done with the event
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            complete_list = np.loadtxt(path+'complete.txt', dtype=str)

            complete_list = np.hstack([complete_list, event_name])
            print('Completed:', event_name)
            np.savetxt(path+'complete.txt', complete_list, fmt='%s')
        #sampler.reset()

# ephermeris
# 2018-08-10
# 2023-04-30
# get emphemeris file from JPL Horizons for SEMB-L2