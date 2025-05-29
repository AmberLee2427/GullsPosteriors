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
from dynesty import pool
from multiprocessing import Pool
from scipy.stats import truncnorm
import time

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
                 start_date='2018-04-25',
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
                #print(tt.shape, self.epochs[obs].shape, ne.shape)
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
# phase0 is the phase at simulation 0 time
# assume a circular orbit and a small mass ratio m << M

class Event:

    def __init__(self, parallax: Parallax, orbit: Orbit, data: dict, truths: dict, t_start: float, t_ref: float):
        self.orbit = orbit
        self.parallax = parallax
        self.parallax.set_ref_frame(t_ref)
        self.data = data
        self.t_ref = t_ref * 1.0
        self.sim_time0 = t_start * 1.0
        #truths = self.tref2t0(truths)
        self.truths = truths
        self.true_params = truths['params']
        self.params = self.true_params.copy()  # don't want to edit true params when we edit params
        #print('debug Event.__init__: sim_time0: ', self.sim_time0)
        #print('debug Event.__init__: t_ref: ', self.t_ref)

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

    def projected_seperation(self, i, period, t, phase_offset=0.0, t_start=None, a=1.0):
        '''Calculate the projected seperation of the binary'''

        # i is the inclination of the orbit in radians
        # period is the period of the orbit (if t is tau then period is in units of tE)
        # t is the time relative to t0 (t')
        # phase_offset is the phase at simulation zero time
        # a is the semi-major axis of the orbit

        if t_start is None:
            t_start = self.t_ref

        phase = 2 * np.pi * (t - t_start) / period + phase_offset
        #phase = phase % (2 * np.pi)
        x = a * np.cos(phase)
        y = a * np.sin(phase) * np.cos(i)
        z = a * np.sin(phase) * np.sin(i)

        # if an inclination of 0 means birds eye view, then use x, y
        s = np.sqrt(x**2 + y**2)

        # if an inclination of 0 means edge on view, then use x, z
        #s = np.sqrt(x**2 + z**2)

        return s, x, y
    
    def tref2t0(self, truths):
        '''convert the ref time from tcroin to t0 for the lightcurve params'''
        
        tcroin = truths['tcroin']
        t0 = truths['params'][5].copy()  # t0lens1 (random)'''

        # Convert to COM origin

        # convert to t0 reference time
        alpha = truths['params'][4].copy()  # 'alpha' * pi / 180 (random)
        s = truths['params'][0].copy()  # 'Planet_s'
        q = truths['params'][1].copy()  # 'q'
        period = truths['params'][11].copy()  # 'Planet_orbperiod' * 365.25
        tE = truths['params'][6].copy()  # tE_ref
        phase0 = truths['params'][10].copy()  # 'Planet_phase' * pi / 180 at tcroin
        t_start = self.sim_time0
        i = truths['params'][9].copy()  # 'Planet_inclination' * pi / 180
        
        a = truths['Planet_semimajoraxis']
        #truths['xCoM'] = s * q/(1.0+q)

        # redefining s is always at time t0

        # O at L1 to CoM
        delta_x0_com = s * q/(1.0+q)
        truths['xCoM'] = delta_x0_com
        #print('\ndebug Event.tref2t0: delta_x0_com: ', delta_x0_com)

        # I think alpha is defined at t0, always

        # phase ofset at time t0
        phase0_t0_L1 = phase0 + (t0 - self.t_ref) * 2.0 * np.pi / period
        #print('\ndebug Event.tref2t0: phase0_t0_L1: ', phase0_t0_L1)

        u0 = truths['params'][3].copy()     # 'u0lens1' closest approach to L1 (random)
                                            # at t0 with L1 at the origin
        piEE = truths['params'][7].copy()   # 'piEE'
        piEN = truths['params'][8].copy()   # 'piEN'
        rho = truths['params'][2].copy()    # 'rho'

        ''' the scale in time is wrong here
        x0 = t0 * np.cos(alpha) - u0 * np.sin(alpha)
        x0_com = x0 - delta_x0_com
        y0 = t0 * np.sin(alpha) + u0 * np.cos(alpha)
        y0_com = y0 - delta_x0_com * np.tan(alpha)
        delta_t0_com = np.sqrt(delta_x0_com**2 + (delta_x0_com*np.tan(alpha))**2)
        t0_com = t0 - delta_t0_com
        delta_u0_com = delta_t0_com * np.tan(alpha)
        u0_com = u0 - delta_u0_com #'''

        # u0, t0, and source position at t0 (x0, y0) 
        x0 = -u0 * np.sin(alpha) # because t0 = 0 in the tau scale
        y0 = u0 * np.cos(alpha)
        x0_com = x0 - delta_x0_com
        y0_com = y0 - delta_x0_com * np.tan(alpha)
        delta_tau_com = np.sqrt(delta_x0_com**2 + (delta_x0_com*np.tan(alpha))**2)
        t0_com = t0 - delta_tau_com * tE
        #print('\ndebug Event.tref2t0: t0_com: ', t0_com)
        delta_u0_com = delta_tau_com * np.tan(alpha)
        u0_com = u0 - delta_u0_com
        #print('debug Event.tref2t0: u0_com: ', u0_com)
        # this is ignoring orital motion

        _, sx_ref, sy_ref = self.projected_seperation(i, period, self.t_ref, phase_offset=phase0, t_start=self.t_ref)
        s_, sx_t0, sy_t0 = self.projected_seperation(i, period, t0, phase_offset=phase0, t_start=self.t_ref)  
        _, sx_t0_com, sy_t0_com = self.projected_seperation(i, period, t0_com, phase_offset=phase0, t_start=self.t_ref)  

        a = s / s_
        rot_ref = np.arctan2(sy_ref, sx_ref)  # again, the size of Sx and Sy are nonsense, but the ratio is valid
        rot_t0_L1 = np.arctan2(sy_t0, sx_t0)
        rot_t0_com = np.arctan2(sy_t0_com, sx_t0_com)
        #print('\ndebug Event.tref2t0: rot_ref: ', rot_ref)  # entirely due to the phase offset
        #print('debug Event.tref2t0: rot_t0_L1: ', rot_t0_L1)
        #print('debug Event.tref2t0: rot_t0_com: ', rot_t0_com)
        rot = rot_t0_L1 - rot_ref
        rot_t0_com = rot_t0_com - rot_ref
        #print('debug Event.tref2t0: rot_t0_com: ', rot_t0_com)
        #print('debug Event.tref2t0: rot: ', rot)

        #  have no clue if this is right and don't wan to check itß
        x0_com_rot = x0_com * np.cos(rot_t0_com) - y0_com * np.sin(rot_t0_com)
        y0_com_rot = x0_com * np.sin(rot_t0_com) + y0_com * np.cos(rot_t0_com)
        #print('\ndebug Event.tref2t0: alpha_gulls: ', alpha)
        alpha_t0_com = np.arctan2(y0_com_rot, x0_com_rot) # alpha changes because t0 changes
        #print('debug Event.tref2t0: alpha_t0_com: ', alpha_t0_com)
        phase0_t0_com = phase0 + (t0_com - self.t_ref) * 2.0 * np.pi / period

        s_com, _, _ = self.projected_seperation(i, period, t0_com, phase_offset=phase0, t_start=self.t_ref)  


        truths['params_t0_L1'] = [s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase0_t0_L1, period]
        # ß
        truths['params_t0_COM'] = [s_com, q, rho, u0_com, alpha_t0_com, t0_com, tE, piEE, piEN, i, phase0_t0_com, period]
        #print('debug Event.tref2t0: truths[params_t0_L1]: ', truths['params_t0_L1'])
        #print('debug Event.tref2t0: truths[params_t0_COM]: ', truths['params_t0_COM'])

        return truths
    
    def croin(self, t0: float, u0: float, s: float, q: float, alpha: float, tE: float):
        '''recalculate tcroin given t0 etc.'''

        sina = np.sin(alpha)
        cosa = np.cos(alpha)

        if s<0.1:
            tc = t0
            uc = u0
            rc = np.nan

        else:
            yc = 0.0

            if s<1.0:
                xc = (s - 1.0/s)*(1-q)/(1+q)
                rc = 2.0 * np.sqrt(q) / (s * np.sqrt(1.0 + s * s))
            else:
                xc = (s - 1.0 / (s * (1.0 + q))) - q * s / (1.0 + q)
                rc = 2.0 * np.sqrt(q)

            xc += s*q/(1+q)  # shift to CoM
            rc *= 2.0 + np.min([45*s*s, 80/s/s])

            tc = t0 + tE * (xc * cosa + yc * sina)  # yc = 0
            uc = u0 + (xc * sina - yc * cosa) / rc  # yc = 0

        print('debug Event.croin: uc:', uc, self.truths['ucroin'])
        print('debug Event.croin: tc:', tc, self.truths['tcroin'])
        print('debug Event.croin: rc:', tc, self.truths['rcroin'])

        self.truths['ucroin_calc'] = uc
        self.truths['tcroin_calc'] = tc
        self.truths['rcroin_calc'] = rc

        return tc

    
    def get_magnification(self, t, obs=None, p0=False):
        '''Calculate the magnification'''

        q = self.params[1].copy()  # mass ratio - not dependent on reference frame
        i = self.params[9].copy()  # inclination of the orbit
        phase0 = self.params[10].copy()  # phase at time tcroin
        period = self.params[11].copy()  # planet orbital period
        t0 = self.params[5].copy()  # time of event peak in BJD
        t_start = self.sim_time0  # I shouldn't need this
        tE = self.params[6].copy()  # Einstein crossing time/event timescale
        t_ref = self.t_ref.copy()  # reference time for the parallax and LOM
        #tau_start = (t_start-t0)/tE
        #tcroin = self.truths['tcroin']
        u0 = self.params[3].copy()  # impact parameter relative to L1
        alpha = self.params[4].copy()  # angle of the source trajectory
        s = self.params[0].copy()  # angular lens seperation at time t0

        # Semimajor axis in units of thetaE
        s_ref, _, _ = self.projected_seperation(i, period, 0.0, phase_offset=phase0, t_start=0.0) 
        a = s/(s_ref)  # angular semimajor axis in uits of thetaE
        #print('debug Event.get_magnification: a/rE:', 
        #      self.truths['Planet_semimajoraxis']/self.truths['rE'], a)

        # COM frame
        a1 = q/(1.0+q) * a
        a2 = a - a1
        s1 = q/(1.0+q) * s  # L1 to CoM offset
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

        tau = (t - t0) / tE  # event time in units of tE (tt) relative to t0
                             # where t0 is the time of closest approach to L1

        # Orbital motion - dsdt - COM frame
        #print('debug Event.get_magnification: i: ', i)
        #print('debug Event.get_magnification: period: ', period)
        #print('debug Event.get_magnification: period(tau): ', period/tE)
        #print('debug Event.get_magnification: tau: ', tau)
        #print('debug Event.get_magnification: phase0: ', phase0)
        #print('debug Event.get_magnification: phase0 - 180: ', phase0+np.pi)
        #print('debug Event.get_magnification: a1: ', a1)
        #print('debug Event.get_magnification: a2: ', a2)

        # phase is always defined at the fixed time t_ref 
        # this coordinate system is defined at time t_ref with a COM origin
        # let ss be the array of angular lens seperations at each epoch
        # ss does not have a reference frame
        ss1, x1, y1 = self.projected_seperation(i, period, t, 
                                                phase_offset=phase0+np.pi,
                                                a=a1,
                                                t_start=t_ref
                                                )  # star - L1
        ss2, x2, y2 = self.projected_seperation(i, period, t, 
                                                phase_offset=phase0, 
                                                a=a2, 
                                                t_start=t_ref
                                                )  # planet - L2
        ss = np.sqrt((x2 - x1)**2+(y2 - y1)**2)  # i don't know that this is strictly necessary since they are always opposite
        self.ss[obs] = ss  # saving these for bug testing
        self.tau[obs] = tau
        #print('\ndebug Event.get_magnification: s: \n', 
        #      ss, ss.shape, '\n', 
        #      ss1+ss2, (ss1+ss2).shape
        #      )  # these should be equal, I think

        #umin = min(umin, np.sqrt(tt**2, uu**2))  # gulls uses this value for weights stuff

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

        beta = np.ones_like(tt) * u0  # L1 impact parameter without parallax
        uu = beta + Delta_beta  # L1 impact parameter with parallax
        self.traj_base_beta[obs] = beta
        self.traj_parallax_beta[obs] = uu

        cosalpha = np.cos(alpha)  # have to do this rotation before the CoM shift
        sinalpha = np.sin(alpha)  # because s, u0, t0, and alpha are defined relative 
                                  # to L1 closeset approach

        delta_x0_com = s * q/(1.0+q)
        self.truths['xCoM'] = delta_x0_com

        # - s1 moves to CoM
        xsin = tt * cosalpha - uu * sinalpha - self.truths['xCoM'] # subtracting L1-COM distance
                                                                   # at time t0
        self.traj_parallax_u1[obs] = xsin  # saving for debugging
        self.traj_base_u1[obs] = tau * cosalpha - beta * sinalpha - delta_x0_com  # sans parallax, for debug

        ysin = tt * sinalpha + uu * cosalpha  # no y-shift because lens axis is fixed
        self.traj_parallax_u2[obs] = ysin  # saving for debugging
        self.traj_base_u2[obs] = tau * sinalpha + beta * cosalpha # sans parallax, for debug

        # Orbital motion - dalphadt
        rot = np.arctan2(y2, x2)  # positions of the planet in the COM-origin, planet-rotation frame
                                  # what is the reference time for this? - currently tref
        _, x0, y0 = self.projected_seperation(i, period, 0.0, t_start=0.0, phase_offset=phase0, a=a)
        rot0 = np.arctan2(y0, x0)  # at reference time
        # the x, y positions are nonsense wihtout a/a1/a2, but their ratios are valid for the rotation either way
        self.dalpha[obs] = rot - rot0  # saving for debugging
        #print('debug Event.get_magnification: rot: ', rot)
        #print('debug Event.get_magnification: rot0: ', rot0)
        #print('debug Event.get_magnification: dalpha: ', self.dalpha[obs])
        cosrot = np.cos(-self.dalpha[obs])  # gulls has -dalpha
        sinrot = np.sin(-self.dalpha[obs])

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

        eps = 1.0e-4  # precision of the numerical integration
        gamma = 0.36
        vbbl = VBBinaryLensing()
        vbbl.a1 = gamma

        A = np.zeros_like(ss)
        
        # vbbl has CoM O
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

    def __init__(self, debug=None) -> None:
        if debug is not None:
            self.debug = debug
        else:
            self.debug = []

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

        if 'fluxes' in self.debug:
            print('debug Fit.get_fluxes: A: ', A)
            print('debug Fit.get_fluxes: C: ', C)
            print('debug Fit.get_fluxes: B: ', B)
            print('debug Fit.get_fluxes: FS: ', FS)
            print('debug Fit.get_fluxes: FB: ', FB)
        
        return FS, FB 

    def get_chi2(self, event, params):

        if 'chi2' in self.debug:
            print('debug Fit.get_chi2: params: ', params)
            print('debug Fit.get_chi2: event type: ', type(event))

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

            if 'chi2' in self.debug:
                print('debug Fit.get_chi2: obs: ', obs)
                print('debug Fit.get_chi2: t: ', t)
                print('debug Fit.get_chi2: f: ', f)
                print('debug Fit.get_chi2: f_err: ', f_err)
                print('debug Fit.get_chi2: A: ', A)
                print('debug Fit.get_chi2: fs: ', fs)
                print('debug Fit.get_chi2: fb: ', fb)
                print('debug Fit.get_chi2: chi2: ', chi2[obs])
                print('debug Fit.get_chi2: chi2sum: ', chi2sum)
            else:
                print('*', end='')

        return chi2, chi2sum

    def lnlike(self, theta, event):  

        _, chi2 = self.get_chi2(event, theta)

        if 'lnlike' in self.debug:
            print('debug Fit.lnlike: chi2: ', chi2)
            print('debug Fit.lnlike: theta: ', theta)
        else:
            print('-', end='')

        return -0.5 * chi2

    def lnprior(self, theta, bound_penalty=False):
        s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period = theta

        if 'ln_prior' in self.debug:
            print('debug Fit.lnprior: s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period: ')
            print('                   ', theta)

        if tE > 0.0 and q <= 1.0 and period/tE > 4 and s > 0.001:
        
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

        else:
            if 'lnprior' in self.debug:
                if tE > 0.0:
                    print('debug Fit.lnprior: tE = ', tE, ' > 0.0')
                if q <= 1.0:
                    print('debug Fit.lnprior: q = ', q, ' <= 1.0')
                if period/tE > 4:
                    print('debug Fit.lnprior: period/tE = ', period/tE, ' > 4')
                if s > 0.001:
                    print('debug Fit.lnprior: s = ', s, ' > 0.01')
            else:
                print('^', end='')

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
        
        if 'lnprob' in self.debug:
            print('debug Fit.lnprob: lp, ll: ', lp, ll)
            print('debug Fit.lnprob: s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period: ')
            print('                  ', theta)
        else:
            print('.', end='')

        return lp + ll

    def prior_transform(self, u, true, prange, normal=False):
        """Transform unit cube to the parameter space. Nested sampling has firm boundaries on the prior space."""
        
        if 'pt' in self.debug:
            print('debug: Fit.prior_transform: logs, logq, logrho, u0, alpha, t0, tE, piEE, piEN, i, sinphase, logperiod:')
            print('                            ', u)

        logs_true = np.log10(true[0])
        logq_true = np.log10(true[1])
        logrho_true = np.log10(true[2])
        u0_true = true[3]
        alpha_true = true[4]
        t0_true = true[5]
        tE_true = true[6]
        piEE_true = true[7]
        piEN_true = true[8]
        i_true = true[9]
        phase_true = true[10]
        logperiod_true = np.log10(true[11]*1.0)
        true_array = np.array([logs_true, logq_true, logrho_true, u0_true, alpha_true, t0_true, tE_true, piEE_true, piEN_true, i_true, phase_true, logperiod_true])

        if 'pt' in self.debug:
            print('        log true         ', true_array)

        min_array = true_array - prange
        max_array = true_array + prange

        if normal:

            def normal(u, mu, sig, bounds):
                '''Maps a uniform random variable u (between 0 and 1) to a truncated normal random value x,
                constrained between bounds[0] and bounds[1], with a mean of mu and standard deviation of sig.

                Parameters:
                u (float): Uniform random variable between 0 and 1.
                mu (float): Mean of the normal distribution.
                sig (float): Standard deviation of the normal distribution.
                bounds (tuple): Tuple containing the lower and upper bounds (bounds[0], bounds[1]).

                Returns:
                float: Truncated normal random value x constrained between bounds[0] and bounds[1].
                '''
                # Calculate the lower and upper bounds in terms of the standard normal distribution
                a, b = (bounds[0] - mu) / sig, (bounds[1] - mu) / sig
                
                # Create a truncated normal distribution
                trunc_normal = truncnorm(a, b, loc=mu, scale=sig)
                
                # Map the uniform random variable to the truncated normal distribution
                x = trunc_normal.ppf(u)
                
                return x

            x = normal(u, true_array, prange/5.0, [min_array, max_array])

            if 'pt' in self.debug:
                print('            normal           ', x)
                print('            min              ', min_array)
                print('            max              ', max_array)
                print('            true             ', true_array)


        else:
            x = (u-0.5)*prange+true_array

        logs, logq, logrho, u0, alpha, t0, tE, piEE, piEN, i, phase, logperiod = x

        s = 10.0**logs  # log uniform samples
        q = 10.0**logq
        rho = 10.0**logrho
        #print('debug: Fit.prior_transform: phase: ', phase)
        period = 10.0**logperiod

        if 'pt' in self.debug:
            print('debug Fit.prior_transform: s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period: ')
            print('                  ', s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period)
            print('       truths:    ', true)
        else:
            print('~', end='')

        return s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period

    def corner_post(self, res, event_name, path, truths):
        labels = ['s', 'q', 'rho', 'u0', 'alpha', 't0', 'tE', 'piEE', 'piEN', 'i', 'phase', 'period']
        fig = corner.corner(res.samples, labels=labels, truths=truths['params'])
        plt.title(event_name)

        fig.savefig(path+'posteriors/'+event_name+'_corner.png')
        plt.close(fig)

        if 'corner' in self.debug:
            print('debug Fit.corner_post: labels: ', labels)
            print('debug Fit.corner_post: truths: ', truths['params'])
            print('debug Fit.corner_post: event_name: ', event_name)
            print('debug Fit.corner_post: path: ', path)

    def runplot(self, res, event_name, path):
        if 'run' in self.debug:
            print('debug Fit.runplot: event_name: ', event_name)
            print('debug Fit.runplot: path: ', path)

        fig, _ = dyplot.runplot(res)

        if 'run' in self.debug:
            print('debug Fit.runplot: dyplot.runplot fig: built')

        plt.title(event_name)

        fig.savefig(path+'posteriors/'+event_name+'_runplot.png')

        if 'run' in self.debug:
            print('debug Fit.runplot: save path: ', path+'posteriors/'+event_name+'_runplot.png')    

        plt.close(fig)

    def traceplot(self, res, event_name, path, truths):
        if 'trace' in self.debug:
            print('debug Fit.traceplot: event_name: ', event_name)
            print('debug Fit.traceplot: path: ', path)
            print('debug Fit.traceplot: truths: ', truths)

        labels = ['s', 'q', 'rho', 'u0', 'alpha', 't0', 'tE', 'piEE', 'piEN', 'i', 'phase', 'period']
        fig, _ = dyplot.traceplot(res, 
                                    truths=np.array(truths['params']),
                                    truth_color='black', 
                                    show_titles=True,
                                    trace_cmap='viridis', 
                                    connect=True,
                                    connect_highlight=range(5), 
                                    labels=labels)
        
        if 'trace' in self.debug:
            print('debug Fit.traceplot: dyplot.traceplot fig: built')

        plt.title(event_name)

        fig.savefig(path+'posteriors/'+event_name+'_traceplot.png')

        if 'trace' in self.debug:
            print('debug Fit.traceplot: save path: ', path+'posteriors/'+event_name+'_traceplot.png')

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
                    #print('event_name = ', event_name)

                    obs0_data = data[0].copy()
                    simt = obs0_data[7]
                    bjd = obs0_data[0]
                    
                    truths = self.get_params(master_file, EventID, SubRun, Field, simt, bjd)  
                    # turns all the degress to radians and sim time to bjd
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

        self.sim_time0 = np.sum(data['BJD'] - data['Simulation_time'])/len(data['BJD'])

        data = data[['BJD', 
                    'measured_relative_flux',
                    'measured_relative_flux_error',
                    'parallax_shift_t',
                    'parallax_shift_u', 
                    'observatory_code',
                    'true_relative_flux', 
                    'true_relative_flux_error',
                    'Simulation_time'
                    ]]
        
        data_dict = {}
        for code in data['observatory_code'].unique():
            data_obs = data[data['observatory_code'] == code][['BJD', 
                                          'measured_relative_flux', 
                                          'measured_relative_flux_error', 
                                          'parallax_shift_t', 
                                          'parallax_shift_u',
                                          'true_relative_flux', 
                                          'true_relative_flux_error',
                                          'Simulation_time'
                                          ]].reset_index(drop=True)
            data_dict[code] = data_obs.to_numpy().T

        return data_dict

    def get_params(self, master_file, EventID, SubRun, Field, epoch=None, bjd=None):
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

        #print(self.sim_time0)
        
        s = truths['Planet_s']
        q = truths['Planet_q']
        rho = truths['rho']
        u0 = truths['u0lens1']  # croin
        alpha = truths['alpha']*np.pi/180 # convert to radians
        if epoch is not None and bjd is not None:
            t0_sim = truths['t0lens1']
            t0_bjd = np.interp(t0_sim, epoch, bjd)
            t0 = t0_bjd
            tc_sim = truths['tcroin']
            tc_bjd = np.interp(tc_sim, epoch, bjd)
            tcroin = tc_bjd
        else:
            t0 = truths['t0lens1'] + self.sim_time0  # convert to BJD
            tcroin = truths['tcroin'] + self.sim_time0  # convert to BJDs
        truths['t0lens1'] = t0
        tE = truths['tE_ref'] 
        piEE = truths['piEE']
        piEN = truths['piEN']
        i = truths['Planet_inclination']*np.pi/180  # convert to radians
        phase = truths['Planet_orbphase']*np.pi/180  # convert to radians # centre on tcroin
        period = truths['Planet_period']*365.25  # convert to days
        #phase_change = truths['tcroin'] / period
        #phase = phase + phase_change  # centre on t0
        #phase = phase % (2.0*np.pi)  # make sure it's between 0 and 2pi
        truths['params'] = [s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period]

        truths['tcroin'] = tcroin

        return truths
    

if __name__ == '__main__':

    start_time = time.time()
    print('Start time = ', start_time)

    nevents = int(sys.argv[1])

    path = sys.argv[2]  # put some error handling in
    if path[-1] != '/':
        path = path + '/'

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
        t0 = truths['params'][5].copy()
        tE = truths['params'][6].copy()

        tu_data = {}
        epochs = {}
        t_data = {}
        f_true = {}
        f_err_true = {}

        for obs in data.keys():
            tu_data[obs] = data[obs][3:5,:].T
            epochs[obs] = data[obs][0,:]
            f_true[obs] = data[obs][5,:]
            f_err_true[obs] = data[obs][6,:]
            t_data[obs] = data[obs][0,:]

            # cutting the data down to the event region
            '''
            tmax = t0+5.0*tE
            tmin = t0-5.0*tE
            points = np.where(np.logical_and(t > tmin, t < tmax))
            
            tu_data[obs] = tu_data[obs][points,:]
            epochs[obs] = epochs[obs][points]
            f_true[obs] = f_true[obs][points]
            t_data[obs] = t_data[obs][points]

            data[obs] = data[obs][:,points]
            #'''

        parallax = Parallax(truths['ra_deg'], truths['dec_deg'], 
                            orbit, truths['tcroin'],
                            tu_data, piE, epochs)
        parallax.update_piE_NE(truths['piEN'], truths['piEE'])

        event_t0 = Event(parallax, orbit, data, 
                         truths, data_structure.sim_time0, truths['t0lens1']
                         )
        event_tc = Event(parallax, orbit, data, 
                         truths, data_structure.sim_time0, truths['tcroin']
                         )
        
        s = truths['params'][0].copy()
        q = truths['params'][1].copy()
        u0 = truths['params'][3].copy()
        alpha = truths['params'][4].copy()
        tc = truths['tcroin'].copy()

        tc_calc = event_tc.croin(t0, u0, s, q, alpha, tE)
        event_tref = Event(parallax, orbit, data,
                           truths, data_structure.sim_time0, tc_calc
                          )

        fit = Fit()

        chi2_ew_t0, chi2_t0 = fit.get_chi2(event_t0, truths['params'])
        chi2_ew_tc, chi2_tc = fit.get_chi2(event_tc, truths['params'])
        chi2_ew_tref, chi2_tref = fit.get_chi2(event_tref, truths['params'])

        tminmax = [t0-5.0*tE, t0+5.0*tE, tc_calc-5.0*tE, tc_calc+5.0*tE, tc-5.0*tE, tc+5.0*tE]
        tmax = np.max(tminmax)
        tmin = np.min(tminmax)
        points = np.where(np.logical_and(t_data[0] > tmin, t_data[0] < tmax))

        chi2_t0_0 = [np.sum(chi2_ew_t0[0]), np.sum(chi2_ew_t0[0][points])]
        cumsum_chi2_t0 = np.cumsum(chi2_ew_t0[0])
        chi2_tc_0 = [np.sum(chi2_ew_tc[0]), np.sum(chi2_ew_tc[0][points])]
        cumsum_chi2_tc = np.cumsum(chi2_ew_tc[0])
        chi2_tref_0 = [np.sum(chi2_ew_tref[0]), np.sum(chi2_ew_tref[0][points])]
        cumsum_chi2_tref = np.cumsum(chi2_ew_tref[0])

        end_preabmle = time.time()
        print('Time to get data = ', end_preabmle - start_time)

        plt.figure()

        N = cumsum_chi2_t0.shape[0]
        n = chi2_ew_t0[0][points].shape[0]
        plt.plot(epochs[0], cumsum_chi2_t0, label=r't0, $\chi^2/N=$%1.0f' %(chi2_t0_0[0]/N), color='blue')
        plt.plot(epochs[0], cumsum_chi2_tc, label=r'tc, $\chi^2/N=$%1.0f' %(chi2_tc_0[0]/N), color='cyan')
        plt.plot(epochs[0], cumsum_chi2_tref, label=r'tref, $\chi^2/N=$%1.0f' %(chi2_tref_0[0]/N), color='purple')

        plt.vlines(t0, color='black', linestyle='--', alpha=0.5, ymin=0, ymax=cumsum_chi2_t0[-1])

        plt.legend()
        plt.title(event_name + ' W146 cumulative chi2')

        plt.savefig(path+'posteriors/'+event_name+'_chi2_cumsum.png')

        with open(path+'posteriors/'+event_name+'t0_chi2_elements.pkl', 'wb') as f:
            pickle.dump(chi2_ew_t0, f)
        with open(path+'posteriors/'+event_name+'tc_chi2_elements.pkl', 'wb') as f:
            pickle.dump(chi2_ew_tc, f)
        with open(path+'posteriors/'+event_name+'tref_chi2_elements.pkl', 'wb') as f:
            pickle.dump(chi2_ew_tref, f)
        np.savetxt(path+'posteriors/'+event_name+'_chi2.txt', np.array([chi2_t0, chi2_tc, chi2_tref, 
                                                                        chi2_t0_0[0], chi2_tc_0[0], chi2_tref_0[0], N, 
                                                                        chi2_t0_0[1], chi2_tc_0[1], chi2_tref_0[1], n]), fmt='%1.0f')

        # VBBL
        #------------------------------------------------
        vbbl = VBBinaryLensing()
        vbbl.a1 = 0.36
        #------------------------------------------------


        # Plot lightcurve and "truths" model
        #------------------------------------------------
        A = {}  # t is data epochs
        A_lin = 0  # linearly spaced time
        A_true = {}
        n = chi2_ew_t0[0][points].shape[0]  # number of points from obs 0 within +-5tE to t0
        nn = 10000  # number of model points
        fb = {}
        t0 = event_tc.true_params[5]
        tE = event_tc.true_params[6]
        ttminmax = [t0-2.0*tE, t0+2.0*tE, tc_calc-2.0*tE, tc_calc+2.0*tE, tc-2.0*tE, tc+2.0*tE]
        ttmax = np.max(ttminmax)
        ttmin = np.min(ttminmax)
        tt = np.linspace(ttmin, ttmax, nn)
        fs = {}
        res = {}

        events = {'t0':event_t0, 'tc':event_tc}

        plt.figure()

        rho = event_tc.true_params[2]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2]})

        colours = ['red', 'orange', 'green']

        for obs in event_tc.data.keys():

            # data
            t = event_tc.data[obs][0]  # BJD
            f = event_tc.data[obs][1]  # obs_rel_flux
            f_err = event_tc.data[obs][2]  # obs_rel_flux_err

            # calculating the model
            A[obs] = event_tc.get_magnification(t, obs)
            fs[obs], fb[obs] = fit.get_fluxes(A[obs], f, f_err**2)
            fstrue, fbtrue = fit.get_fluxes(A[obs], f_true[obs], f_err_true[obs]**2)
            A_true[obs] = (f_true[obs] - fbtrue)/fstrue
            if int(obs) == 0:
                At0 = event_t0.get_magnification(t, obs)
                fst0, fbt0 = fit.get_fluxes(At0, f_true[obs], f_err_true[obs]**2)
                Atref = event_tref.get_magnification(t, obs)
                fstref, fbtref = fit.get_fluxes(Atref, f_true[obs], f_err_true[obs]**2)

                res['t0'] = f_true[obs]-(fst0*At0+fbt0)  # A_true[obs] - At0
                res['tc'] =  f_true[obs]-(fstrue*A[obs]+fbtrue)  #A_true[obs] - A[obs]
                res['tref'] = f_true[obs]-(fstref*Atref+fbtref)  #A_true[obs] - Atref

                chi2_ew_t0_true0 = (res['t0'])**2 / f_err_true[obs]**2
                chi2_ew_tc_true0 = (res['tc'])**2 / f_err_true[obs]**2
                chi2_ew_tref_true0 = (res['tref'])**2 / f_err_true[obs]**2

                chi2_t0_true0 = [np.sum(chi2_ew_t0_true0), np.sum(chi2_ew_t0_true0[points])]
                chi2_tc_true0 = [np.sum(chi2_ew_tc_true0), np.sum(chi2_ew_tc_true0[points])]
                chi2_tref_true0 = [np.sum(chi2_ew_tref_true0), np.sum(chi2_ew_tref_true0[points])]

                ax2.plot(t, res['tc'], '.', color='cyan', alpha=0.35, zorder=0, ms=1.5, label=r'$\chi^2=%1.2f$' %(chi2_tc_true0[1]))
                ax2.plot(t, res['t0'], '.', color='blue', alpha=0.35, zorder=0, ms=1.5, label=r'$\chi^2=%1.2f$' %(chi2_t0_true0[1]))
                ax2.plot(t, res['tref'], '.', color='purple', alpha=0.35, zorder=0, ms=2, label=r'$\chi^2=%1.2f$, n=%i' %(chi2_tref_true0[1], n))

                if chi2_t0_true0[1] <= chi2_tc_true0[1]:
                    fit_tref = t0
                else:
                    fit_tref = truths['tcroin']

            res[obs] = A_true[obs] - A[obs]
            # plotting the data
            ax1.plot(t, (f-fb[obs])/fs[obs], '.', color=colours[int(obs)], label=str(obs)+' data', alpha=0.5, zorder=0)

        np.savetxt(path+'posteriors/'+event_name+'chi2_true.txt', np.array([chi2_t0_true0, chi2_tc_true0, chi2_tref_true0]), fmt='%1.0f')
        
        # plotting the model
        A_lin = event_tc.get_magnification(tt, obs)
        ax1.plot(tt, A_lin, 
                 '-', 
                 label=r'$t_c=%1.1f$, $\chi^2/n=%1.2f$' %(event_tc.t_ref, chi2_tc_0[1]/n),
                 zorder=6,
                 color='cyan', 
                 alpha=0.75, 
                 lw=1
                 )

        A_lin = event_t0.get_magnification(tt, obs)
        ax1.plot(tt, A_lin, 
                '-', 
                label=r'$t_0=%1.1f$, $\chi^2/n=%1.2f$' %(event_t0.t_ref, chi2_t0_0[1]/n),
                zorder=4, 
                color='blue', 
                alpha=0.75, 
                lw=1
                )
        
        A_lin = event_tref.get_magnification(tt, obs)
        ax1.plot(tt, A_lin, 
                '-', 
                label=r'$t_{c,calc}=%1.1f$, $\chi^2/n=%1.2f$' %(event_tref.t_ref, chi2_tref_0[1]/n),
                zorder=2, 
                color='purple', 
                alpha=0.75, 
                lw=1
                )
        
        # plot vertical lines at time t0 and tcroin (tref)
        ax1.axvline(x=t0, color='blue', linestyle='--', alpha=0.5, zorder=0, linewidth=1)
        ax1.axvline(x=event_tc.t_ref, color='cyan', linestyle='--', alpha=0.5, zorder=0, linewidth=1)
        ax1.axvline(x=event_tref.t_ref, color='purple', linestyle='--', alpha=0.5, zorder=0, linewidth=1)

        ax2.axvline(x=t0, color='k', linestyle='--', alpha=0.5, zorder=0, linewidth=1)
        ax2.axvline(x=event_tc.t_ref, color='k', linestyle='--', alpha=0.5, zorder=0, linewidth=1)
        ax2.axvline(x=event_tref.t_ref, color='k', linestyle='--', alpha=0.5, zorder=0, linewidth=1)

        ax2.set_xlabel('BJD')
        xmin = np.min([t0 - 1.0*tE, event_tc.t_ref - 2.0, event_t0.t_ref - 2.0, event_tref.t_ref - 2.0])
        xmax = np.max([t0 + 1.0*tE, event_tc.t_ref + 2.0, event_t0.t_ref + 2.0, event_tref.t_ref + 2.0])
        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)
        ax1.set_ylabel('Magnification')
        ax2.set_ylabel('Residuals')
        ax1.set_title('s=%.2f, q=%.6f, rho=%.6f, u0=%.2f, alpha=%.2f, t0=%.2f, \ntE=%.2f, piEE=%.2f, piEN=%.2f, i=%.2f, phase=%.2f, period=%.2f' %tuple(event_tc.true_params))
        ax1.legend()
        ax2.legend()

        plt.savefig(path+'posteriors/'+event_name+'_truths_lightcurve.png', dpi=300)
        #plt.savefig(event_name+'_truths_lightcurve_test.png', dpi=300)
        plt.close()
        #------------------------------------------------

        # caustic "truths"
        #------------------------------------------------
        # plot lenses
        s = event_tc.true_params[0]
        q = event_tc.true_params[1]
        t0 = event_tc.true_params[5]
        tE = event_tc.true_params[6]
        tc = truths['tcroin']
        a = truths['Planet_semimajoraxis']/truths['rE']
        phase0 = event_tc.true_params[10]
        period = event_tc.true_params[11]
        i = event_tc.true_params[9]

        print('s, q', s, q)

        def iterate_from(item):
            while item is not None:
                yield item
                item = item.next

        s_t0, _, _ = event_tc.projected_seperation(i, period, t0, phase_offset=phase0, t_start=t0, a = a)
        s_tc, _, _ = event_tc.projected_seperation(i, period, tc, phase_offset=phase0, t_start=t0, a = a)
        s_tref, _, _ = event_tc.projected_seperation(i, period, tc_calc, phase_offset=phase0, t_start=t0, a = a)
        print(s_tc, s_t0, s_tref)

        solutions_t0 = vbbl.PlotCrit(s, q) # Returns _sols object containing n crit. curves followed by n caustic curves
        solutions_tc = vbbl.PlotCrit(s_tc, q)
        solutions_tref = vbbl.PlotCrit(s_tref, q)

        curves_t0 = []
        curves_tc = []
        curves_tref = []
        for curve in iterate_from(solutions_t0.first):
            for point in iterate_from(curve.first):
                curves_t0.append((point.x1, point.x2))
        for curve in iterate_from(solutions_tc.first):
            for point in iterate_from(curve.first):
                curves_tc.append((point.x1, point.x2))
        for curve in iterate_from(solutions_tref.first):
            for point in iterate_from(curve.first):
                curves_tref.append((point.x1, point.x2))
                
        critical_curves_t0 = np.array(curves_t0[:int(len(curves_t0)/2)])
        caustic_curves_t0 = np.array(curves_t0[int(len(curves_t0)/2):])
        critical_curves_tc = np.array(curves_tc[:int(len(curves_tc)/2)])
        caustic_curves_tc = np.array(curves_tc[int(len(curves_tc)/2):])
        critical_curves_tref = np.array(curves_tref[:int(len(curves_tref)/2)])
        caustic_curves_tref = np.array(curves_tref[int(len(curves_tref)/2):])
        
        plt.figure()

        print('\nplotting lenses\n')
        plt.plot(event_tc.lens1_0[0], 
                event_tc.lens1_0[1], 
                'o', ms=10, color='red', zorder=0
                )
        plt.plot(event_tc.lens2_0[0], 
                event_tc.lens2_0[1], 
                'o', ms=10**q, color='red', zorder=0
                )

        print('\nselecting important epochs\n')
        t = event_tc.data[0][0]  # BJD
        tmax = np.max([t0+5.0*tE, tc+2.0, tc_calc+2.0])
        tmin = np.min([t0-5.0*tE, tc-2.0, tc_calc-2.0])
        points = np.where(np.logical_and(t > tmin, t < tmax))

        # plot trajectory
        print('\nplotting standard trajectory\n')
        plt.plot(event_tc.traj_base_u1[0][points], 
                event_tc.traj_base_u2[0][points], 
                ':', color='black'
                )
        print('\nadding parallax purturbation\n')
        plt.plot(event_tc.traj_parallax_u1[0][points], 
                event_tc.traj_parallax_u2[0][points], 
                '--', color='black'
                )
        print('\nadding LOM purturbation\n')
        plt.plot(event_t0.traj_parallax_dalpha_u1[0][points], 
                 event_t0.traj_parallax_dalpha_u2[0][points], 
                 '-', color='cyan', alpha=0.5
                 )
        plt.plot(event_tc.traj_parallax_dalpha_u1[0][points],
                 event_tc.traj_parallax_dalpha_u2[0][points],
                 '-', color='black', alpha=0.8
                 )
        plt.plot(event_tref.traj_parallax_dalpha_u1[0][points],
                 event_tref.traj_parallax_dalpha_u2[0][points],
                 '-', color='purple', alpha=0.5
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
        plt.plot(caustic_curves_t0[:,0], 
                 caustic_curves_t0[:,1], 
                 '.', color='blue', ms=0.2, 
                 zorder=1
                 )

        plt.plot(caustic_curves_tc[:,0], 
                 caustic_curves_tc[:,1], 
                 '.', color='cyan', ms=0.2, 
                 zorder=0, alpha = 0.5
                 )
        
        plt.plot(caustic_curves_tref[:,0], 
                 caustic_curves_tref[:,1], 
                 '.', color='purple', ms=0.2, 
                 zorder=0, alpha = 0.5
                 )

        print('\nplotting criticals\n')
        plt.plot(critical_curves_t0[:,0], 
                 critical_curves_t0[:,1], 
                 '.', color='grey', ms=0.2, alpha=0.5, 
                 zorder=1
                 )
        
        u0 = event_tc.true_params[3]
        alpha = event_tc.true_params[4]
        xcom = truths['xCoM']
        x0 = -u0 * np.sin(alpha) - xcom
        y0 = u0 * np.cos(alpha)
        x0_c = -u0 * np.sin(alpha) + (event_tc.t_ref-t0)/tE * np.cos(alpha) - xcom
        y0_c = u0 * np.cos(alpha) + (event_tc.t_ref-t0)/tE * np.sin(alpha)
        x0_ref = -u0 * np.sin(alpha) + (event_tref.t_ref-t0)/tE * np.cos(alpha) - xcom
        y0_ref = u0 * np.cos(alpha) + (event_tref.t_ref-t0)/tE * np.sin(alpha)
        plt.plot(x0, y0, 'x', ms=7, color='blue', label=r'$t_0$')
        plt.plot(x0_c, y0_c, '+', ms=7, color='cyan', label=r'$t_c$')
        plt.plot(x0_ref, y0_ref, '+', ms=7, color='purple', label=r'$t_{c_{\rm calc}}$')
        
        xx = [1.0, -1.0, x0, x0_c, x0_ref]
        yy = [1.0, -1.0, y0, y0_c, y0_ref]

        print('\naesthetics\n')
        plt.grid()
        plt.axis('equal')
        plt.xlim(np.min(xx)-0.1, np.max(xx)+0.1)
        plt.ylim(np.min(yy)-0.1, np.max(yy)+0.1)
        plt.legend()

        plt.savefig(path+'posteriors/'+event_name+'_truths_caustic.png')
        plt.close()
        #------------------------------------------------

        #------------------------------------------------
        plt.figure()

        plt.plot(event_tc.tau[0], event_tc.ss[0], '.', label='ss', alpha=0.1)

        plt.xlabel(r'$\tau$')
        plt.ylabel('s')

        # plot a vertical line at time t0 and tcroin (tref)
        plt.axvline(x=0, color='b', linestyle='--', label=r'$t_0$', alpha=0.5)
        plt.axvline(x=(event_tc.t_ref-t0)/tE, color='cyan', linestyle='--', label=r'$t_{c}$', alpha=0.5)
        plt.axvline(x=(event_tc.sim_time0-t0)/tE, color='k', linestyle='--', label=r'$t_{sim0}$', alpha=0.5)
        plt.axvline(x=(event_tref.t_ref-t0)/tE, color='purple', linestyle='--', label=r'$t_{c,calc}$', alpha=0.5)

        plt.savefig(path+'posteriors/'+event_name+'_dsdtau.png')
        plt.close()
        #------------------------------------------------

        #------------------------------------------------
        plt.figure()

        plt.plot(event_tc.tau[0], event_tc.dalpha[0], '.', alpha=0.1)

        plt.xlabel(r'$\tau$')
        plt.ylabel(r'd$\alpha$')

        # plot a vertical line at time t0 and tcroin (tref)
        plt.axvline(x=0, color='b', linestyle='--', label=r'$t_0$', alpha=0.5)
        plt.axvline(x=(event_tc.t_ref-t0)/tE, color='cyan', linestyle='--', label=r'$t_{c}$', alpha=0.5)
        plt.axvline(x=(event_tc.sim_time0-t0)/tE, color='k', linestyle='--', label=r'$t_{sim0}$', alpha=0.5)
        plt.axvline(x=(event_tref.t_ref-t0)/tE, color='purple', linestyle='--', label=r'$t_{c,calc}$', alpha=0.5)

        plt.savefig(path+'posteriors/'+event_name+'_dalphadtau.png')
        plt.close()
        #------------------------------------------------

        end_initial_figures = time.time()
        print('Time to make initial figures = ', end_initial_figures - end_preabmle)

        # Prior Volume
        # I think these ranges need to stay the same for the logz values to be comparable
        # check how big these uncertainties normally are and adjust the ranges accordingly
        prange = np.array([0.05, 0.05, 0.5, 0.1, 0.1, 1.0, 1.0, 5.0, 10.0, np.pi/2.0, np.pi/2.0, 0.2])
        print('\n u prior ranges = ', prange)

        '''
        print('\nTesting the fit functions')
        print('--------------------------------')

        fit.debug = ['ln_like', 'ln_prior', 'pt']

        print(type(event_tc), type(event_tc.truths['params']))

        fit.get_chi2(event_tc, event_tc.truths['params'])
        fit.lnlike(event_tc.truths['params'], event_tc)
        u = np.random.rand(12)
        print('u = ', u)

        fit.prior_transform(u, event_tc.truths['params'], prange=prange, normal=True)
        fit.prior_transform(u, event_tc.truths['params'], prange=prange)

        sys.exit() #'''

        print()
        print('Sampling Posterior using Dynesty')
        print('--------------------------------')

        pooling = True
        normal = True
        nl = 200
        mi = 1000

        tminmax = [fit_tref-2.0*tE, fit_tref+2.0*tE, t0-2.0*tE, t0+2.0*tE]
        tmin = np.min(tminmax)
        tmax = np.max(tminmax)
        points = np.where(np.logical_and(t_data[0] > tmin, t_data[0] < tmax))

        data_cropped = {}
        for obs in data.keys():
            if obs == 0:
                data_cropped[obs] = data[obs][:,points]

        event_fit = Event(parallax, orbit, data_cropped, 
                         truths, data_structure.sim_time0, fit_tref
                         )

        if pooling:
            cores=mp.cpu_count()
            print(f'Pooling on {cores} threads')

            with Pool(cores) as pool:
                sampler = dynesty.DynamicNestedSampler(fit.lnprob, 
                                                fit.prior_transform, 
                                                ndim, 
                                                pool=pool,
                                                sample='rwalk', 
                                                bound='multi',
                                                nlive=nl,
                                                queue_size=cores,
                                                logl_args=[event_tc],
                                                ptform_args=[event_tc.truths['params'], prange],
                                                ptform_kwargs={'normal':normal}
                                                )
        
                fit.debug = []
                sampler.run_nested(print_progress=True, checkpoint_file=path+'posteriors/'+event_name+'.save') #, wt_kwargs={'pfrac': 1.0}, maxiter=mi, maxiter=res.niter+res.nlive, use_stop=False

        else:
            sampler = dynesty.DynamicNestedSampler(fit.lnprob, 
                                            fit.prior_transform, 
                                            ndim, 
                                            nlive=nl, 
                                            sample='rwalk', 
                                            bound='multi', 
                                            logl_args=[event_tc],
                                            ptform_args=[event_tc.truths['params'], prange],
                                            ptform_kwargs={'normal':normal}
                                            )

            sampler.run_nested(maxiter=mi, print_progress=True) # , checkpoint_file=path+'posteriors/'+event_name+'.save'


        '''sampler = dynesty.NestedSampler(
                                        fit.lnprob, 
                                        fit.prior_transform, 
                                        ndim, 
                                        nlive=1000, 
                                        sample='rwalk', 
                                        bound='multi', 
                                        pool=mp.Pool(4),
                                        logl_args=[event_tc],
                                        ptform_args=[event_tc.truths]
                                        )  # 'rwalk' is best for 10 < ndim < 20'''

        end_dynesty = time.time()
        print('Time to run dynesty (nl, mi)= ', end_dynesty - end_initial_figures, nl, mi)

        # Save the sampler as a pickle file
        with open(path+'posteriors/'+event_name+'_sampler.pkl', 'wb') as f:
            pickle.dump(sampler, f)

        res = sampler.results

        # print for logs
        print('Event', i, '(', event_name, ') is done')
        print(res.summary())

        # Save plots
        fit.corner_post(res, event_name, path, truths)
        fit.runplot(res, event_name, path)
        fit.traceplot(res, event_name, path, truths)

        samples = res.samples
        np.save(path+'posteriors/'+event_name+'_post_samples.npy', samples)

        with open(path+'posteriors/'+event_name+'end_truths.pkl', 'wb') as f:
            pickle.dump(event_tc.truths, f)

        #'''
        # Done with the event
        if not os.path.exists(path+'complete.txt'):
            complete_list = np.array([])
            np.savetxt(path+'complete.txt', complete_list, fmt='%s')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            complete_list = np.loadtxt(path+'complete.txt', dtype=str)

            complete_list = np.hstack([complete_list, event_name])
            print('Completed:', event_name)
            np.savetxt(path+'complete.txt', complete_list, fmt='%s')
        #sampler.reset()

        end_time = time.time()

        print('\n\nTime Summary')
        print('--------------------------------')
        print('Start time = ', start_time)
        print('Time to get data = ', end_preabmle - start_time)
        print('Time to make initial figures = ', end_initial_figures - end_preabmle)
        print('Time to run dynesty (nl, mi)= ', end_dynesty - end_initial_figures, nl, mi)
        print('Time to wrap up = ', end_time - end_dynesty)
        print('Total time = ', end_time - start_time)
        print('End time = ', end_time)
        print('--------------------------------\n\n\n')

# ephermeris
# 2018-08-10
# 2023-04-30
# get emphemeris file from JPL Horizons for SEMB-L2