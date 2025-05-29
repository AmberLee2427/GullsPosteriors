import numpy as np
import os
import sys
from astropy import units as u
from astropy import coordinates as astrocoords


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
            tushift = [0, 0]
            tushift[0] = -np.linalg.norm(self.piE) * ( NE[obs][0]*cs + NE[obs][1]*sn)  # Delta_tau - shift in the relative-motion direction
            tushift[1] = -np.linalg.norm(self.piE) * (-NE[obs][0]*sn + NE[obs][1]*cs)  # Delta_beta - shift perpendicular to the lens-source motion

            tu[obs] = tushift

        # ideally, this will cause a break if the t logic is wrong
        if len(tu.keys()) == 1:  # unpack from the dictionary if there is only one observatorty
            obs = list(tu.keys())[0]
            tu = tu[obs]

        return tu