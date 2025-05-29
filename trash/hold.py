import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

G_km = 6.67428e-20  # km^3/kg/s^2
G = 6.67428e-11  # m^3/kg/s^2
c_km = 2.99792458e5  # speed of light in km/s
pi = np.pi
r2d = 180.0/np.pi  # radians to degrees
d2r = np.pi/180.0  # degrees to radians
r2h = 12.0/np.pi  # radians to hrs
h2r = np.pi/12.0  # hrs to radians
arcsec2rad = np.pi/180.0/3600.0
sqrt2 = np.sqrt(2.0)
#oosqrt2 = 1.0/sqrt2
expe = np.exp(np.e)  # 2.71828182845904523536028747
log10e = np.log10(np.e)  # 0.4342944819032518276511289189
ln10 = np.log(10.0)

Rsun = 0.00465  # AU
Rearth = 4.2635212e-5  # AU
Mjup = 0.000954922  # solar masses
Mearth = 3.00447e-6  # solar masses
rEsun = 2.85412

def setup_obs_groups(paramfile, event):
    # Parse the observatory groups string
    start = 0
    rest = paramfile.obsgroupstr
    tmp = []
    
    while start is not None:
        # Groups are contained within parentheses, unless special codes
        start = rest.find("(")
        # Should be 0 unless a special multi-group codeword
        if paramfile.verbosity > 1:
            print(f"setupObsGroups: start = {start}")

        if start > 0 or start == -1:
            special = rest[:start]
            if "PERMUTE_PAIRS" in special:
                for obsidx in range(paramfile.numobservatories):
                    for obsjdx in range(obsidx + 1, paramfile.numobservatories):
                        pair = [obsidx, obsjdx]
                        event.obsgroups.append(pair)
            if "PERMUTE_REMOVE" in special:
                for obsidx in range(paramfile.numobservatories - 1, -1, -1):
                    set_ = [obsjdx for obsjdx in range(paramfile.numobservatories) if obsjdx != obsidx]
                    event.obsgroups.append(set_)
            if "EACH_INDIVIDUAL" in special:
                for obsidx in range(paramfile.numobservatories):
                    set_ = [obsidx]
                    event.obsgroups.append(set_)

        if start != -1:
            rest = rest[start + 1:]
            end = rest.find(")")
            thisgrp = rest[:end - start]
            event.obsgroups.append(tmp)

            if paramfile.verbosity > 1:
                print("setupObsGroups: start, rest, end, thisgrp")
                print(f"setupObsGroups: {start}\t{rest}\t{end}\t{thisgrp}")

            # Elements are separated by commas
            elstart = -1
            while elstart != -1:
                comma = thisgrp.find(",", elstart + 1)
                if comma == -1:
                    elstr = thisgrp[elstart + 1:comma]
                else:
                    elstr = thisgrp[elstart + 1:comma - elstart - 1]

                if paramfile.verbosity > 1:
                    print("setupObsGroups: elstart, comma, elstr")
                    print(f"setupObsGroups: {elstart}\t{comma}\t{elstr}")

                elstart = comma
                if "ALL" in elstr:
                    event.obsgroups[-1].clear()
                    for obsidx in range(paramfile.numobservatories):
                        event.obsgroups[-1].append(obsidx)
                    elstart = -1
                else:
                    candidate = int(elstr)
                    if (0 <= candidate < paramfile.numobservatories):
                        event.obsgroups[-1].append(candidate)
                    else:
                        print(f"Invalid observatory number in group {len(event.obsgroups) - 1} ({elstr})")
                        exit(1)

            rest = rest[end + 1:]

    if paramfile.verbosity:
        print("Observing groups specified:")
        for i, group in enumerate(event.obsgroups):
            print(f"\tGroup {i}\n\t\t{' '.join(map(str, group))}")

def setup_parallax(tref, paramfile, world, event, sources, lenses):
    sn = event.source
    ln = event.lens

    for obsidx in range(paramfile.numobservatories):
        # Prepare the epochs correctly
        jdepochs = [paramfile.simulation_zerotime]

        if len(jdepochs) > 0:
            event.pllx[obsidx].reset()
            event.pllx[obsidx].set_reference(paramfile.simulation_zerotime + tref, world[0].orbit)
            event.pllx[obsidx].set_orbit(world[obsidx].orbit)
            event.pllx[obsidx].set_lb(event.l, event.b)
            event.pllx[obsidx].set_pm_lb(lenses.data[ln][MUL] - sources.data[sn][MUL], lenses.data[ln][MUB] - sources.data[sn][MUB])
            event.pllx[obsidx].set_piE(event.piE)
            event.pllx[obsidx].set_tE_h(event.tE_h)

            # Do this for a dummy epoch now, but redo this at the end of timeSequencer
            event.pllx[obsidx].load_epochs(world[obsidx].jd)
            event.pllx[obsidx].initialize()

            if paramfile.verbosity > 3:
                for idx in range(len(world[obsidx].jd)):
                    print(f"NEShift event {event.id} obs {obsidx} {world[obsidx].jd[idx]:14.6f} {event.pllx[obsidx].NEshift[idx][0] + (event.pllx[obsidx].epochs[idx] - event.pllx[obsidx].tref) * event.pllx[obsidx].vref[0]} {event.pllx[obsidx].NEshift[idx][1] + (event.pllx[obsidx].epochs[idx] - event.pllx[obsidx].tref) * event.pllx[obsidx].vref[1]}")

        if paramfile.verbosity > 0:
            print(f"Observatory {obsidx}:")
            print("Orbit:")
            print(f"{world[obsidx].orbit[0].xh} {world[obsidx].orbit[0].yh} {world[obsidx].orbit[0].zh}")
            print(f"tref = {event.pllx[obsidx].tref}")
            print(f"l,b,a,d = {event.pllx[obsidx].l} {event.pllx[obsidx].b} {event.pllx[obsidx].a} {event.pllx[obsidx].d}")
            print(f"mul, mub, mua, mud = {event.pllx[obsidx].mul} {event.pllx[obsidx].mub} {event.pllx[obsidx].mua} {event.pllx[obsidx].mud}")
            print(f"piEN, piEE, piEll, piErp, piE = {event.pllx[obsidx].piEN} {event.pllx[obsidx].piEE} {event.pllx[obsidx].piEll} {event.pllx[obsidx].piErp} {event.pllx[obsidx].piE}")
            print(f"tE_h, tE_r = {event.pllx[obsidx].tE_h} {event.pllx[obsidx].tE_r}")

    event.piEN = event.pllx[0].piEN * event.piE
    event.piEE = event.pllx[0].piEE * event.piE
    event.tE_h = event.pllx[0].tE_h
    event.tE_r = event.pllx[0].tE_r

def fold(x, min_val, max_val):
    if max_val < min_val:
        min_val, max_val = max_val, min_val

    if min_val == max_val:
        return x  # wtf are you doing?

    if min_val <= x < max_val:
        return x

    r = max_val - min_val

    if x < min_val:
        diff = x - min_val
        N = np.floor(diff / r)
        y = x - N * r
    elif x >= max_val:
        if x == max_val:
            y = min_val
        else:
            diff = x - max_val
            N = np.ceil(diff / r)
            y = x - N * r

    return y

class Parallax:
    def __init__(self):
        self.status_refframe = False
        self.status_orbit = False
        self.status_position = False
        self.status_sourcedir = False
        self.status_parallax = False
        self.status_params = False
        self.status_epochs = False
        self.status_pllxinitialized = False
        self.progress = [0] * 16

    def initialize(self):
        self.setup_reference_frame()
        self.compute_NEshifts()
        self.fit_reinit()

    def fit_reinit(self):
        # Reinitialize assuming only tE and parallax vector has changed
        self.compute_directions()
        self.status_pllxinitialized = True
        self.compute_tushifts()

    def reset(self):
        self.epochs = []
        self.NEshift = []
        self.tushift = []
        self.progress = [0] * 16
        self.status_refframe = False
        self.status_orbit = False
        self.status_position = False
        self.status_sourcedir = False
        self.status_parallax = False
        self.status_params = False
        self.status_epochs = False
        self.status_pllxinitialized = False

    def print_uninit(self):
        if not self.status_refframe:
            print("REFFRAME Uninitialized")
        if not self.status_orbit:
            print("ORBIT Uninitialized")
        if not self.status_position:
            print("POSITION Uninitialized")
        if not self.status_sourcedir:
            print("SOURCEDIR Uninitialized")
        if not self.status_parallax:
            print("PARALLAX Uninitialized")
        if not self.status_params:
            print("PARAMS Uninitialized")
        if not self.status_epochs:
            print("EPOCHS Uninitialized")
        if not self.status_pllxinitialized:
            print("PLLXINITIALIZED Uninitialized")

    def compute_NEshifts(self):
        if not (self.status_refframe and self.status_orbit and self.status_position and self.status_sourcedir and self.status_parallax and self.status_params and self.status_epochs):
            print(f"{__file__}: {__name__}: Error: Some element of the parallax computation is uninitialized:")
            self.print_uninit()
            exit(1)

        self.NEshift = np.zeros((len(self.epochs), 2))

        for i in range(len(self.epochs)):
            x = np.zeros(3)
            for j in range(len(self.orbit)):
                xp = self.orbit[j].viewfrom(self.epochs[i], self.a, self.d)
                x += xp
            self.NEshift[i][0] = x[0] - self.xref[0] - (self.epochs[i] - self.tref) * self.vref[0]
            self.NEshift[i][1] = x[1] - self.xref[1] - (self.epochs[i] - self.tref) * self.vref[1]

    def compute_tushifts(self):
        if not self.status_pllxinitialized:
            print(f"{__file__}: {__name__}: Error: Some element of the parallax computation is uninitialized")
            exit(1)

        self.tushift = np.zeros((len(self.epochs), 2))

        cs = np.cos(self.phi_pi)
        sn = np.sin(self.phi_pi)

        for i in range(len(self.epochs)):
            self.tushift[i][0] = -self.piE * (self.NEshift[i][0] * cs + self.NEshift[i][1] * sn)
            self.tushift[i][1] = -self.piE * (-self.NEshift[i][0] * sn + self.NEshift[i][1] * cs)

    def setup_reference_frame(self):
        exitnow = False

        if not self.status_refframe:
            print(f"{__file__}: Error: Reference frame not set")
            exitnow = True

        if not self.status_orbit:
            print(f"{__file__}: Error: Orbit not set")
            exitnow = True

        if not self.status_position:
            print(f"{__file__}: Error: Event position not set")
            exitnow = True

        if exitnow:
            exit(1)

        self.xref = np.zeros(3)
        self.vref = np.zeros(3)
        self.aref = np.zeros(3)

        for i in range(len(self.oref)):
            xp = self.oref[i].viewfrom(self.tref, self.a, self.d)
            vp = self.oref[i].velocity(self.tref, self.a, self.d)
            ap = self.oref[i].acceleration(self.tref, self.a, self.d)
            self.xref += xp
            self.vref += vp
            self.aref += ap

    def compute_directions(self):
        exitnow = False

        if not self.status_refframe:
            print(f"{__file__}: Error: Reference frame not set")
            exitnow = True

        if not self.status_orbit:
            print(f"{__file__}: Error: Orbit not set (either through piE components or proper motions).")
            exitnow = True

        if not self.status_position:
            print(f"{__file__}: Error: Event position not set")
            exitnow = True

        if not self.status_sourcedir:
            print(f"{__file__}: Error: Source direction not set (either through piE components or proper motions).")
            exitnow = True

        if not self.status_params:
            print(f"{__file__}: Error: Event parameters not set.")
            exitnow = True

        if exitnow:
            exit(1)

        wref = np.zeros(2)
        wref[0] = self.vref[0] * self.piE
        wref[1] = self.vref[1] * self.piE

        self.phi_llN = np.atan2(-self.aref[1], -self.aref[0])

        if self.progress[LPARAMS] == 1 and self.progress[LSOURCEDIR] == 1:
            wh = np.zeros(2)
            wr = np.zeros(2)
            wh[0] = self.mud / self.tE_h
            wh[1] = self.mua / self.tE_h
            wr[0] = wh[0] - wref[0]
            wr[1] = wh[1] - wref[1]
            self.tE_r = 1.0 / np.hypot(wr[0], wr[1])
            self.piEN = wr[0] * self.tE_r
            self.piEE = wr[1] * self.tE_r
            self.piEll = self.piEN * np.cos(-self.phi_llN) - self.piEE * np.sin(-self.phi_llN)
            self.piErp = self.piEN * np.sin(-self.phi_llN) + self.piEE * np.cos(-self.phi_llN)
            self.progress[LSOURCEDIR] = 2
        elif self.progress[LPARAMS] == 2 and self.progress[LSOURCEDIR] > 1:
            if self.progress[LSOURCEDIR] == 2:
                self.piEll = self.piEN * np.cos(-self.phi_llN) - self.piEE * np.sin(-self.phi_llN)
                self.piErp = self.piEN * np.sin(-self.phi_llN) + self.piEE * np.cos(-self.phi_llN)
            elif self.progress[LSOURCEDIR] == 3:
                self.piEN = self.piEll * np.cos(self.phi_llN) - self.piErp * np.sin(self.phi_llN)
                self.piEE = self.piEll * np.sin(self.phi_llN) + self.piErp * np.cos(self.phi_llN)
            wr = np.zeros(2)
            wh = np.zeros(2)
            wr[0] = self.piEN / self.tE_r
            wr[1] = self.piEE / self.tE_r
            wh[0] = wr[0] + wref[0]
            wh[1] = wr[1] + wref[1]
            self.tE_h = 1.0 / np.hypot(wh[0], wh[1])
            self.mud = wh[0] * self.tE_h
            self.mua = wh[1] * self.tE_h
            self.mul, self.mub = self.ad2lb(self.a, self.d, self.mua, self.mud)
        else:
            print(f"{__file__}: {__name__}: Error: Must specify parameters and source directions as heliocentric or reference-frame-centric")

        self.phi_pi = np.atan2(self.piEE, self.piEN)

    def set_reference(self, tref_, oref_):
        self.tref = tref_
        self.oref = oref_
        self.status_refframe = True
        return self.status_refframe

    def set_orbit(self, orbit_):
        self.orbit = orbit_
        self.status_orbit = True
        return self.status_orbit

    def set_radec(self, ra, dec, deghr):
        if deghr > 2:
            print(f"{__file__}: {__name__}: Error setting position - bad angle unit code ({deghr})")
            exit(1)

        if deghr == 0:
            self.a = fold(ra, 0, 2 * np.pi)
            self.d = fold(dec, -np.pi / 2, np.pi / 2)
        elif deghr == 1:
            self.a = fold(np.deg2rad(ra), 0, 2 * np.pi)
            self.d = fold(np.deg2rad(dec), -np.pi / 2, np.pi / 2)
        else:
            self.a = fold(np.deg2rad(ra * 15), 0, 2 * np.pi)
            self.d = fold(np.deg2rad(dec), -np.pi / 2, np.pi / 2)

        self.l, self.b = self.ad2lb(self.a, self.d)
        self.status_position = True
        return self.status_position

    def set_radec_sexagesimal(self, ra, dec):
        if len(ra) != 3 or len(dec) != 3:
            print(f"{__file__}: {__name__}: Error setting position - input sexagesimal vectors (ra_hr, ra_min, ra_sec), (dec_deg, dec_min, dec_sec)")
            exit(1)

        sgn_a = -1 if ra[0] < 0 else 1
        sgn_d = -1 if dec[0] < 0 else 1
        a_ = np.deg2rad((ra[0] + sgn_a * (ra[1] / 60.0 + ra[2] / 3600.0)) * 15.0)
        d_ = np.deg2rad(dec[0] + sgn_d * (dec[1] / 60.0 + dec[2] / 3600.0))

        self.a = fold(a_, 0, 2 * np.pi)
        self.d = fold(d_, -np.pi / 2, np.pi / 2)

        self.l, self.b = self.ad2lb(self.a, self.d)
        self.status_position = True
        return self.status_position

    def set_lb(self, l_, b_, deg):
        if deg > 1:
            print(f"{__file__}: {__name__}: Error setting position - bad angle unit code ({deg})")
            exit(1)

        if deg == 0:
            self.l = fold(np.deg2rad(l_), 0, 2 * np.pi)
            self.b = fold(np.deg2rad(b_), -np.pi / 2, np.pi / 2)
        else:
            self.l = fold(l_, 0, 2 * np.pi)
            self.b = fold(b_, -np.pi / 2, np.pi / 2)

        self.a, self.d = self.lb2ad(self.l, self.b)
        self.status_position = True
        return self.status_position

    def set_pm_lb(self, mul_, mub_):
        if not self.status_position:
            print(f"{__file__}: {__name__}: Must set position in order to do proper motion unit conversions")
            exit(1)

        mu = np.hypot(mul_, mub_)
        self.mul = mul_ / mu
        self.mub = mub_ / mu
        self.mua, self.mud = self.lb2ad(self.l, self.b, self.mul, self.mub)

        self.progress[LSOURCEDIR] = 1
        self.status_sourcedir = True
        return self.status_sourcedir

    def set_pm_ad(self, mua_, mud_):
        if not self.status_position:
            print(f"{__file__}: {__name__}: Must set position in order to do proper motion unit conversions")
            exit(1)

        mu = np.hypot(mua_, mud_)
        self.mua = mua_ / mu
        self.mud = mud_ / mu
        self.mul, self.mub = self.ad2lb(self.a, self.d, self.mua, self.mud)

        self.progress[LSOURCEDIR] = 1
        self.status_sourcedir = True
        return self.status_sourcedir

    def set_piEpp(self, piEll_, piErp_):
        self.piE = np.hypot(piEll_, piErp_)
        self.piEll = piEll_ / self.piE
        self.piErp = piErp_ / self.piE

        self.progress[LSOURCEDIR] = 3
        self.status_parallax = True
        self.status_sourcedir = True
        return self.status_parallax

    def set_piE(self, piE_):
        self.piE = piE_
        self.status_parallax = True
        return self.status_parallax

    def set_piENE(self, piEN_, piEE_):
        self.piE = np.hypot(piEN_, piEE_)
        self.piEN = piEN_ / self.piE
        self.piEE = piEE_ / self.piE

        self.progress[LSOURCEDIR] = 2
        self.status_parallax = True
        self.status_sourcedir = True
        return self.status_parallax

    def set_tE_h(self, tE):
        self.tE_h = tE
        self.progress[LPARAMS] = 1
        self.status_params = True
        return self.status_params

    def set_tE_r(self, tE):
        self.tE_r = tE
        self.progress[LPARAMS] = 2
        self.status_params = True
        return self.status_params

    def load_epochs(self, epochs_):
        self.epochs = epochs_
        if len(epochs_) > 0:
            self.status_epochs = True
        return self.status_epochs

    def load_epochs_vector(self, epochs_):
        self.epochs = epochs_
        if len(epochs_) > 0:
            self.status_epochs = True
        return self.status_epochs

    def ad2lb(self, ra, dec):
        coord = SkyCoord(ra=ra * u.radian, dec=dec * u.radian, frame='icrs')
        l = coord.galactic.l.radian
        b = coord.galactic.b.radian
        return l, b

    def lb2ad(self, l, b):
        coord = SkyCoord(l=l * u.radian, b=b * u.radian, frame='galactic')
        ra = coord.icrs.ra.radian
        dec = coord.icrs.dec.radian
        return ra, dec

    def lb2ad_pm(self, l, b, mul, mub):
        coord = SkyCoord(l=l * u.radian, b=b * u.radian, pm_l_cosb=mul * u.mas/u.yr, pm_b=mub * u.mas/u.yr, frame='galactic')
        ra = coord.icrs.ra.radian
        dec = coord.icrs.dec.radian
        pm_ra_cosdec = coord.icrs.pm_ra_cosdec.value
        pm_dec = coord.icrs.pm_dec.value
        return ra, dec, pm_ra_cosdec, pm_dec

    def ad2lb_pm(self, ra, dec, pm_ra_cosdec, pm_dec):
        coord = SkyCoord(ra=ra * u.radian, dec=dec * u.radian, pm_ra_cosdec=pm_ra_cosdec * u.mas/u.yr, pm_dec=pm_dec * u.mas/u.yr, frame='icrs')
        l = coord.galactic.l.radian
        b = coord.galactic.b.radian
        pm_l_cosb = coord.galactic.pm_l_cosb.value
        pm_b = coord.galactic.pm_b.value
        return l, b, pm_l_cosb, pm_b