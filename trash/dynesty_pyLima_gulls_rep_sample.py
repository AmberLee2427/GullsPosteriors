import os
import sys
import dynesty
import numpy as np
from pyLIMA import event
from pyLIMA import telescopes
import multiprocessing as mp
from astropy import units as u
import astropy.coordinates as astrocoords

#Here is where parallax mags are computed
#bozzaPllxOMLCGen.cpp
 
#Here is where chi2 calculated
#pllxLightcurveFitter.cpp

# debug event: 331
# N = 6757
# chi2 = 6762 - binary
# chi2 = 6857 - single
# delta chi2 = 95

nevents = int(sys.argv[1])
if len(sys.argv) == 2:
    sort = sys.argv[2]
else:
    sort = 'alphanumeric'
ndim = 11

# limb darkening coefficients !!!
# 0.36 in all bands - linear limb darkening
# W146, Z087, K213

# probably dynesty would make more sense for posterior sampling because it will have z values.

parameters_to_fit = ['t_0', 'u_0', 't_E', 'rho', 'q', 's', 'rho', 'alpha', 'pi_E_N', 'pi_E_E', 'ds_dt', 'dalpha_dt', 'dzdt']
# check with Mathew about the model used to evaluate the BL chi2  !!!
# check with mathew about the parallax stuff. !!!

def lnlike(theta, event):
    for (parameter, value) in zip(parameters_to_fit, theta):
        setattr(event.model.parameters, parameter, value)

    # After that, calculating chi2 is trivial:
    chi2 = event.get_chi2()
    return -0.5 * chi2

def lnprior(params, event):
    logs, logq, logrho, u0, alpha, t0, tE, piE, piN, dsdt, dalphadt, dzdt = params
    q = 10.0**logq
    if tE > 0.0 and q <= 1.0:
        return 0.0
    return -np.inf

def lnprob(params, event):
    # prior
    lp = lnprior(params, event)
    if not np.isfinite(lp):
        return -np.inf
    
    # likelihood
    ll = lnlike(params, event)
    if not np.isfinite(ll):
        return -np.inf
    
    # prob
    return lp + ll

def prior_transform(utheta):
    """Transform unit cube to the parameter space"""
    logs, logq, logrho, u0, alpha, t0, tE, piE, piN, dsdt, dalphadt, dzdt = utheta
    logs = -2 + 4 * logs  # example range for logs
    logq = -3 + 6 * logq  # example range for logq
    logrho = -3 + 6 * logrho  # example range for logrho
    u0 = -1 + 2 * u0  # example range for u0
    alpha = 0 + 360 * alpha  # example range for alpha
    t0 = 0 + 100 * t0  # example range for t0
    tE = 0 + 100 * tE  # example range for tE
    piE = -1 + 2 * piE  # example range for piE
    piN = -1 + 2 * piN  # example range for piN
    dsdt = -1 + 2 * dsdt  # example range for dsdt
    dalphadt = -1 + 2 * dalphadt  # example range for dalphadt
    dzdt = -1 + 2 * dzdt  # example range for dzdt
    return logs, logq, logrho, u0, alpha, t0, tE, piE, piN, dsdt, dalphadt, dzdt
    ## SAVE !!!

def new_event(sort):
    '''get the data and true params for the next event'''

    if sort == 'alphanumeric':
        files = os.listdir('data')
        files = sorted(files)
        if os.path.exists('runlist.npy'):
            runlist = np.loadtxt('runlist.npy')
        else:
            runlist = np.array([])
        for i in range(len(files)):
            if files[i] not in runlist:
                runlist = np.vstack(files[i])
                np.savetxt('runlist.txt', runlist, fmt='%s')
                data = mm.MulensData(file_name='data/' + files[i])
                true_params = np.loadtxt('true_params/' + files[i].split('.')[0] + '.txt')
                break

    if ".txt" in sort:
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
                break
        
    os.mkdir('posteriors/' + files[i].split('.')[0])

    return true_params, ra, dec, data_file

# The lightcurve columns are:
# Simulation_time measured_relative_flux measured_relative_flux_error true_relative_flux 
# true_relative_flux_error observatory_code saturation_flag best_single_lens_fit parallax_shift_t 
# parallax_shift_u BJD source_x source_y lens1_x lens1_y lens2_x lens2_y
# Magnitudes can be computed using:
# m = m_source + 2.5 log f_s - 2.5 log(F)
# where F=fs*mu + (1-fs) is the relative flux (in the file), mu is the magnification, and
# sigma_m = 2.5/ln(10) sigma_F/F
# these are listed in the header information in lines #fs and #Obssrcmag with order matching the observatory code order. 
# The observatory codes correspond to 0=W146, 1=Z087, 2=K213
# Bugs/issues/caveats:
# The output file columns list a limb darkening parameter of Gamma=0, it is actually Gamma=0.36 (in all filters)
# The orbit for the Z087 observatory appears to be different to the W146 and K213 observatory
# I'm working on producing the ephemerides, but for single observatory parallax, using interpolated versions of the ones 
# available for the data challenge will probably be accurate enough, or an Earth ephemeris with the semimajor axis (but 
# not period) increased by 0.01 AU
# Lenses with masses smaller than the isochrone grid limits (I believe 0.1 MSun, will have filler values for magnitudes 
# and lens stellar properties).
# There may be some spurious detections in the list where the single lens fit failed. Please let us know if you find any 
# of these events so that we can improve the single lens fitter.

for i in range (nevents):
    true_params, ra, dec, data_file = new_event(sort)

    logs, logq, logrho, u0, alpha, t0, tE, piE, piN, dsdt, dalphadt, dzdt = true_params
    tE = tE * u.day
    s = 10.0**logs
    q = 10.0**logq
    rho = 10.0**logrho
    alpha = alpha * u.deg
    dsdt = dsdt / u.year
    dalphadt = dalphadt * u.rad/u.day
    t_0_par = t0
    p_dic = {'t_0': t0, 
                     'u_0': u0, 
                     't_E': tE, 
                     'rho': rho, 
                     'q': q, 
                     's': s, 
                     'alpha': alpha,
                     'pi_E_N': piN, 
                     'pi_E_E': piE, 
                     't_0_par': t_0_par,
                     'ds_dt': dsdt,
                     'dalpha_dt': dalphadt}

    gulls_event = event.Event()
    name = data_file.split('.')[0]
    gulls_event.name = name
    data = np.loadtxt('./repsample/OMPLLD_croin_cassan_2_0_312.det.lc')
    simulation_time = data[:,0]
    date = simulation_time + HJD(2018-08-16)
    mag = data[:,1]

    W146 = telescopes.Telescope(name = 'Roman0', 
                                   camera_filter = 'I',
                                   light_curve = data,
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

    sampler = dynesty.NestedSampler(
                                    lnprob, 
                                    prior_transform, 
                                    ndim, 
                                    nlive=500, 
                                    sample='rwalk', 
                                    bound='multi', 
                                    pool=mp.Pool(mp.cpu_count())
                                    )
    sampler.run_nested()

    print('Event', i, 'done')

# ephermeris
# 2018-08-10
# 2023-04-30
# get emphemeris file from JPL Horizons for SEMB-L2

# MulensModel.satelliteskycoord.SatelliteSkyCoord(ephemerides_file, satellite='Roman')
