#import multiprocessing
#multiprocessing.set_start_method('fork', force=True)
import warnings
#warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked semaphore objects")

import os
import sys
import emcee
import numpy as np
from astropy import units as u
import astropy.coordinates as astrocoords
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.time import Time
from scipy.interpolate import interp1d
from astroquery.jplhorizons import Horizons
import pandas as pd
import pickle
import corner
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import time
from multiprocessing import Pool
import multiprocessing as mp


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
    


# 2 pi radians / period
# phase is an angle in radians
# phase0 is the phase at simulation 0 time
# assume a circular orbit and a small mass ratio m << M


    

if __name__ == '__main__':

    start_time = time.time()
    print('Start time = ', start_time)

    nevents = int(sys.argv[1])

    #if '-e' in sys.argv:
    #    engine_index = sys.argv.index('-e') + 1
    #    engine = sys.argv[engine_index]
    #else:
    #   engine = 'VBM'  # default to VBM

    if '-s' in sys.argv:
        sampler_index = sys.argv.index('-s') + 1
        sampling_package = sys.argv[sampler_index]
    else:
        sampling_package = 'emcee'

    if '-t' in sys.argv:
        threads_index = sys.argv.index('-t') + 1
        threads = int(sys.argv[threads_index])
        if threads == 0:
            threads = mp.cpu_count()
    else:
        pooling = False

    path = sys.argv[2]  # put some error handling in
    if path[-1] != '/':
        path = path + '/'

    if len(sys.argv) == 4:
        sort = sys.argv[3]
    else:
        sort = 'alphanumeric'
    ndim = 12

    from Data import Data
    from Parallax import Parallax
    from Event import Event
    from Fit import Fit
    from Orbit import Orbit
    from Fit._dynesty import prior_transform

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

        tc_calc = event_tc.croin(t0, u0, s, q, alpha, tE)
        event_tref = Event(parallax, orbit, data,
                           truths, data_structure.sim_time0, tc_calc
                          )

        fit = Fit(sampling_package=sampling_package)

        chi2_ew_t0, chi2_t0 = fit.get_chi2(event_t0, truths['params'])
        chi2_ew_tc, chi2_tc = fit.get_chi2(event_tc, truths['params'])
        chi2_ew_tref, chi2_tref = fit.get_chi2(event_tref, truths['params'])

        tmin = np.min([t0-2.0*tE, tc_calc-2.0*tE])
        tmax = np.max([t0+2.0*tE, tc_calc+2.0*tE])
        points = np.where(np.logical_and(t_data[0] > tmin, t_data[0] < tmax))

        chi2_t0_0 = [np.sum(chi2_ew_t0[0]), np.sum(chi2_ew_t0[0][points])]
        cumsum_chi2_t0 = np.cumsum(chi2_ew_t0[0])
        chi2_tc_0 = [np.sum(chi2_ew_tc[0]), np.sum(chi2_ew_tc[0][points])]
        cumsum_chi2_tc = np.cumsum(chi2_ew_tc[0])
        chi2_tref_0 = [np.sum(chi2_ew_tref[0]), np.sum(chi2_ew_tref[0][points])]
        cumsum_chi2_tref = np.cumsum(chi2_ew_tref[0])

        end_preabmle = time.time()
        print('Time to get data = ', end_preabmle - start_time)

        N = cumsum_chi2_t0.shape[0]
        n = chi2_ew_t0[0][points].shape[0]

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


        A = {}  # t is data epochs
        A_lin = 0  # linearly spaced time
        A_true = {}
        n = chi2_ew_t0[0][points].shape[0]  # number of points from obs 0 within +-5tE to t0
        nn = 10000  # number of model points
        fb = {}
        t0 = event_tc.true_params[5]
        tE = event_tc.true_params[6]
        ttmin = np.max([t0-5.0*tE, tc_calc-5.0*tE])
        ttmax = np.max([t0+5.0*tE, tc_calc+5.0*tE])
        tt = np.linspace(ttmin, ttmax, nn)
        fs = {}
        res = {}

        events = {'t0':event_t0, 'tc':event_tc}

        rho = event_tc.true_params[2]

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

                if chi2_t0_true0[1] <= chi2_tc_true0[1]:
                    fit_tref = t0
                else:
                    fit_tref = truths['tcroin']

            res[obs] = A_true[obs] - A[obs]
            
        np.savetxt(path+'posteriors/'+event_name+'chi2_true.txt', np.array([chi2_t0_true0, chi2_tc_true0, chi2_tref_true0]), fmt='%1.0f')
        
        A_lin = event_tc.get_magnification(tt, obs)
        A_lin = event_t0.get_magnification(tt, obs)
        A_lin = event_tref.get_magnification(tt, obs)

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

        s_t0, _, _ = event_tc.projected_seperation(i, period, t0, phase_offset=phase0, t_start=t0, a = a)
        s_tc, _, _ = event_tc.projected_seperation(i, period, tc, phase_offset=phase0, t_start=t0, a = a)
        s_tref, _, _ = event_tc.projected_seperation(i, period, tc_calc, phase_offset=phase0, t_start=t0, a = a)
        print(s_tc, s_t0, s_tref)
        
        t = event_tc.data[0][0]  # BJD
        tmax = np.max([t0+5.0*tE, tc+2.0*tE, tc_calc+2.0*tE])
        tmin = np.min([t0-5.0*tE, tc-2.0*tE, tc_calc-2.0*tE])
        points = np.where(np.logical_and(t > tmin, t < tmax))
        
        u0 = event_tc.true_params[3]
        alpha = event_tc.true_params[4]
        xcom = truths['xCoM']
        x0 = -u0 * np.sin(alpha) - xcom
        y0 = u0 * np.cos(alpha)
        x0_c = -u0 * np.sin(alpha) + (event_tc.t_ref-t0)/tE * np.cos(alpha) - xcom
        y0_c = u0 * np.cos(alpha) + (event_tc.t_ref-t0)/tE * np.sin(alpha)
        x0_ref = -u0 * np.sin(alpha) + (event_tref.t_ref-t0)/tE * np.cos(alpha) - xcom
        y0_ref = u0 * np.cos(alpha) + (event_tref.t_ref-t0)/tE * np.sin(alpha)
        
        xx = [1.0, -1.0, x0, x0_c, x0_ref]
        yy = [1.0, -1.0, y0, y0_c, y0_ref]

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

        prior_transform(fit, u, event_tc.truths['params'], prange=prange, normal=True)
        prior_transform(fit, u, event_tc.truths['params'], prange=prange)

        sys.exit() #'''





        print()
        print('Sampling Posterior using emcee')
        print('--------------------------------')

        normal = True
        # Define the number of walkers and the initial positions of the walkers
        nl = 50  # number of live walkers
        mi = 1000  # max iterations
        stepi = 50  # steps between saving the sampler 
        u0 = np.ones((nl, ndim)) * 0.5
        initial_pos = u0 + 1e-4 * np.random.rand(nl, ndim)
        labels = ['s', 'q', 'rho', 'u0', 'alpha', 't0', 'tE', 'piEE', 'piEN', 'i', 'phase', 'period']
        bounds = prange*2.0

        #print('Debug main: pos = ', initial_pos, initial_pos.shape)
        initial_state = prior_transform(fit, initial_pos, truths['params'], bounds, normal)
        #print('Debug main: state = ', initial_state, initial_state.shape)

        # Cropping the data to near-event
        tminmax = [fit_tref-1.0*tE, fit_tref+1.0*tE, t0-1.0*tE, t0+1.0*tE]
        tmin = np.min(tminmax)
        tmax = np.max(tminmax)
        points = np.where(np.logical_and(t_data[0] > tmin, t_data[0] < tmax))

        data_cropped = {}
        for obs in data.keys():
            if obs == 0:
                data_obs = data[obs].T  # this selection results in an extra dimension without the T's
                data_cropped[obs] = data_obs[points].T  

        # New event object with cropped data
        event_fit = Event(parallax, orbit, data_cropped, 
                         truths, data_structure.sim_time0, fit_tref
                         )

        
        pos = initial_state.copy()
        sampler = fit.run_emcee(nl, ndim, stepi, mi, fit.lnprob, pos, event_fit, threads=1, event_name=event_name, path=path, labels=labels)

                
        #print('Debug main: pos = ', pos, pos.shape)
        #print('Debug main: state = ', initial_state, initial_state.shape)
        #fit.debug = ['lnprob']

        end_dynesty = time.time()
        print('Time to run emcee (nl, mi)= ', end_dynesty - end_initial_figures, nl, mi)

        res = sampler.results
        res.samples = prior_transform(fit, res.samples, truths['params'], prange, normal=normal)

        # print for logs
        print('Event', i, '(', event_name, ') is done')
        print(res.summary())


        # Save plots
        fit.corner_post(res, event_name, path, truths)
        #fit.runplot(res, event_name, path)
        #fit.traceplot(res, event_name, path, truths)

        samples = res.samples
        np.save(path+'posteriors/'+event_name+'_post_samples.npy', samples)

        with open(path+'posteriors/'+event_name+'end_truths.pkl', 'wb') as f:
            pickle.dump(event_tc.truths, f)

        #'''
        # Done with the event
        if not os.path.exists(path+'emcee_complete.txt'):
            complete_list = np.array([])
            np.savetxt(path+'emcee_complete.txt', complete_list, fmt='%s')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            complete_list = np.loadtxt(path+'emcee_complete.txt', dtype=str)

            complete_list = np.hstack([complete_list, event_name])
            print('Completed:', event_name)
            np.savetxt(path+'emcee_complete.txt', complete_list, fmt='%s')
        #sampler.reset()

        end_time = time.time()

        print('\n\nTime Summary')
        print('--------------------------------')
        print('Start time = ', start_time)
        print('Time to get data = ', end_preabmle - start_time)
        print('Time to make initial figures = ', end_initial_figures - end_preabmle)
        print('Time to run emcee (nl, mi)= ', end_dynesty - end_initial_figures, nl, mi)
        print('Time to wrap up = ', end_time - end_dynesty)
        print('Total time = ', end_time - start_time)
        print('End time = ', end_time)
        print('--------------------------------\n\n\n')

# ephermeris
# 2018-08-10
# 2023-04-30
# get emphemeris file from JPL Horizons for SEMB-L2