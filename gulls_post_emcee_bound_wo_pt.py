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

    # script options
    #------------------------------------------------
    nevents = int(sys.argv[1])

    #if '-e' in sys.argv:
    #    engine_index = sys.argv.index('-e') + 1
    #    engine = sys.argv[engine_index]
    #else:
    #   engine = 'VBM'  # default to VBM

    # -s sampler
    if '-s' in sys.argv:
        sampler_index = sys.argv.index('-s') + 1
        sampling_package = sys.argv[sampler_index]
    else:
        sampling_package = 'emcee'

    # -t threads
    if '-t' in sys.argv:
        threads_index = sys.argv.index('-t') + 1
        threads = int(sys.argv[threads_index])
        if threads == 0:
            threads = mp.cpu_count()
    else:
        threads = 1  # Default to single thread if not specified
        pooling = False

    # default plotting options
    if sampling_package == 'dynesty':
        plot_chain = False
        plot_trace = True
        plot_post = False
        plot_initial_figures = True
        plot_final_figures = True
        plot_run = True
    if sampling_package == 'emcee':
        plot_chain = True
        plot_trace = False
        plot_post = True
        plot_initial_figures = True
        plot_final_figures = True
        plot_run = False

    # -f plotting options
    if '-f' in sys.argv:
        plot_index = sys.argv.index('-p') + 1

        if 'n' in sys.argv[plot_index]:
            plot_initial_figures = False
            plot_final_figures = False
            plot_chain = False
            plot_trace = False
            plot_post = False
            plot_run = False

        if 'i' in sys.argv[plot_index]:
            plot_initial_figures = True
        else:
            plot_initial_figures = False

        if 'c' in sys.argv[plot_index]:
            plot_chain = True
        else:
            plot_chain = False

        if 't' in sys.argv[plot_index]:
            plot_trace = True
        else:
            plot_trace = False

        if 'p' in sys.argv[plot_index]:
            plot_post = True
        else:
            plot_post = False

        if 'f' in sys.argv[plot_index]:
            plot_final_figures = True
            # these options are turned on by f, even if not explicity set
        if plot_final_figures:
            if sampling_package == 'dynesty':
                plot_trace = True 
                plot_run = True
            if sampling_package == 'emcee':
                plot_post = True

    path = sys.argv[2]  # put some error handling in
    if path[-1] != '/':
        path = path + '/'

    if '-sort' in sys.argv:
        sort_index = sys.argv.index('-sort') + 1
        sort = sys.argv[sort_index]
    else:
        sort = 'alphanumeric'
    #------------------------------------------------


    ndim = 12  # number of parameters
               # This is very much hard-coded in for now


    # Importing the event modelling classes that match the gulls parameterisation
    from Data import Data  # imports data in the gulls output format
    from Parallax import Parallax  # hack replacement of the parallax calculations, calculating n^ and e^ 
                                   # from the LC file
    from Event import Event  # has the event model functions like magnification, projected_seperation, etc.
    from Fit import Fit  # has the fitting functions like get_chi2, get_fluxes, etc.
    from Orbit import Orbit  # not actually used yet
    from Fit._dynesty import prior_transform  # to get initial conditions for the emcee sampler


    # Orbit
    orbit_obj = Orbit()  # I need this to feed in to Parallax


    # Fit
    fit_obj = Fit(sampling_package=sampling_package)


    # Data infrastructure
    if not os.path.exists(path + 'posteriors/'):  # make a directory for the posteriors
            os.mkdir(path + 'posteriors/')


    for i in range (nevents):

        # Data
        data_obj = Data()
        event_name, truths, data = data_obj.new_event(path, sort)
        
        print('\n\n\n\n\n\nevent_name = ', event_name)
        print('---------------------------------------')
        print('truths = ', truths)

        # Re-packaging the data for the parallax hack and some plotting stuff
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

        # Parallax
        parallax_obj = Parallax(truths['ra_deg'], truths['dec_deg'], 
                            orbit_obj, truths['tcroin'],
                            tu_data, piE, epochs)
        parallax_obj.update_piE_NE(truths['piEN'], truths['piEE'])  # this is a hack to get the piE values

        # It is unclear which tref is being used at each step of the lightcurve generation, in gulls
        event_t0 = Event(parallax_obj, orbit_obj, data, 
                         truths, data_obj.sim_time0, truths['t0lens1']
                         )
        event_tc = Event(parallax_obj, orbit_obj, data, 
                         truths, data_obj.sim_time0, truths['tcroin']
                         )
        
        s = truths['params'][0].copy()
        q = truths['params'][1].copy()
        u0 = truths['params'][3].copy()
        alpha = truths['params'][4].copy()

        # Checking the t_croin calculation
        tc_calc = event_tc.croin(t0, u0, s, q, alpha, tE)
        event_tref = Event(parallax_obj, orbit_obj, data,
                           truths, data_obj.sim_time0, tc_calc
                          )

        # Chi2 comparison for determinging the tref being used for LOM
        #------------------------------------------------
        # full lc chi2
        chi2_ew_t0, chi2_t0 = fit_obj.get_chi2(event_t0, truths['params'])
        chi2_ew_tc, chi2_tc = fit_obj.get_chi2(event_tc, truths['params'])
        chi2_ew_tref, chi2_tref = fit_obj.get_chi2(event_tref, truths['params'])

        # cropped lc chi2
        tmin = np.min([t0-2.0*tE, tc_calc-2.0*tE])  # tc_calc is the t_croin calculated from the true params
        tmax = np.max([t0+2.0*tE, tc_calc+2.0*tE])
        points = np.where(np.logical_and(t_data[0] > tmin, t_data[0] < tmax))

        chi2_t0_0 = [np.sum(chi2_ew_t0[0]), np.sum(chi2_ew_t0[0][points])]
        cumsum_chi2_t0 = np.cumsum(chi2_ew_t0[0])
        chi2_tc_0 = [np.sum(chi2_ew_tc[0]), np.sum(chi2_ew_tc[0][points])]
        cumsum_chi2_tc = np.cumsum(chi2_ew_tc[0])
        chi2_tref_0 = [np.sum(chi2_ew_tref[0]), np.sum(chi2_ew_tref[0][points])]
        cumsum_chi2_tref = np.cumsum(chi2_ew_tref[0])

        # Timing the preamble
        end_preabmle = time.time()
        print('Time to get data = ', end_preabmle - start_time)


        # Plotting the initial figures
        #================================================

        # Cumulative chi2
        #------------------------------------------------
        N = cumsum_chi2_t0.shape[0]
        n = chi2_ew_t0[0][points].shape[0]

        if plot_initial_figures:
            plt.figure()
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
        #------------------------------------------------


        # VBM
        from VBMicrolensing import VBMicrolensing
        vbm = VBMicrolensing()
        vbm.a1 = 0.36


        # Initial lightcurve plot
        #------------------------------------------------
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

        if plot_initial_figures:
            plt.figure()
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2]})
            colours = ['orange', 'red', 'green']
            data_labels = ['W146', 'Z087', 'K213']
            # Z, W, K from shortes wavelength to longest

        rho = event_tc.true_params[2]

        for obs in event_tc.data.keys():

            # data
            t = event_tc.data[obs][0]  # BJD
            f = event_tc.data[obs][1]  # obs_rel_flux
            f_err = event_tc.data[obs][2]  # obs_rel_flux_err

            # calculating the model
            A[obs] = event_tc.get_magnification(t, obs)
            fs[obs], fb[obs] = fit_obj.get_fluxes(A[obs], f, f_err**2)
            fstrue, fbtrue = fit_obj.get_fluxes(A[obs], f_true[obs], f_err_true[obs]**2)
            A_true[obs] = (f_true[obs] - fbtrue)/fstrue
            if int(obs) == 0:
                At0 = event_t0.get_magnification(t, obs)
                fst0, fbt0 = fit_obj.get_fluxes(At0, f_true[obs], f_err_true[obs]**2)
                Atref = event_tref.get_magnification(t, obs)
                fstref, fbtref = fit_obj.get_fluxes(Atref, f_true[obs], f_err_true[obs]**2)

                res['t0'] = f_true[obs]-(fst0*At0+fbt0)  # A_true[obs] - At0
                res['tc'] =  f_true[obs]-(fstrue*A[obs]+fbtrue)  #A_true[obs] - A[obs]
                res['tref'] = f_true[obs]-(fstref*Atref+fbtref)  #A_true[obs] - Atref

                chi2_ew_t0_true0 = (res['t0'])**2 / f_err_true[obs]**2
                chi2_ew_tc_true0 = (res['tc'])**2 / f_err_true[obs]**2
                chi2_ew_tref_true0 = (res['tref'])**2 / f_err_true[obs]**2

                chi2_t0_true0 = [np.sum(chi2_ew_t0_true0), np.sum(chi2_ew_t0_true0[points])]
                chi2_tc_true0 = [np.sum(chi2_ew_tc_true0), np.sum(chi2_ew_tc_true0[points])]
                chi2_tref_true0 = [np.sum(chi2_ew_tref_true0), np.sum(chi2_ew_tref_true0[points])]

                if plot_initial_figures:
                    ax2.plot(t, res['tc'], '.', color='cyan', alpha=0.35, zorder=0, ms=1.5, label=r'$\chi^2=%1.2f$' %(chi2_tc_true0[1]))
                    ax2.plot(t, res['t0'], '.', color='blue', alpha=0.35, zorder=0, ms=1.5, label=r'$\chi^2=%1.2f$' %(chi2_t0_true0[1]))
                    ax2.plot(t, res['tref'], '.', color='purple', alpha=0.35, zorder=0, ms=2, label=r'$\chi^2=%1.2f$, n=%i' %(chi2_tref_true0[1], n))

                # determining the tref used for LOM: fit_tref
                tc_label = ''
                t0_label = ''
                tref_label = ''
                if chi2_t0_true0[1] <= chi2_tc_true0[1] and chi2_t0_true0[1] <= chi2_tref_true0[1]:
                    fit_tref = t0
                    t0_label = '*'
                elif chi2_tc_true0[1] <= chi2_t0_true0[1] and chi2_tc_true0[1] <= chi2_tref_true0[1]:
                    fit_tref = truths['tcroin']
                    tc_label = '*'
                elif chi2_tref_true0[1] <= chi2_t0_true0[1] and chi2_tref_true0[1] <= chi2_tc_true0[1]:
                    fit_tref = tc_calc
                    tref_label = '*'

            res[obs] = A_true[obs] - A[obs]

            if plot_initial_figures:
                ax1.plot(t, (f-fb[obs])/fs[obs], '.', color=colours[obs], label=data_labels[obs], alpha=0.5, zorder=0)
            
        np.savetxt(path+'posteriors/'+event_name+'chi2_true.txt', np.array([chi2_t0_true0, chi2_tc_true0, chi2_tref_true0]), fmt='%1.0f')
        
        if plot_initial_figures:
            A_lin = event_tc.get_magnification(tt, obs)
            ax1.plot(tt, A_lin, 
                     '-', 
                     label=r'$t_c=%1.1f$, $\chi^2/n=%1.2f$%s' %(event_tc.t_ref, chi2_tc_0[1]/n, tc_label),
                     zorder=6,
                     color='cyan', 
                     alpha=0.75, 
                     lw=1
                     )
            
            A_lin = event_t0.get_magnification(tt, obs)
            ax1.plot(tt, A_lin, 
                     '-', 
                     label=r'$t_0=%1.1f$, $\chi^2/n=%1.2f$%s' %(event_t0.t_ref, chi2_t0_0[1]/n, t0_label),
                     zorder=4, 
                     color='blue', 
                     alpha=0.75, 
                     lw=1
                     )
            A_lin = event_tref.get_magnification(tt, obs)
            ax1.plot(tt, A_lin, 
                     '-', 
                     label=r'$t_{c,calc}=%1.1f$, $\chi^2/n=%1.2f$%s' %(event_tref.t_ref, chi2_tref_0[1]/n, tref_label),
                     zorder=2, 
                     color='purple', 
                     alpha=0.75, 
                     lw=1
                     )
            
            # plot vertical lines at time t0 and tcroin (tref)
            ax1.axvline(x=t0, color='blue', linestyle='--', alpha=0.5, zorder=0, linewidth=1)
            ax1.axvline(x=event_tc.t_ref, color='cyan', linestyle='--', alpha=0.5, zorder=0, linewidth=1)
            ax1.axvline(x=event_tref.t_ref, color='purple', linestyle='--', alpha=0.5, zorder=0, linewidth=1)
            ax1.axvline(x=fit_tref, color='orange', linestyle='-', alpha=0.25, zorder=0, linewidth=4)

            ax2.axvline(x=t0, color='k', linestyle='--', alpha=0.5, zorder=0, linewidth=1)
            ax2.axvline(x=event_tc.t_ref, color='k', linestyle='--', alpha=0.5, zorder=0, linewidth=1)
            ax2.axvline(x=event_tref.t_ref, color='k', linestyle='--', alpha=0.5, zorder=0, linewidth=1)
            ax2.axvline(x=fit_tref, color='orange', linestyle='-', alpha=0.25, zorder=0, linewidth=4)

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

        #solutions_t0 = vbbl.PlotCrit(s, q) # Returns _sols object containing n crit. curves followed by n caustic curves
        #solutions_tc = vbbl.PlotCrit(s_tc, q)
        #solutions_tref = vbbl.PlotCrit(s_tref, q)

        #def iterate_from(item):
        #    while item is not None:
        #        yield item
        #        item = item.next

        #curves_t0 = []
        #curves_tc = []
        #curves_tref = []
        
        #for curve in iterate_from(solutions_t0.first):
        #    for point in iterate_from(curve.first):
        #        curves_t0.append((point.x1, point.x2))
        #for curve in iterate_from(solutions_tc.first):
        #    for point in iterate_from(curve.first):
        #        curves_tc.append((point.x1, point.x2))
        #for curve in iterate_from(solutions_tref.first):
        #    for point in iterate_from(curve.first):
        #        curves_tref.append((point.x1, point.x2))
                
        #critical_curves_t0 = np.array(curves_t0[:int(len(curves_t0)/2)])
        #caustic_curves_t0 = np.array(curves_t0[int(len(curves_t0)/2):])
        #critical_curves_tc = np.array(curves_tc[:int(len(curves_tc)/2)])
        #caustic_curves_tc = np.array(curves_tc[int(len(curves_tc)/2):])
        #critical_curves_tref = np.array(curves_tref[:int(len(curves_tref)/2)])
        #caustic_curves_tref = np.array(curves_tref[int(len(curves_tref)/2):])

        caustics_t0 = vbm.Caustics(s, q) 
        caustics_tc = vbm.Caustics(s_tc, q)
        caustics_tref = vbm.Caustics(s_tref, q)

        criticalcurves_t0 = vbm.Criticalcurves(s, q)
        criticalcurves_tc = vbm.Criticalcurves(s_tc, q)
        criticalcurves_tref = vbm.Criticalcurves(s_tref, q)

        
        if plot_initial_figures:
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
        tmax = np.max([t0+5.0*tE, tc+2.0*tE, tc_calc+2.0*tE])
        tmin = np.min([t0-5.0*tE, tc-2.0*tE, tc_calc-2.0*tE])
        points = np.where(np.logical_and(t > tmin, t < tmax))


        if plot_initial_figures:
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
                    '-', color='blue', alpha=0.5
                    )
            plt.plot(event_tc.traj_parallax_dalpha_u1[0][points],
                    event_tc.traj_parallax_dalpha_u2[0][points],
                    '-', color='cyan', alpha=0.5
                    )
            plt.plot(event_tref.traj_parallax_dalpha_u1[0][points],
                    event_tref.traj_parallax_dalpha_u2[0][points],
                    '-', color='purple', alpha=0.5
                    )
            
            
            #print('\nskipping contours\n')
            '''
            plt.contour(u1, u2, np.log10(Amap), 
                        levels=50, 
                        linewidths=0.5, 
                        colors='black', 
                        zorder=0
                        )#'''
            

            print('\nplotting caustics\n')
            #plt.plot(caustic_curves_t0[:,0], 
            #        caustic_curves_t0[:,1], 
            #        '.', color='blue', ms=0.2, 
            #        zorder=1
            #        )

            #plt.plot(caustic_curves_tc[:,0], 
            #        caustic_curves_tc[:,1], 
            #        '.', color='cyan', ms=0.2, 
            #        zorder=0, alpha = 0.5
            #        )
            
            #plt.plot(caustic_curves_tref[:,0], 
            #        caustic_curves_tref[:,1], 
            #        '.', color='purple', ms=0.2, 
            #        zorder=0, alpha = 0.5
            #        )

            for closed in caustics_t0:
                plt.plot(closed[0], closed[1],
                         '-', color='blue', ms=0.2, 
                         zorder=1
                         )
            for closed in caustics_tc:
                plt.plot(closed[0], closed[1],
                         '-', color='cyan', ms=0.2, 
                         zorder=0, alpha = 0.5
                         )
            for closed in caustics_tref:
                plt.plot(closed[0], closed[1],
                         '-', color='purple', ms=0.2, 
                         zorder=0, alpha = 0.5
                         )
                
            print('\nplotting criticals\n')
            #plt.plot(critical_curves_t0[:,0], 
            #        critical_curves_t0[:,1], 
            #        '.', color='grey', ms=0.2, alpha=0.5, 
            #        zorder=1
            #        )

            for closed in criticalcurves_t0:
                plt.plot(closed[0], closed[1],
                         '-', color='grey', ms=0.2, alpha=0.5, 
                         zorder=0
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
        
        xx = [1.0, -1.0, x0, x0_c, x0_ref]
        yy = [1.0, -1.0, y0, y0_c, y0_ref]

        if plot_initial_figures:
            print('\naesthetics\n')
            plt.grid()
            plt.axis('equal')
            plt.xlim(np.min(xx)-0.1, np.max(xx)+0.1)
            plt.ylim(np.min(yy)-0.1, np.max(yy)+0.1)
            plt.legend()

            plt.savefig(path+'posteriors/'+event_name+'_truths_caustic.png')
            plt.close()
        #------------------------------------------------


        # delta s
        #------------------------------------------------
        if plot_initial_figures:
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


        # delta alpha
        #------------------------------------------------
        if plot_initial_figures:
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
        if plot_initial_figures:
            print('Time to make initial figures = ', end_initial_figures - end_preabmle)

        #================================================


        # Prior Volume
        # I think these ranges need to stay the same for the logz values to be comparable
        # check how big these uncertainties normally are and adjust the ranges accordingly
        p_unc = np.array([0.2, 0.2, 0.5, 0.5, 0.1, 1.0, 5.0, 5.0, 10.0, np.pi/2.0, np.pi/2.0, 0.4])
        print('\n u prior uncertainty bounds = ', p_unc)


        # In your main script, where you define p_unc


        # NEW: Define p_unc for LOG parameters (s, q, rho, period)
        # This is now the desired uncertainty in log10 space.
        # Example: a log-uncertainty of 0.3 means a factor of 10**0.3 ~= 2
        p_unc_log_space = np.array([
            0.3, # s: e.g., factor of 2
            0.3, # q: e.g., factor of 2
            0.3, # rho: e.g., factor of 2
            0.1  # period: e.g., factor of 1.25
        ])
        prange_log = p_unc_log_space * 2.0

        # You'll also need to separate the linear ones
        linear_indices = [3, 4, 5, 6, 7, 8, 9, 10]
        p_unc_linear_space = p_unc[linear_indices]
        prange_linear = p_unc_linear_space * 2.0

        # FINALLY, when you call your sampler, you need to pass these new ranges.
        # The call to run_emcee and lnprob_transform will need to be updated to take
        # prange_linear and prange_log instead of the single 'bounds'.

        # ---- START: Sanity Check Snippet for prior_transform (v2 - with zoom) ----
        print('\\n\\n----------------------------------------------------')
        print('--- Running prior_transform Sanity Check ---')
        print('----------------------------------------------------')

        # The center of the unit hypercube
        u_center = np.ones(ndim) * 0.5

        # Use your (fixed) prior_transform to get physical parameters
        recovered_params = fit_obj.prior_transform(u_center, truths['params'], prange_linear, prange_log, normal=True)


        print('Original truths:\\n', truths['params'])
        print('Recovered from transform:\\n', recovered_params)
        print('Percent difference (%):\\n', (recovered_params - truths['params']) / truths['params'] * 100)


        # Generate a model light curve from these recovered parameters
        event_check = Event(parallax_obj, orbit_obj, data,
                            truths, data_obj.sim_time0, fit_tref)
        event_check.set_params(recovered_params)

        # Setup plot
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle('Sanity Check: Is prior_transform working? (Zoomed In)', fontsize=16)

        # Time range for smooth model
        t0_true = truths['params'][5]
        tE_true = truths['params'][6]
        tt = np.linspace(t0_true - 2.5 * tE_true, t0_true + 2.5 * tE_true, 2000)

        # Plot data points for context (using the first observatory)
        # NOTE: This uses fs/fb calculated from your *original* truths plot. Make sure that code has run.
        t_data_full = data[0][0,:]
        magnification_data = (data[0][1,:] - fb[0])/fs[0]
        ax1.plot(t_data_full, magnification_data, 'k.', alpha=0.2, label='Data (W146)')

        # Plot the TRUE model from your earlier plots
        true_model_A = event_tc.get_magnification(tt, obs=0)
        ax1.plot(tt, true_model_A, 'g-', lw=4, alpha=0.7, label='Original "Truths" Model')

        # Plot the model from the TRANSFORMED parameters
        check_model_A = event_check.get_magnification(tt, obs=0)
        ax1.plot(tt, check_model_A, 'r--', lw=2, label='Model from Transformed Center (u=0.5)')

        # Plot the residuals between the two models
        residuals = check_model_A - true_model_A
        ax2.plot(tt, residuals, 'r-')
        ax2.axhline(0, color='black', linestyle='--')

        ax1.set_ylabel('Magnification')
        ax2.set_ylabel('Residual (Transform - True)')
        ax2.set_xlabel('BJD')
        ax1.legend()
        ax1.set_title('If red dashed line is not perfectly on green, the transform is broken.')
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        # --- NEW LINES TO FIX THE ZOOM ---
        zoom_width_tE = 1.5  # Adjust this value as needed. 1.0 = +/- 1 tE.
        ax1.set_xlim(t0_true - zoom_width_tE * tE_true, t0_true + zoom_width_tE * tE_true)
        # --- END NEW LINES ---

        plt.savefig(path + 'posteriors/' + event_name + '_transform_sanity_check_zoomed.png', dpi=300)
        print('--- Sanity Check Plot Saved ---')
        print('----------------------------------------------------\\n\\n')
        # ---- END: Sanity Check Snippet ----


        '''
        print('\nTesting the fit functions')
        print('--------------------------------')

        fit.debug = ['ln_like', 'ln_prior', 'pt']

        print(type(event_tc), type(event_tc.truths['params']))

        fit.get_chi2(event_tc, event_tc.truths['params'])
        fit.lnlike(event_tc.truths['params'], event_tc)
        u = np.random.rand(12)
        print('u = ', u)

        prior_transform(fit, u, event_tc.truths['params'], prange=p_unc*2.0, normal=True)
        prior_transform(fit, u, event_tc.truths['params'], prange=p_unc*2.0,)

        sys.exit() #'''


        print()
        print('Sampling Posterior using emcee')
        print('--------------------------------')

        # Set up the sampler
        normal = True
        # Define the number of walkers and the initial positions of the walkers
        nl = 200  # number of live walkers
        mi = 2000  # max iterations
        stepi = 100  # steps between saving the sampler 
        u0 = np.ones((nl, ndim)) * 0.5
        initial_pos = u0 + 1e-12 * np.random.rand(nl, ndim)
        labels = ['s', 'q', 'rho', 'u0', 'alpha', 't0', 'tE', 'piEE', 'piEN', 'i', 'phase', 'period']
        bounds = p_unc*2.0
        normal = True

        #print('Debug main: pos = ', initial_pos, initial_pos.shape)
        initial_state = fit_obj.prior_transform(u_center, truths['params'], prange_linear, prange_log, normal=True)
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
        event_fit = Event(parallax_obj, orbit_obj, data_cropped, 
                         truths, data_obj.sim_time0, fit_tref
                         )


        # Run the sampler
        #pos = initial_state.copy()
        pos = initial_pos.copy()
        #fit_obj.debug = ['lnprob']
        #sampler = fit_obj.run_emcee(nl, ndim, stepi, mi, fit_obj.lnprob, pos, event_fit, threads=threads, event_name=event_name, path=path, labels=labels)
        sampler = fit_obj.run_emcee(nl, ndim, stepi, mi,
                                    fit_obj.lnprob_transform,
                                    pos,  # see next step for how to set this up
                                    event_fit, truths['params'],
                                    prange_linear,
                                    prange_log,
                                    normal,
                                    threads=threads, 
                                    event_name=event_name, 
                                    path=path, 
                                    labels=labels,
                                    )
        
        #print('Debug main: pos = ', pos, pos.shape)
        #print('Debug main: state = ', initial_state, initial_state.shape)
        #fit.debug = ['lnprob']


        # Timing the sampler
        end_dynesty = time.time()
        print('Time to run emcee (nl, mi)= ', end_dynesty - end_initial_figures, nl, mi)


        # Save the results
        #------------------------------------------------
        if sampling_package=='dynesty':
            res = sampler.results
            samples = res.samples
        if sampling_package=='emcee':
            res = sampler
            samples = res.chain.copy()
            flat_chain = res.chain.reshape(-1, samples.shape[-1])
        with open(path+'posteriors/'+event_name+'end_samples.pkl', 'wb') as f:
            pickle.dump(samples, f)

        print('samples.ndim:', samples.ndim)
        print('samples.shape:', samples.shape)

        samples = prior_transform(fit_obj, flat_chain, truths['params'],prange_linear, prange_log, normal=True)

        np.save(path+'posteriors/'+event_name+'_post_samples.npy', flat_chain)

        with open(path+'posteriors/'+event_name+'end_truths.pkl', 'wb') as f:
            pickle.dump(event_tc.truths, f)

        # print for logs
        print('Event', i, '(', event_name, ') is done')
        #------------------------------------------------

        # Final plots
        #================================================
        if plot_post:
            fit_obj.corner_post(samples, event_name, path, truths)
        if plot_run:
            fit_obj.runplot(res, event_name, path)
        if plot_trace:
            fit_obj.traceplot(res, event_name, path, truths)


        # In gulls_post_emcee_bound_wo_pt.py

        # lightcurve with samples
        #------------------------------------------------
        if not plot_final_figures:
            print('\\nskipping plotting lightcurve with samples\\n')
            # plot_final_figures = True  # Commented out for safety
        if plot_final_figures:
            print('\\nplotting lightcurve with samples\\n')
            plt.figure()
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2]})
            colours = ['orange', 'red', 'green']
            data_labels = ['W146', 'Z087', 'K213']
            ns = 20  # number of samples to plot

            # --- NEW LOGIC: Establish a CONSISTENT reference frame first ---
            # Use the initial "truth" parameters (p0) to define the reference magnification model
            p0 = prior_transform(fit_obj, np.array([0.5]*ndim), truths['params'], prange_linear, prange_log, normal=True)
            event_fit.set_params(p0)
            
            # Calculate fs and fb for EACH observatory based on this true model
            # and store them.
            fs_ref, fb_ref = {}, {}
            for obs in event_fit.data.keys():
                t_obs = event_fit.data[obs][0]
                f_obs = event_fit.data[obs][1]
                ferr_obs = event_fit.data[obs][2]
                A_ref = event_fit.get_magnification(t_obs, obs)
                fs_ref[obs], fb_ref[obs] = fit_obj.get_fluxes(A_ref, f_obs, ferr_obs**2)
            
            # --- NOW, PLOT EVERYTHING IN THIS CONSISTENT FRAME ---

            # Plot the initial "truth" model (red dashed line)
            tt = np.linspace(np.min(event_fit.data[0][0]), np.max(event_fit.data[0][0]), 2000)
            A0_fit = event_fit.get_magnification(tt, obs=0)
            ax1.plot(tt, A0_fit, '--', color='r', alpha=0.7, zorder=5, label='Initial Guess')
            
            # Get posterior samples for plotting
            flat_chain = res.get_chain(flat=True)
            samples = prior_transform(fit_obj, flat_chain, truths['params'], prange_linear, prange_log, normal=True)
            
            # Plot random posterior samples (blue lines)
            chain_index = np.random.randint(0, len(samples), ns)
            for i in range(ns):
                p = samples[chain_index[i], :]
                event_fit.set_params(p)
                A_p = event_fit.get_magnification(tt, obs=0)
                ax1.plot(tt, A_p, '-', color='blue', alpha=0.1, zorder=10)
            
            # Plot residuals of samples relative to the true model
            event_fit.set_params(p0) # reset to truth for residual calculation
            A_truth_on_data_grid = event_fit.get_magnification(event_fit.data[0][0], obs=0)
            for i in range(ns):
                p = samples[chain_index[i], :]
                event_fit.set_params(p)
                A_p_on_data_grid = event_fit.get_magnification(event_fit.data[0][0], obs=0)
                ax2.plot(event_fit.data[0][0], A_p_on_data_grid - A_truth_on_data_grid, '-', color='blue', alpha=0.05, zorder=10)

            # Loop through observatories to plot data points
            for obs in event_fit.data.keys():
                t = event_fit.data[obs][0]
                f = event_fit.data[obs][1]
                
                # Convert data flux to magnification using the STORED reference fs and fb
                A_data = (f - fb_ref[obs]) / fs_ref[obs]
                ax1.plot(t, A_data, '.', color=colours[obs], label=data_labels[obs], alpha=0.5, zorder=1)

                # Plot residuals of data relative to the true model
                if obs == 0:
                    A_truth_on_data_grid = event_fit.get_magnification(t, obs)
                    ax2.plot(t, A_data - A_truth_on_data_grid, '.', color='k', alpha=0.2, zorder=1)


            ax1.set_ylabel('Magnification')
            ax2.set_ylabel('Residuals')
            ax2.set_xlabel('BJD')
            ax1.set_title('s=%.2f, q=%.6f, rho=%.6f, u0=%.2f, alpha=%.2f, t0=%.2f, \ntE=%.2f, piEE=%.2f, piEN=%.2f, i=%.2f, phase=%.2f, period=%.2f' %tuple(event_fit.true_params)) 
            ax1.legend()

            plt.savefig(path+'posteriors/'+event_name+'_lightcurve_samples.png', dpi=300)
            plt.close()
        #------------------------------------------------


        # Timing the final plots
        end_final_figures = time.time()
        print('Time to make final figures = ', end_final_figures - end_dynesty)

        #================================================


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


        # Timing the wrap up
        end_time = time.time()


        # Timing Summary
        print('\n\nTime Summary')
        print('--------------------------------')
        print('Start time = ', start_time)
        print('Time to get data = ', end_preabmle - start_time)
        print('Time to make initial figures = ', end_initial_figures - end_preabmle)
        print('Time to run %s (nl, mi)= ' %(sampling_package), end_dynesty - end_initial_figures, nl, mi)
        print('Time to make final figures = ', end_final_figures - end_dynesty)
        print('Time to wrap up = ', end_time - end_dynesty)
        print('Total time = ', end_time - start_time)
        print('End time = ', end_time)
        print('--------------------------------\n\n\n')


# ephermeris
# 2018-08-10
# 2023-04-30
# get emphemeris file from JPL Horizons for SEMB-L2

#
#debug Fit.lnprob: s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period: 
# s = 7.07360619e-01 
# q = 6.18405751e-04 
# rho = 5.50690071e-04 
# u0 = 3.72512756e-01
# alpha = 6.53109803e-01 
# t0 = 2.45841982e+06 
# tE = 2.26521744e+01 
# piEE = 2.88142171e-02
# piEN = 8.73733186e-02 
# i = 6.58998872e-01 
# phase = 4.86893527e+00 
# period = 1.60797061e+03

