# In Fit/__init__.py

import os
import sys
import emcee
import numpy as np
import pandas as pd
import pickle
import corner
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import time
from multiprocessing import Pool
import multiprocessing as mp


class Fit:
    ''' Fit class doc string '''

    from ._emcee import run_emcee, lnprob_transform, plot_chain, corner_post
    # prior_transform will now be fully defined in _dynesty.py
    # runplot and traceplot are also in _dynesty.py
    from ._dynesty import prior_transform, runplot, traceplot 

    # MODIFIED __init__ to accept and store ndim and labels
    def __init__(self, sampling_package='emcee', debug=None, LOM_enabled=True, ndim=None, labels=None):
        if debug is not None:
            self.debug = debug
        else:
            self.debug = []
    
        self.sampling_package = sampling_package
        if sampling_package != 'dynesty' and sampling_package != 'emcee':
            print('Invalid sampling package. Must be dynesty or emcee')
            sys.exit()
        
        self.LOM_enabled = LOM_enabled
        self.ndim = ndim     # NEW: Store ndim
        self.labels = labels # NEW: Store labels

    def get_fluxes(self, model:np.ndarray, f:np.ndarray, sig2:np.ndarray):
        # ... (this function is fine) ...
        if model.shape[0] != f.shape[0]:
            print('debug Fit.get_fluxes: model and f have different lengths')
            sys.exit()
        if model.shape[0] != sig2.shape[0]:
            print('debug Fit.get_fluxes: model and sig2 have different lengths')
            sys.exit()
        if f.shape[0] != sig2.shape[0]:
            print('debug Fit.get_fluxes: f and sig2 have different lengths')
            sys.exit()

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
        # ... (this function is fine) ...
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

            chi2[obs] = ((f - (A*fs + fb)) / f_err) ** 2

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

        return chi2, chi2sum

    def lnlike(self, theta, event):  
        # ... (this function is fine) ...
        _, chi2 = self.get_chi2(event, theta)

        if 'lnlike' in self.debug:
            print('debug Fit.lnlike: chi2: ', chi2)
            print('debug Fit.lnlike: theta: ', theta)

        return -0.5 * chi2

    # MODIFIED: lnprior to use self.LOM_enabled
    def lnprior(self, theta, bound_penalty=False):

        if self.LOM_enabled:
            s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period = theta
            if 'ln_prior' in self.debug:
                print('debug Fit.lnprior (LOM):', theta)
            if tE > 0.0 and q <= 1.0 and period/tE > 4 and s > 0.001 and rho > 0.0:
                return 0.0 # Add bound_penalty logic if you ever use it
            else: # Debug prints for failing conditions
                # ... (your original debug prints for tE, q, period/tE, s, rho)
                return -np.inf
        else: # No LOM
            s, q, rho, u0, alpha, t0, tE, piEE, piEN = theta
            if 'ln_prior' in self.debug:
                print('debug Fit.lnprior (No LOM):', theta)
            if tE > 0.0 and q <= 1.0 and s > 0.001 and rho > 0.0:
                 return 0.0
            else: # Debug prints for failing conditions
                # ... (your original debug prints for tE, q, s, rho)
                return -np.inf

    # MODIFIED: lnprob to use self.LOM_enabled
    def lnprob(self, theta, event):
        if self.LOM_enabled:
            theta[4] %= (2 * np.pi)  # alpha
            theta[10] %= (2 * np.pi) # phase
            # For inclination 'i', it's usually 0 to pi. If it's 0 to 2pi in your setup, this is fine.
            # Otherwise, you might need theta[9] = np.abs(theta[9] % np.pi) or similar.
            theta[9] %= (2 * np.pi)  # i 

        else:
            theta[4] %= (2*np.pi)  # alpha

        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.lnlike(theta, event)
        if not np.isfinite(ll):
            return -np.inf
        
        if 'lnprob' in self.debug:
            print('debug Fit.lnprob: lp, ll: ', lp, ll)
            print('                  ', theta)
            
        return lp + ll