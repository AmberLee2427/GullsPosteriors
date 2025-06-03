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
    #from ._dynesty import prior_transform, runplot, traceplot

    def __init__(self, sampling_package='emcee', debug=None) -> None:
        if debug is not None:
            self.debug = debug
        else:
            self.debug = []
    
        self.sampling_package = sampling_package
        if sampling_package != 'dynesty' and sampling_package != 'emcee':
            print('Invalid sampling package. Must be dynesty or emcee')
            sys.exit()

    def get_fluxes(self, model:np.ndarray, f:np.ndarray, sig2:np.ndarray):
        '''Solves for the flux parameters for a given model using least squares.
        
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

        if 'lnprior' in self.debug:
            print('debug Fit.lnprior: s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period: ')
            print('                   ', theta)

        if tE > 0.0 and q <= 1.0 and period/tE > 4 and s > 0.001 and rho>0.0:
        
            if bound_penalty:   # i'm not using this. I need to redo the calculation
                # calculate beta and require the orbits conserve energy
                #G_au = 1.0  # gravitational constant in AU^3 / (M1 * years^2)  / (2pi)^2
                #m1 = 1  # mass of the first object
                #m2 = m1*q  # mass of the second object
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
                if tE < 0.0:
                    print('debug Fit.lnprior: tE = ', tE, ' > 0.0')
                if q >= 1.0:
                    print('debug Fit.lnprior: q = ', q, ' >= 1.0')
                if period/tE < 4:
                    print('debug Fit.lnprior: period/tE = ', period/tE, ' < 4')
                if s < 0.001:
                    print('debug Fit.lnprior: s = ', s, ' < 0.01')
                if rho < 0.0:
                    print('debug Fit.lnprior: rho = ', rho, ' < 0.0')
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
