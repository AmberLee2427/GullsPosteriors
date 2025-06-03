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
mp.set_start_method('fork', force=True)


def run_emcee(self, nl, ndim, stepi, mi, log_prob_function, state, event_obj, truths, prange_linear, prange_log, normal, threads=1, event_name='', path='./', labels=None):
    """
    Edited run_emcee function.
    'bounds' has been replaced by 'prange_linear' and 'prange_log'.
    """
    # The new arguments to be passed to the log probability function
    log_prob_args = [event_obj, truths, prange_linear, prange_log, normal]

    if threads > 1:
        with Pool(threads) as pool:
            # Initialize the sampler with the new arguments
            sampler = emcee.EnsembleSampler(nl, ndim, log_prob_function, args=log_prob_args, pool=pool)

            # --- The rest of the function is identical ---
            # Run the sampler
            count = 0
            steps = 0
            while steps < mi:
                state, lnp, _ = sampler.run_mcmc(state, stepi, progress=True)
                flatchain = sampler.flatchain
                flatlnprobability = sampler.flatlnprobability

                # Save the samples
                np.save(path+'posteriors/'+event_name+'_emcee_samples.npy', flatchain)
                np.save(path+'posteriors/'+event_name+'_emcee_lnprob.npy', flatlnprobability)
                np.save(path+'posteriors/'+event_name+'_emcee_state.npy', state)

                self.plot_chain(sampler, event_name, path, labels=labels)

                steps += stepi
                count += 1
                
    else:
        # Initialize the sampler with the new arguments
        sampler = emcee.EnsembleSampler(nl, ndim, log_prob_function, args=log_prob_args)

        # --- The rest of the function is identical ---
        # Run the sampler
        steps = 0
        count = 0
        while steps < mi:
            state, lnp, _ = sampler.run_mcmc(state, stepi, progress=True)
            flatchain = sampler.flatchain
            flatlnprobability = sampler.flatlnprobability

            # Save the samples
            np.save(path+'posteriors/'+event_name+'_emcee_samples.npy', flatchain)
            np.save(path+'posteriors/'+event_name+'_emcee_lnprob.npy', flatlnprobability)
            np.save(path+'posteriors/'+event_name+'_emcee_state.npy', state)

            self.plot_chain(sampler, event_name, path, labels=labels)

            steps += stepi
            count += 1
    
    return sampler


def lnprob_transform(self, u, event, true, prange_linear, prange_log, normal=False):
    '''
    Edited lnprob_transform function.
    Transforms the unit cube to the parameter space and calculates the log probability.
    'prange' has been replaced by 'prange_linear' and 'prange_log'.
    '''
    # Check if any walker is somehow outside the unit cube.
    for uu in u:
        if not (0.0 <= uu <= 1.0):
            return -np.inf
            
    # Call the new, improved prior_transform with the correct arguments
    theta = self.prior_transform(u, true, prange_linear, prange_log, normal)
    
    # Calculate the log probability (likelihood) with the transformed parameters
    lp = self.lnprob(theta, event)
        
    return lp


def plot_chain(self, res, event_name, path, labels=None, pt_args=[]):
    '''e.g. fit.plot_chain(sampler, event_name, path, labels=labels, pt_arg=[truths['params'], bounds, normal])'''

    chain = res.chain  # shape = (nwalkers, nsteps, ndim)
    #print('Debug Fit.plot_chain: chain shape: ', chain.shape)
    lnprobability = res.lnprobability  # shape = (nwalkers, nsteps)
    #print('Debug Fit.plot_chain: lnprobability shape: ', lnprobability.shape)
    ndim = chain.shape[2]
    nsteps = chain.shape[1]
    nwalkers = chain.shape[0]

    if labels is None:
        labels = [f"theta[{i}]" for i in range(ndim)]

    fig, axes = plt.subplots(ndim+1, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        for j in range(nwalkers):
            ax.plot(chain[j, :, i], "k", alpha=0.1)
        ax.set_xlim(0, nsteps)
        ax.set_ylabel(labels[i])
        #ax.yaxis.set_label_coords(-0.1, 0.5)

    ax = axes[-1]
    for j in range(nwalkers):
        ax.plot(lnprobability[j], "r", alpha=0.3)
    ax.plot(lnprobability, "r", alpha=0.3)
    ax.set_xlim(0, nsteps)
    ax.set_ylabel("lnprob")

    axes[-1].set_xlabel("step number")
    fig.suptitle(event_name)
    fig.savefig(path+'posteriors/'+event_name+'_chain.png')
    plt.close(fig)


def corner_post(self, samples, event_name, path, truths):
    if self.LOM_enabled:
        labels = ['s', 'q', 'rho', 'u0', 'alpha', 't0', 'tE', 'piEE', 'piEN', 'i', 'phase', 'period']
        true_params = truths['params']
    else:
        labels = ['s', 'q', 'rho', 'u0', 'alpha', 't0', 'tE', 'piEE', 'piEN']
        true_params = truths['params'][:9] # Use only the first 9 truth values

    fig = corner.corner(samples, labels=labels, truths=true_params)
    plt.title(event_name)

    fig.savefig(path+'posteriors/'+event_name+'_corner.png')
    plt.close(fig)

    if 'corner' in self.debug:
        print('debug Fit.corner_post: labels: ', labels)
        print('debug Fit.corner_post: truths: ', truths['params'])
        print('debug Fit.corner_post: event_name: ', event_name)
        print('debug Fit.corner_post: path: ', path)
