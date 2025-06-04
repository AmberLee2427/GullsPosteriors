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

mp.set_start_method("fork", force=True)


def run_emcee(
    self,
    nl,
    ndim,
    stepi,
    mi,
    log_prob_function,
    state,
    event_obj,
    threads=1,
    event_name="",
    path="./",
    labels=None,
):
    if threads > 1:
        with Pool(threads) as pool:
            # Initialize the sampler
            sampler = emcee.EnsembleSampler(
                nl, ndim, log_prob_function, args=[event_obj], pool=pool
            )

            # Run the sampler
            count = 0
            steps = 0
            while steps < mi:
                print("Debug main: loop", count)
                state, lnp, _ = sampler.run_mcmc(state, stepi, progress=False)
                chain = sampler.chain
                flatchain = sampler.flatchain
                lnprobability = sampler.lnprobability
                flatlnprobability = sampler.flatlnprobability

                # Save the sampler as a pickle file
                # with open(path+'posteriors/'+event_name+'_emcee_sampler.pkl', 'wb') as f:
                #    pickle.dump(sampler, f)
                # can't pickle the sampler object because of the VBM parent object

                # Save the samples
                np.save(
                    path + "posteriors/" + event_name + "_emcee_samples.npy",
                    flatchain,
                )
                np.save(
                    path + "posteriors/" + event_name + "_emcee_lnprob.npy",
                    flatlnprobability,
                )
                np.save(
                    path + "posteriors/" + event_name + "_emcee_state.npy",
                    state,
                )

                self.plot_chain(sampler, event_name, path, labels=labels)

                steps += stepi
                count += 1

    else:
        # Initialize the sampler
        sampler = emcee.EnsembleSampler(
            nl, ndim, log_prob_function, args=[event_obj]
        )

        # Run the sampler
        steps = 0
        count = 0
        while steps < mi:
            print("Debug main: loop", count)
            state, lnp, _ = sampler.run_mcmc(state, stepi, progress=False)
            chain = sampler.chain
            flatchain = sampler.flatchain
            lnprobability = sampler.lnprobability
            flatlnprobability = sampler.flatlnprobability

            # Save the sampler as a pickle file
            # with open(path+'posteriors/'+event_name+'_emcee_sampler.pkl', 'wb') as f:
            #    pickle.dump(sampler, f)

            # Save the samples
            np.save(
                path + "posteriors/" + event_name + "_emcee_samples.npy",
                flatchain,
            )
            np.save(
                path + "posteriors/" + event_name + "_emcee_lnprob.npy",
                flatlnprobability,
            )
            np.save(
                path + "posteriors/" + event_name + "_emcee_state.npy", state
            )

            self.plot_chain(sampler, event_name, path, labels=labels)

            steps += stepi
            count += 1

    return sampler


def lnprob_transform(self, u, event, true, prange, normal=False):
    """Transform the unit cube to the parameter space and calculate the log probability."""

    for uu in u:
        if uu < 0.0 or uu > 1.0:
            return -np.inf
    theta = self.prior_transform(u, true, prange, normal)
    lp = self.lnprob(theta, event)

    return lp


def plot_chain(self, res, event_name, path, labels=None, pt_args=[]):
    """e.g. fit.plot_chain(sampler, event_name, path, labels=labels, pt_arg=[truths['params'], bounds, normal])"""

    chain = res.chain  # shape = (nwalkers, nsteps, ndim)
    # print('Debug Fit.plot_chain: chain shape: ', chain.shape)
    lnprobability = res.lnprobability  # shape = (nwalkers, nsteps)
    # print('Debug Fit.plot_chain: lnprobability shape: ', lnprobability.shape)
    ndim = chain.shape[2]
    nsteps = chain.shape[1]
    nwalkers = chain.shape[0]

    if labels is None:
        labels = [f"theta[{i}]" for i in range(ndim)]

    fig, axes = plt.subplots(ndim + 1, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        for j in range(nwalkers):
            ax.plot(chain[j, :, i], "k", alpha=0.1)
        ax.set_xlim(0, nsteps)
        ax.set_ylabel(labels[i])
        # ax.yaxis.set_label_coords(-0.1, 0.5)

    ax = axes[-1]
    for j in range(nwalkers):
        ax.plot(lnprobability[j], "r", alpha=0.3)
    ax.plot(lnprobability, "r", alpha=0.3)
    ax.set_xlim(0, nsteps)
    ax.set_ylabel("lnprob")

    axes[-1].set_xlabel("step number")
    fig.suptitle(event_name)
    fig.savefig(path + "posteriors/" + event_name + "_chain.png")
    plt.close(fig)


def corner_post(self, res, event_name, path, truths):
    labels = [
        "s",
        "q",
        "rho",
        "u0",
        "alpha",
        "t0",
        "tE",
        "piEE",
        "piEN",
        "i",
        "phase",
        "period",
    ]
    fig = corner.corner(res.samples, labels=labels, truths=truths["params"])
    plt.title(event_name)

    fig.savefig(path + "posteriors/" + event_name + "_corner.png")
    plt.close(fig)

    if "corner" in self.debug:
        print("debug Fit.corner_post: labels: ", labels)
        print("debug Fit.corner_post: truths: ", truths["params"])
        print("debug Fit.corner_post: event_name: ", event_name)
        print("debug Fit.corner_post: path: ", path)
