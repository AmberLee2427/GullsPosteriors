# import VBBinaryLensing as vbl
import os
import emcee
import dynesty
import numpy as np
import MulensModel as mm
import multiprocessing as mp
from astropy import units as u

# debug event: 331
# N = 6757
# chi2 = 6762 - binary
# chi2 = 6857 - single
# delta chi2 = 95

nevents = int(sys.argv[1])
if len(sys.argv) == 2:
    sort = sys.argv[2]
else:
    sort = "alphanumeric"
nwalkers = 50
nsteps = 200
ndim = 11

# limb darkening coefficients !!!
# 0.36 in all bands - linear limb darkening

# probably dynesty would make more sense for posterior sampling because it will have z values.


parameters_to_fit = [
    "t_0",
    "u_0",
    "t_E",
    "rho",
    "q",
    "s",
    "rho",
    "alpha",
    "pi_E_N",
    "pi_E_E",
    "ds_dt",
    "dalpha_dt",
    "dzdt",
]
# check with Mathew about the model used to evaluate the BL chi2  !!!
# check with mathew about the parallax stuff. !!!


def lnlike(theta, event):
    for parameter, value in zip(parameters_to_fit, theta):
        setattr(event.model.parameters, parameter, value)

    # After that, calculating chi2 is trivial:
    chi2 = event.get_chi2()
    return -0.5 * chi2


def lnprior(params, event):
    logs, logq, logrho, u0, alpha, t0, tE, piE, piN, dsdt, dalphadt, dzdt = (
        params
    )
    # s = 10.0**logs
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
    return lp + lnlike(params)


def run_emcee(p0, data):
    logs, logq, logrho, u0, alpha, t0, tE, piE, piN, dsdt, dalphadt, dzdt = p0
    tE = tE * u.day
    s = 10.0**logs
    q = 10.0**logq
    rho = 10.0**logrho
    alpha = alpha * u.deg
    dsdt = dsdt / u.year
    dalphadt = dalphadt * u.rad / u.day
    t_0_par = t0

    event = 0  # makesure we are removing the old event
    event = mm.Event(datasets=data, model=model)

    model = mm.Model(
        {
            "t_0": t0,
            "u_0": u0,
            "t_E": tE,
            "rho": rho,
            "q": q,
            "s": s,
            "alpha": alpha,
            "pi_E_N": piN,
            "pi_E_E": piE,
            "t_0_par": t_0_par,
            "ds_dt": dsdt,
            "dalpha_dt": dalphadt,
        }
    )

    model.set_magnification_methods(["VBBL"])
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, threads=mp.cpu_count()
    )

    sampler.run_mcmc(p0, nsteps, args=(event))

    ## SAVE !!!


def new_event(sort):
    """get the data and true params for the next event"""

    if sort == "alphanumeric":
        files = os.listdir("data")
        files = sorted(files)
        if os.path.exists("runlist.npy"):
            runlist = np.loadtxt("runlist.npy")
        else:
            runlist = np.array([])
        for i in range(len(files)):
            if files[i] not in runlist:
                runlist = np.vstack(files[i])
                np.savetxt("runlist.txt", runlist, fmt="%s")
                data = mm.MulensData(file_name="data/" + files[i])
                true_params = np.loadtxt(
                    "true_params/" + files[i].split(".")[0] + ".txt"
                )
                break

    if ".txt" in sort:
        files = np.loadtxt(sort)
        for i in range(len(files)):
            if os.path.exists("runlist.npy"):
                runlist = np.loadtxt("runlist.npy")
            else:
                runlist = np.array([])
            if files[i] not in runlist:
                runlist = np.vstack(files[i])
                np.savetxt("runlist.txt", runlist, fmt="%s")
                data = mm.MulensData(file_name="data/" + files[i])
                true_params = np.loadtxt(
                    "true_params/" + files[i].split(".")[0] + ".txt"
                )
                break

    os.mkdir("posteriors/" + files[i].split(".")[0])

    return data, true_params


for i in range(nevents):
    true_params, data = new_event(sort)
    p0 = true_params
    run_emcee(p0, data)
    print("Event", i, "done")
