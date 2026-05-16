import glob, json
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
import h5py
import matplotlib.pyplot as plt
import corner
import pickle
import os

def load_ptmcmc_chains(dirpath, ntemps=5, burn_in=int(250), thin=5):
    """
    Load PT-MCMC chains from a given directory.

    Parameters
    ----------
    dirpath : str
        Path to the run directory (must contain prior_dict.json and .h5 files).
        injected_params.json is optional.
    ntemps : int
        Number of parallel-tempering temperatures (default 5).
    burn_in : int
        Number of iterations to discard as burn-in.
    thin : int
        Thinning factor.

    Returns
    -------
    chains_physical : np.ndarray, shape (niters, ntemps, nwalkers, nparams)
    loglike         : np.ndarray, shape (niters, ntemps, nwalkers)
    params_sampled  : list of str
    injected_dict   : dict or None
    cov_matrix      : np.ndarray, shape (nparams, nparams)
    k_samples       : np.ndarray or None, shape (niters, ntemps, nwalkers)
        Model index array if present in h5 files, else None.
    """

    dirpath = dirpath.rstrip('/')

    # ------------------------------------------------------------------ #
    # 1. Load metadata
    # ------------------------------------------------------------------ #
    injected_dict = None
    if os.path.exists(f'{dirpath}/injected_params.json'):
        with open(f'{dirpath}/injected_params.json', 'r') as f:
            injected_dict = json.load(f)

    with open(f'{dirpath}/prior_dict.json', 'r') as f:
        priors_all = json.load(f)

    # ------------------------------------------------------------------ #
    # 2. Build prior transform
    # ------------------------------------------------------------------ #
    params_sampled = [
        k for k, v in priors_all.items() if v.get('dist') != 'constant'
    ]
    n_params = len(params_sampled)

    normal_indices, uniform_indices = [], []
    normal_mus, normal_sigmas       = [], []
    uniform_lbs, uniform_ubs        = [], []

    for i, name in enumerate(params_sampled):
        prior = priors_all[name]
        if prior['dist'] == 'normal':
            normal_indices.append(i)
            normal_mus.append(prior['mu'])
            normal_sigmas.append(prior['sigma'])
        elif prior['dist'] == 'uniform':
            uniform_indices.append(i)
            uniform_lbs.append(prior['min'])
            uniform_ubs.append(prior['max'])

    normal_mus      = jnp.array(normal_mus)
    normal_sigmas   = jnp.array(normal_sigmas)
    uniform_lbs     = jnp.array(uniform_lbs)
    uniform_ubs     = jnp.array(uniform_ubs)
    eps             = 1e-10
    normal_indices  = np.array(normal_indices)
    uniform_indices = np.array(uniform_indices)

    @jax.jit
    def prior_transform_single(u):
        x = jnp.zeros(n_params)
        if len(normal_indices) > 0:
            x = x.at[normal_indices].set(
                normal_mus + jnp.maximum(normal_sigmas, eps) * ndtri(u[normal_indices])
            )
        if len(uniform_indices) > 0:
            x = x.at[uniform_indices].set(
                uniform_lbs + u[uniform_indices] * (uniform_ubs - uniform_lbs)
            )
        return x

    transform_chains = jax.vmap(jax.vmap(jax.vmap(prior_transform_single)))

    # ------------------------------------------------------------------ #
    # 3. Read .h5 chain files
    # ------------------------------------------------------------------ #
    chain_files = sorted(glob.glob(f'{dirpath}/*.h5'))
    print(f"Found {len(chain_files)} chain file(s): {[f.split('/')[-1] for f in chain_files]}")

    all_p, all_ll, all_k = [], [], []
    has_k = None  # will be determined from first file

    for fpath in chain_files:
        with h5py.File(fpath, 'r') as hf:
            for ekey in sorted(hf.keys()):
                p_raw  = hf[f'{ekey}/samples/p'][()]
                ll_raw = hf[f'{ekey}/samples/ll'][()]

                niters   = p_raw.shape[-1]
                nwalkers = int(p_raw.shape[0] / ntemps)

                all_p.append( p_raw.reshape(nwalkers, ntemps, n_params, niters))
                all_ll.append(ll_raw[:, 0, :].reshape(nwalkers, ntemps, niters))

                # k is optional — only present in model selection runs
                if has_k is None:
                    has_k = f'{ekey}/samples/k' in hf
                if has_k:
                    k_raw = hf[f'{ekey}/samples/k'][()]
                    all_k.append(k_raw[:,0,:].reshape(nwalkers, ntemps, niters))

    # ------------------------------------------------------------------ #
    # 4. Concatenate along iterations axis
    # ------------------------------------------------------------------ #
    samples_unit = np.concatenate(all_p,  axis=3)  # (nwalkers, ntemps, nparams, niters)
    loglike_raw  = np.concatenate(all_ll, axis=2)  # (nwalkers, ntemps, niters)

    nwalkers, ntemps_out, n_params_out, niters = samples_unit.shape
    print(f"nwalkers={nwalkers}, ntemps={ntemps_out}, nparams={n_params_out}, niters={niters}")

    samples_unit_r = np.transpose(samples_unit, (3, 1, 0, 2))[burn_in::thin]  # (niters, ntemps, nwalkers, nparams)
    loglike        = np.transpose(loglike_raw,  (2, 1, 0))[burn_in::thin]     # (niters, ntemps, nwalkers)

    k_samples = None
    if has_k:
        k_raw_cat  = np.concatenate(all_k, axis=2)                            # (nwalkers, ntemps, niters)
        k_samples  = np.transpose(k_raw_cat, (2, 1, 0))[burn_in::thin]        # (niters, ntemps, nwalkers)

    # ------------------------------------------------------------------ #
    # 5. Covariance from cold chain (unit cube space)
    # ------------------------------------------------------------------ #
    cold_chain_flat = samples_unit_r[:, 0, :, :].reshape(-1, n_params)
    cov_matrix      = np.cov(cold_chain_flat.T)

    # ------------------------------------------------------------------ #
    # 6. Transform to physical space
    # ------------------------------------------------------------------ #
    chains_physical = np.array(transform_chains(jnp.array(samples_unit_r)))
    print(f"chains_physical shape: {chains_physical.shape}")
    print(f"loglike shape:         {loglike.shape}")
    if has_k:
        print(f"k_samples shape:       {k_samples.shape}")

    return chains_physical, loglike, params_sampled, injected_dict, cov_matrix, k_samples
