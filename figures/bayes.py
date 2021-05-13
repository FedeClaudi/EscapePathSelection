import sys
sys.path.append('./')   

import numpy as np
import pandas as pd

import pymc3 as pm
from scipy.special import binom

from fcutils.maths.distributions import get_parametric_distribution, beta_distribution_params
from fcutils.maths.array import percentile_range


class Bayes:
    # Bayes hyper params
    hyper_mode = (1, 1)  # a,b of hyper beta distribution (modes)
    concentration_hyper = (1, 10)  # mean and std of hyper gamma distribution (concentrations)
    k_hyper_shape, k_hyper_rate = 0.01, 0.01

    a, b  = 1.00, 1.00 # Parameters of priors Beta for individual and grouped analytical solution

    def __init__(self):
        pass


    def grouped_bayes_analytical(self, n, k):
        """[Solves the bayesia model for p(R) for grouped data]
        
        Arguments:
            n {[int]} -- [tot number of trials]
            k {[int]} -- [tot number of hits]
        """
        # Compute posterior function
        fact = binom(n, k)
        a2 = self.a + k - 1
        b2 = self.b + n -k -1

        # Plot mean and mode of posterior
        mean =  a2 / (a2 + b2)
        mode = (a2 -1)/(a2 + b2 -2)
        sigmasquared = (a2 * b2)/((a2+b2)**2 * (a2 + b2 + 1)) # https://math.stackexchange.com/questions/497577/mean-and-variance-of-beta-distributions
        beta = get_parametric_distribution("beta", a2, b2)[1]
        prange = percentile_range(beta)

        return (a2, b2, mean, mode, sigmasquared, prange, beta)


    def individuals_hierarchical_bayes(self, N, K, N_trials, **kwargs):
        """
        :param N: number of individuals
        :param K: number of hits per individual
        :param N_trials: number of trials for each individual
        """

        trace_length = kwargs.pop("trace_length", 5000)
        tune_length = kwargs.pop("tune_length", 500)
        n_cores = kwargs.pop("n_cores", 2)

        # bayes param
        print(f"Fitting hierarchical model for {N} mice")
        with pm.Model() as model:
            # Define hyperparams
            modes_hyper = pm.Beta("mode_hyper", 
                                        alpha=self.hyper_mode[0], beta=self.hyper_mode[1])
            concentrations_hyper = pm.Gamma("concentration_hyper", 
                                        alpha=self.k_hyper_shape, beta=self.k_hyper_rate)

            # Define priors
            prior_a, prior_b = beta_distribution_params(omega=modes_hyper, kappa=concentrations_hyper)
            prior = pm.Beta("beta_prior", alpha=prior_a, beta=prior_b, shape=N)

            # Define likelihood
            likelihood = pm.Binomial("likelihood", n=N_trials, p=prior, observed=K)

            # Fit
            trace = pm.sample(trace_length, tune=tune_length, cores=n_cores, 
                                nuts_kwargs={'target_accept': 0.99}, progressbar=True)
        
        # Extract stats from traces
        trace = pm.trace_to_dataframe(trace)

        posteriors = [trace[c].values for c in trace.columns if 'beta_prior' in c]
        means = [np.mean(p) for p in posteriors]
        stds = [np.std(p) for p in posteriors]

        return trace, means, stds


