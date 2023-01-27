import stan 
import numpy as np
from scipy.special import expit, logit
import pandas as pd
import pickle


n_rows = 20
n_cols = 50
n_features = 3
K = 3
n_entries = 1000
sigma = 0.25


mu = np.random.standard_normal(size=(K, n_features))
V = np.random.standard_normal(size=(n_cols, n_features))

clusters = np.random.choice(K, size=n_rows, replace=True)
W = np.empty((n_rows, n_features))
for i in range(n_rows):
    W[i,:] = mu[clusters[i],:] + sigma*np.random.standard_normal(size=n_features)

ii = np.random.choice(n_rows, size=n_entries, replace=True)
jj = np.random.choice(n_cols, size=n_entries, replace=True)

P = np.einsum('ik, jk -> ij', W, V)
y = P[ii,jj] + sigma*np.random.standard_normal(size=n_entries)



mixture_norm_mf = """
data {
    int<lower=1> n_rows;                              // Number of rows
    int<lower=1> n_cols;                              // Number of columns
    int<lower=1> n_features;                          // Number of features 
    int<lower=1> n_entries;                           // Number of entries in matrix 
    int<lower=1> K;
    real<lower=0> sigma;
    real<lower=0> sigma_emb;
    array[n_entries] int<lower=1,upper=n_rows> ii;    // Row indices
    array[n_entries] int<lower=1,upper=n_cols> jj;    // Col indices
    array[n_entries] real y;                          // observations (force int) 

}
parameters {
    matrix[n_rows, n_features] W;         
    matrix[n_cols, n_features] V;
    matrix[K, n_features] mu;
}
model {
    for (n in 1:n_entries) {
        y[n] ~ normal( dot_product(W[ii[n],:], V[jj[n],:]) , sigma);            
    }

    for (n in 1:n_cols) {
        V[n,:] ~ std_normal();
    }

    for (k in 1:K) {
        mu[k,:] ~ std_normal();
    }

    for (n in 1:n_rows) {
        vector[K] lps;
        for (k in 1:K) {
            lps[k] = normal_lpdf(W[n,:] | mu[k,:], sigma_emb);
        }
        target += log_sum_exp(lps);
    }
}
"""

dataset = {"n_rows":n_rows, "n_cols":n_cols, "n_features":n_features, "n_entries":n_entries, "K":3, "sigma":sigma, "sigma_emb":sigma, "ii":(ii+1), "jj":(jj+1), "y":y}

num_samples = 50
n_thin = 5
model = stan.build(mixture_norm_mf, data=dataset, random_seed=10)
fit = model.sample(num_chains=1, num_samples=num_samples, num_thin=n_thin, num_warmup=1000)

W_fit = np.moveaxis(fit['W'],[0,1,2], [1,2,0])
V_fit = np.moveaxis(fit['V'],[0,1,2], [1,2,0])
P_fit = expit(np.einsum('tik, tjk -> tij',W_fit, V_fit ))

P_last = P_fit[-1]
np.mean(np.square(P_last[ii,jj] - y))

