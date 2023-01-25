import stan 
import numpy as np
from scipy.special import expit, logit
import pandas as pd
from alt_min import alternating_minimization
import pickle


df = pd.read_csv("../data/brenton_2022_proc.csv")
ii = df['ii'].values
jj = df['jj'].values
y = df["z-score"].values
n_rows = np.max(ii)+1
n_cols = np.max(jj)+1
n_features = 5
n_entries = len(ii)

W, V = alternating_minimization(ii, jj, y, n_features=n_features, n_epochs=1000)

P = np.einsum('ik, jk -> ij', W, V)
print(np.mean(np.square(P[ii,jj] - y)))

fit = {"W":W, "V":V}

with open("../data/porkka_fit.pkl", 'wb') as io:
    pickle.dump(fit, io)

norm_mf = """
data {
    int<lower=1> n_rows;                              // Number of rows
    int<lower=1> n_cols;                              // Number of columns
    int<lower=1> n_features;                          // Number of features 
    int<lower=1> n_entries;                           // Number of entries in matrix 
    array[n_entries] int<lower=1,upper=n_rows> ii;    // Row indices
    array[n_entries] int<lower=1,upper=n_cols> jj;    // Col indices
    array[n_entries] real y;                          // observations (force int) 
}
parameters {
    matrix[n_rows, n_features] W;         
    matrix[n_cols, n_features] V;
    real<lower=0> sigma;
}

model {
    for (n in 1:n_entries) {
            y[n] ~ normal( dot_product(W[ii[n],:], V[jj[n],:]) , sigma);
    }

    for (n in 1:n_cols) {
        V[n,:] ~ std_normal();
    }
    for (n in 1:n_rows) {
        W[n,:] ~ std_normal();
    }

    sigma ~ inv_gamma(2, 1);
}
"""

dataset = {"n_rows":n_rows, "n_cols":n_cols, "n_features":n_features, "n_entries":n_entries, "ii":(ii+1), "jj":(jj+1), "y":y}

num_samples = 50
n_thin = 5
model = stan.build(norm_mf, data=dataset, random_seed=10)
fit = model.sample(num_chains=1, num_samples=num_samples, num_thin=n_thin, num_warmup=100)

# W_fit = np.moveaxis(fit['W'],[0,1,2], [1,2,0])
# V_fit = np.moveaxis(fit['V'],[0,1,2], [1,2,0])
# P_fit = expit(np.einsum('tik, tjk -> tij',W_fit, V_fit ))

# P = P_fit[-1]
# np.mean(np.square(P[ii,jj] - y))

