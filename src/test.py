import stan 
import numpy as np
from scipy.special import expit, logit


bern_fm = """
data {
    int<lower=1> n_rows;                              // Number of rows
    int<lower=1> n_cols;                              // Number of columns
    int<lower=1> n_features;                          // Number of features 
    int<lower=1> n_entries;                           // Number of entries in matrix 
    array[n_entries] int<lower=1,upper=n_rows> ii;    // Row indices
    array[n_entries] int<lower=1,upper=n_cols> jj;    // Col indices
    array[n_entries] int<lower=0, upper=1> y;           // observations (force int) 
}
parameters {
    matrix[n_rows, n_features] W;         
    matrix[n_cols, n_features] V;
}

model {
    for (n in 1:n_entries) {
            y[n] ~ bernoulli_logit( dot_product(W[ii[n],:], V[jj[n],:]) );
    }

    for (n in 1:n_cols) {
        V[n,:] ~  std_normal();
    }
    for (n in 1:n_rows) {
        W[n,:] ~ std_normal();
    }
}
"""

n_rows = 20
n_cols = 50
n_features = 3
W = np.random.standard_normal(size=(n_rows,n_features))
V = np.random.standard_normal(size=(n_cols,n_features))
P = expit( np.dot(W, V.transpose()) )

n_entries = 250
ii = np.random.choice(n_rows, size=n_entries)
jj = np.random.choice(n_cols, size=n_entries)


z = np.random.uniform(size=n_entries)
y = (z<P[ii,jj]).astype(int)

dataset = {"n_rows":n_rows, "n_cols":n_cols, "n_features":n_features, "n_entries":n_entries, "ii":(ii+1), "jj":(jj+1), "y":y}

num_samples = 100
n_thin = 5
model = stan.build(bern_fm, data=dataset)
fit = model.sample(num_chains=2, num_samples=num_samples, num_thin=n_thin, num_warmup=1000)

W_fit = np.moveaxis(fit['W'],[0,1,2], [1,2,0])
V_fit = np.moveaxis(fit['V'],[0,1,2], [1,2,0])
P_fit = expit(np.einsum('tik, tjk -> tij',W_fit, V_fit ))

np.mean(np.square(P_fit[0] - P))

