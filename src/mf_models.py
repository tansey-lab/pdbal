import numpy as np
import abc
from numpy.linalg import multi_dot
from dists import MFDistance
import stan 


norm_mf = """
data {
    int<lower=1> n_rows;                              // Number of rows
    int<lower=1> n_cols;                              // Number of columns
    int<lower=1> n_features;                          // Number of features 
    int<lower=1> n_entries;                           // Number of entries in matrix 
    real<lower=0> sigma;
    array[n_entries] int<lower=1,upper=n_rows> ii;    // Row indices
    array[n_entries] int<lower=1,upper=n_cols> jj;    // Col indices
    array[n_entries] real y;                          // observations (force int) 
}
parameters {
    matrix[n_rows, n_features] W;         
    matrix[n_cols, n_features] V;
}

model {
    for (n in 1:n_entries) {
            y[n] ~ normal( dot_product(W[ii[n],:], V[jj[n],:]) , sigma);
    }

    for (n in 1:n_cols) {
        V[n,:] ~  std_normal();
    }
    for (n in 1:n_rows) {
        W[n,:] ~ std_normal();
    }
}
"""


bern_mf = """
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


class BayesianMFModel(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, **kwargs):
        self.ii = []
        self.jj = []
        self.y = []

    def sample(self, n:int):
        raise NotImplementedError

    def update(self, ii_new:np.ndarray, jj_new:np.ndarray, y_new:np.ndarray, **kwargs):
        self.ii.extend(ii_new)
        self.jj.extend(jj_new)
        self.y.extend(y_new)

    def eval(self, n:int, W:np.ndarray, V:np.ndarray, prod:np.ndarray, distance:MFDistance, **kwargs):
        W_list, V_list = self.sample(n) ## n x r x dim, n x c x dim
        prod_list = np.einsum('tik, tjk -> tij',W_list, V_list) ## n x r x c
        avg_dist = distance.average_distance(W_list, V_list, prod_list, W, V, prod)
        return(avg_dist)


#############################################
#### PyStan models for generalized linear regression.
#############################################
class PyStanMFModel(BayesianMFModel):
    def __init__(self, n_rows:int, n_cols:int, n_features:int, **kwargs):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_features = n_features
        self.thin = 5
        self.nchains = 2

    def get_model_code(self):
        raise NotImplementedError

    def get_dataset(self):
        raise NotImplementedError

    def sample(self, n:int):
        if len(self.y) == 0:
            W = np.random.standard_normal(size=(n, self.n_rows, self.n_features))
            V = np.random.standard_normal(size=(n, self.n_cols, self.n_features))
        else:
            dataset = self.get_dataset()
            model_code = self.get_model_code()
            seed = np.random.randint(0, 2**20)
            model = stan.build(model_code, data=dataset, random_seed=seed)

            num_samples = int(n/self.nchains)*self.thin
            fit = model.sample(num_chains=self.nchains, num_samples=num_samples, num_thin=self.thin, num_warmup=num_samples)
            W = np.moveaxis(fit['W'],[0,1,2], [1,2,0])
            V = np.moveaxis(fit['V'],[0,1,2], [1,2,0])
        return(W, V)



class BayesBernMFModel(PyStanMFModel):        
    def __init__(self, n_rows:int, n_cols:int, n_features:int, **kwargs):
        super().__init__(n_rows=n_rows, n_cols=n_cols, n_features=n_features)

    def get_model_code(self):
        return(bern_mf)

    def get_dataset(self):
        ii = np.array(self.ii).astype(int)
        jj = np.array(self.jj).astype(int)
        y = np.array(self.y).astype(int)
        n_entries = len(ii)
        dataset = {"n_rows":self.n_rows, "n_cols":self.n_cols, "n_features":self.n_features, "n_entries":n_entries, "ii":(ii+1), "jj":(jj+1), "y":y}
        return(dataset)

class BayesNormMFModel(PyStanMFModel):        
    def __init__(self, n_rows:int, n_cols:int, n_features:int, sigma:float, **kwargs):
        super().__init__(n_rows=n_rows, n_cols=n_cols, n_features=n_features)
        self.sigma = sigma

    def get_model_code(self):
        return(norm_mf)

    def get_dataset(self):
        ii = np.array(self.ii).astype(int)
        jj = np.array(self.jj).astype(int)
        y = np.array(self.y).astype(int)
        n_entries = len(ii)
        dataset = {"n_rows":self.n_rows, "n_cols":self.n_cols, "n_features":self.n_features, "n_entries":n_entries, "sigma":self.sigma, "ii":(ii+1), "jj":(jj+1), "y":y}
        return(dataset)

