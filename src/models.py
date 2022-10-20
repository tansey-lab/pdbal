import numpy as np
import abc
from numpy.linalg import multi_dot
from dists import Distance, EuclideanDistance
import stan 


logreg_model = """
data {
    int<lower=0> N;                     // count of observations
    int<lower=0> D;                     // count of exog features
    matrix[N, D] X;                     // features
    array[N] int<lower=0, upper=1> y;         // endog feature (force int)
}
parameters {
    vector[D] w;                        // coeffs
}
transformed parameters {}
model {
    to_vector(w) ~ std_normal();        // prior
    y ~ bernoulli_logit(X * w);         // Logistic bernoulli
}
"""

poisson_model = """
data {
    int<lower=0> N;                     // count of observations
    int<lower=0> D;                     // count of exog features
    matrix[N, D] X;                     // exog features
    array[N] int<lower=0> y;                  // endog feature (force int)
}
parameters {
    vector[D] w;                        //  coeffs
}
transformed parameters {}
model {
    to_vector(w) ~ std_normal();             // prior

    y ~ poisson(exp(X * w));         // poisson likelihood
}
"""


beta_model = """
data {
    int<lower=0> N;                        // num observations
    int<lower=0> D;                        // num features
    real<lower=0> phi;                     // beta proportion phi
    matrix[N, D] X;                        // feature matrix
    array[N] real<lower=0, upper=1> y;     // observations
}
parameters {
    vector[D] w;                        // coeffs
}
transformed parameters {}
model {
    vector[N] mu;
    to_vector(w) ~ std_normal();              // prior
    mu = 0.001 + (0.998 * inv_logit(X * w));  // Numerical stability
    y ~ beta_proportion(mu, phi);             // beta likelihood
}
"""


class BayesianModel():
    __metaclass__ = abc.ABCMeta
    def __init__(self, **kwargs):
        self.X = []
        self.y = []

    def sample(self, n:int):
        raise NotImplementedError

    def update(self, x:np.ndarray, y:float, **kwargs):
        self.X.append(x)
        self.y.append(y)

    def eval(self, n:int, w_star:np.ndarray, distance:Distance, **kwargs):
        W = self.sample(n)

        avg_dist = distance.average_distance(W, w_star)

        return(avg_dist)

#############################################
#### PyStan models for generalized linear regression.
#############################################
class PyStanModel(BayesianModel):
    def __init__(self, d:int, **kwargs):
        super().__init__()
        self.d = d
        self.thin = 5
        self.nchains = 2

    def get_model_code(self):
        raise NotImplementedError

    def get_dataset(self):
        raise NotImplementedError

    def sample(self, n:int):
        if len(self.X) == 0:
            W = np.random.standard_normal(size=(n, self.d))
        else:
            dataset = self.get_dataset()
            model_code = self.get_model_code()
            model = stan.build(model_code, data=dataset)

            num_samples = int(n/self.nchains)*self.thin
            fit = model.sample(num_chains=self.nchains, num_samples=num_samples, num_thin=self.thin, num_warmup=num_samples)
            W = fit['w'].transpose()
        return(W)

###########################
#### PyStan implementation
###########################
class BayesLogisticRegression(PyStanModel):        
    def __init__(self, d:int):
        super().__init__(d=d)

    def get_model_code(self):
        return(logreg_model)

    def get_dataset(self):
        X = np.stack(self.X)
        y = np.array(self.y).astype(int)
        dataset = {"N":len(y), "D":self.d, "X":X, "y":y}
        return(dataset)


###########################
#### PyStan implementation
###########################
class BayesPoissonRegression(PyStanModel):        
    def __init__(self, d:int):
        super().__init__(d=d)

    def get_model_code(self):
        return(poisson_model)

    def get_dataset(self):
        X = np.stack(self.X)
        y = np.array(self.y).astype(int)
        dataset = {"N":len(y), "D":self.d, "X":X, "y":y}
        return(dataset)


###########################
#### PyStan implementation
###########################
class BayesBetaRegression(PyStanModel):        
    def __init__(self, d:int, phi:float):
        super().__init__(d=d)
        self.phi = phi

    def get_model_code(self):
        return(beta_model)

    def get_dataset(self):
        X = np.stack(self.X)
        y = np.array(self.y).astype(float)
        dataset = {"N":len(y), "D":self.d, "phi": self.phi, "X":X, "y":y}
        return(dataset)



####################################################
########## Implemented from scratch ################
####################################################
class BayesLinearRegression(BayesianModel):
    def __init__(self, d:int, var_0:float, obs_var:float):
        super().__init__()

        self.w = np.zeros(d)
        self.cov = var_0*np.eye(d)
        self.inv_cov = np.eye(d)/var_0
        self.obs_var = obs_var

    def sample(self, n:int):
        results = np.random.multivariate_normal(mean=self.w, cov=self.cov, size=n)
        return(results)

    def update(self, x:np.ndarray, y:float):
        X = np.atleast_2d(x).T
        new_inv_cov = self.inv_cov + np.dot(X, X.T)/self.obs_var
        new_cov = self.cov - multi_dot([self.cov, X, X.T, self.cov])/(self.obs_var + multi_dot([X.T, self.cov, X]))

        self.w = multi_dot([new_cov, self.inv_cov, self.w]) + (y/self.obs_var)*np.dot(new_cov, x)

        self.cov = new_cov
        self.inv_cov = new_inv_cov

