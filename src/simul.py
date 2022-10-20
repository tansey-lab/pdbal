import numpy as np
from dbal import DBALSelector, DBALLinearRegression, DBALLogisticRegression, DBALPoissonRegression, DBALBetaRegression
from eig import EIGSelector, EIGLinearRegression, EIGLogisticRegression, EIGPoissonRegression, EIGBetaRegression
from var import VarSelector, VarLinearRegression, VarLogisticRegression, VarPoissonRegression, VarBetaRegression
from dists import Distance, MaxCoordDistance, CoordSignDistance, EuclideanDistance, CoordInfluenceDistance, KendallDistance
from models import BayesianModel, BayesLinearRegression, BayesLogisticRegression, BayesPoissonRegression, BayesBetaRegression
from copy import deepcopy
from numpy.linalg import norm
from scipy.special import expit
from tqdm import trange
import argparse
import os
import pickle
import random

class MixtureDistribution():
    def __init__(self, 
                 p:float, 
                 dim:int, 
                 mode:str, 
                 bsize:int, 
                 length:float=2.0, 
                 stdv:float=0.25, 
                 phi:float=2.5, 
                 **kwargs):
        self.p = p 
        self.dim = dim
        self.mode = mode
        w_star = np.random.standard_normal(size=dim)
        self.w_star = length*w_star/norm(w_star)
        self.stdv = stdv
        self.bsize = bsize
        self.phi = phi

    def sample_observations(self):
        n = self.bsize
        X = np.random.standard_normal(size=(n, self.dim))
        sparsity_pattern = (np.random.random_sample(size=(n, self.dim)) <= 1.0/self.dim).astype(float)
        
        ## Generate from mixture
        sparse_comp = (np.random.random_sample(size=n) <= self.p)
        X[sparse_comp] = X[sparse_comp]*sparsity_pattern[sparse_comp]

        ## Normalize rows
        norms = norm(X, axis=1)
        pos_norm = norms > 0.0
        X[pos_norm] = X[pos_norm]/norms[pos_norm,np.newaxis]

        ground_y = np.dot(X, self.w_star)

        if self.mode == 'linreg':
            y = ground_y + self.stdv*np.random.standard_normal(size=n)
        elif self.mode == 'logreg':
            probs = expit(ground_y)
            y = (np.random.rand(n) < probs).astype(int)
        elif self.mode == 'poisson':
            y = np.random.poisson(np.exp(ground_y))
        elif self.mode == 'beta':
            mu = expit(ground_y)
            a = self.phi*mu
            b = self.phi*(1-mu)
            y = np.clip(np.random.beta(a,b), a_min=0.001, a_max=0.999) ## Numerical stability
        return(X, y)


def active_regression(distribution:MixtureDistribution, 
                      dbal_selector:DBALSelector, 
                      eig_selector:EIGSelector,
                      var_selector:VarSelector,
                      model:BayesianModel, 
                      distance:Distance,
                      nqueries:int,
                      fname:str):

    w_star = distribution.w_star

    print("Cloning models...")
    eig_model = deepcopy(model)
    dbal_model = deepcopy(model)
    var_model = deepcopy(model)
    random_model = deepcopy(model)
    

    ## Initialize with 2 random points
    X, y = distribution.sample_observations()
    for i in range(2):
        eig_model.update(x=X[i], y=y[i])
        dbal_model.update(x=X[i], y=y[i])
        var_model.update(x=X[i], y=y[i])
        random_model.update(x=X[i], y=y[i])


    results = []
    n_samples = 300
    print("Beginning loop")
    for t in trange(nqueries):
        ## Draw from the mixture distribution
        X, y = distribution.sample_observations()
        bsize, _ = X.shape

        ## EIG query:
        idx = eig_selector.select(model=eig_model, X=X)
        eig_model.update(x=X[idx], y=y[idx])
        avg_dist = eig_model.eval(n=n_samples, w_star=w_star, distance=distance)
        results.append({"Query":t, 
                        "Strategy":"EIG", 
                        "Objective distance":avg_dist})


        ## DBAL query:
        idx = dbal_selector.select(model=dbal_model, X=X)
        dbal_model.update(x=X[idx], y=y[idx])
        avg_dist = dbal_model.eval(n=n_samples, w_star=w_star, distance=distance)
        results.append({"Query":t, 
                        "Strategy":"DBAL", 
                        "Objective distance":avg_dist})

        ## Variance query:
        idx = var_selector.select(model=var_model, X=X)
        var_model.update(x=X[idx], y=y[idx])
        avg_dist = var_model.eval(n=n_samples, w_star=w_star, distance=distance)
        results.append({"Query":t, 
                        "Strategy":"Var", 
                        "Objective distance":avg_dist})

        ## Random query:
        idx = np.random.choice(bsize)
        random_model.update(x=X[idx], y=y[idx])
        avg_dist = random_model.eval(n=n_samples, w_star=w_star, distance=distance)
        results.append({"Query":t, 
                        "Strategy":"Random", 
                        "Objective distance":avg_dist})
        ## Save out results
        with open(fname, 'wb') as io:
            pickle.dump(results, io)

    return(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation arguments')
    parser.add_argument('--seed', type=int, default=100, help="The random seed.")
    parser.add_argument('--obj', type=str, default='first', help="The objective we use.")
    parser.add_argument('--model', type=str, default='linreg', help="The model we use.")
    parser.add_argument('--p', type=float, default=0.1, help="Mixing proportions of the distribution.")
    parser.add_argument('--stdv', type=float, default=0.25, help="Gaussian noise for linear regression.")
    parser.add_argument('--phi', type=float, default=3.0, help="Beta regression parameter.")
    parser.add_argument('--d', type=int, default=10, help="Dimension of data.")
    parser.add_argument('--nqueries', type=int, default=100, help="Number of rounds.")
    parser.add_argument('--length', type=float, default=2.0, help="Length of ground truth w^* vector.")

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    nqueries = args.nqueries
    d = args.d 
    p = args.p
    stdv = args.stdv
    length = args.length
    obj = args.obj
    model_type = args.model
    phi = args.phi
    
    ## Folder name
    folder = os.path.join('results', model_type, obj)
    
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, str(args.seed)+".pkl")
    
    distribution = MixtureDistribution(p=p, dim=d, mode=model_type, bsize=2000, length=length, stdv=stdv, phi=phi)

    ## Choose distances
    if obj == 'first':
        distance = CoordSignDistance(idx=0)
    elif obj == 'euclidean':
        distance = EuclideanDistance()
    elif obj == 'max':
        distance = MaxCoordDistance()
    elif obj == 'kendall':
        distance = KendallDistance()
    elif obj == 'influence':
        k = int(d/2.)
        coords = np.arange(k)
        X, _ = distribution.sample_observations()
        X = X[np.arange(100),:]
        distance = CoordInfluenceDistance(X_data=X, coords=coords)

    n_samples = 300
    max_triples = 5000
    dbal_kwargs = {'n_samples':n_samples, 'dist':distance, 'max_triples':max_triples}
    ## Choose models + selectors
    if model_type == 'linreg':
        obs_var = np.square(stdv)
        var_0 = length/d
        model = BayesLinearRegression(d=d, var_0=var_0, obs_var=obs_var)
        eig_selector = EIGLinearRegression()
        dbal_selector = DBALLinearRegression(**dbal_kwargs)
        var_selector = VarLinearRegression()
    elif model_type == 'logreg':
        model = BayesLogisticRegression(d=d)
        eig_selector = EIGLogisticRegression(n_samples=n_samples)
        dbal_selector = DBALLogisticRegression(**dbal_kwargs)
        var_selector = VarLogisticRegression(n_samples=n_samples)
    elif model_type == 'poisson':
        model = BayesPoissonRegression(d=d)
        eig_selector = EIGPoissonRegression(n_samples=n_samples)
        dbal_selector = DBALPoissonRegression(**dbal_kwargs)
        var_selector = VarPoissonRegression(n_samples=n_samples)
    elif model_type == 'beta':
        model = BayesBetaRegression(d=d, phi=phi)
        eig_selector = EIGBetaRegression(n_samples=n_samples)
        dbal_kwargs['max_triples'] = 2500
        dbal_selector = DBALBetaRegression(**dbal_kwargs)
        var_selector = VarBetaRegression(n_samples=n_samples)

    result = active_regression(distribution=distribution, 
                               dbal_selector=dbal_selector, 
                               eig_selector=eig_selector, 
                               var_selector=var_selector,
                               model=model, 
                               distance=distance, 
                               nqueries=nqueries,
                               fname=fname)
    