import numpy as np
from dbal_momf import DBALNormMOMF
from eig_momf import EIGNormMOMF
from var_momf import VarNormMOMF
from mps_momf import MPSNormMOMF
from dists import MOMFDistance, MOMFClusterDistance
from momf_models import NormMixtureMFModel
from copy import deepcopy
from numpy.linalg import norm
from scipy.special import expit
from tqdm import trange
import argparse
import os
import pickle
import random

class MOMFDistribution():
    def __init__(self,
                 n_rows:int,
                 n_cols:int,
                 n_features:int,
                 K:int,
                 **kwargs):

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_features = n_features
        self.K = K

        self.sigma_obs = 0.25
        self.sigma_emb = 0.25
        self.sigma_norm = 2.0
        self.inform_frac = 0.1

        ## Generate column embeddings
        inform_n_cols = int(self.n_cols * self.inform_frac)
        inform_cols = np.random.choice(self.n_cols, size=inform_n_cols, replace=False)
        self.V = 0.5*self.sigma_norm*np.random.standard_normal(size=(self.n_cols, self.n_features))

        for i in inform_cols:
            self.V[i,:] = 8. * self.V[i,:]

        ## Generate mean row embeddings
        self.mu = self.sigma_norm * np.random.standard_normal(size=(self.K, self.n_features))

        ## Generate row embeddings
        self.z = np.random.choice(self.K, size=self.n_rows, replace=True)

        self.W = np.empty((n_rows, n_features))
        for i in range(n_rows):
            self.W[i,:] = self.mu[self.z[i],:] + self.sigma_emb*np.random.standard_normal(size=n_features)


        self.prod =  np.einsum('ik, jk -> ij', self.W, self.V)

    def sample_observations(self, ii:np.ndarray, jj:np.ndarray):
        n = len(ii)
        y = self.prod[ii, jj] + self.sigma_obs*np.random.standard_normal(size=n)
        return(y)


def active_momf(distribution:MOMFDistribution, 
              dbal_selector:DBALNormMOMF, 
              eig_selector:EIGNormMOMF,
              var_selector:VarNormMOMF,
              mps_selector:MPSNormMOMF,
              model:NormMixtureMFModel, 
              distance:MOMFDistance,
              nqueries:int,
              fname:str):

    W = distribution.W
    V = distribution.V
    z = distribution.z
    n_rows = W.shape[0]
    n_cols = V.shape[0]

    print("Cloning models...")
    eig_model = deepcopy(model)
    dbal_model = deepcopy(model)
    var_model = deepcopy(model)
    mps_model = deepcopy(model)
    random_model = deepcopy(model)
    

    ## Initialize with 1 point per column
    ii = np.random.choice(n_rows, size=n_cols, replace=True)
    jj = np.arange(n_cols)
    yy = distribution.sample_observations(ii, jj)

    eig_model.update(ii,jj,yy)
    dbal_model.update(ii,jj,yy)
    var_model.update(ii,jj,yy)
    mps_model.update(ii,jj,yy)
    random_model.update(ii,jj,yy)


    results = []
    n_samples = 100
    print("Beginning loop")
    for t in trange(nqueries):
        ## EIG query:
        i, j = eig_selector.select(model=eig_model)
        ii = np.array([i])
        jj = np.array([j])
        yy = distribution.sample_observations(ii, jj)

        eig_model.update(ii,jj,yy)
        avg_dist = eig_model.eval(n=n_samples, W=W, V=V, z=z, distance=distance)
        results.append({"Query":t, 
                        "Strategy":"EIG", 
                        "Objective distance":avg_dist})


        ## DBAL query:
        i, j = dbal_selector.select(model=dbal_model)
        ii = np.array([i])
        jj = np.array([j])
        yy = distribution.sample_observations(ii, jj)

        dbal_model.update(ii,jj,yy)
        avg_dist = dbal_model.eval(n=n_samples, W=W, V=V, z=z, distance=distance)
        results.append({"Query":t, 
                        "Strategy":"DBAL", 
                        "Objective distance":avg_dist})

        ## Variance query:
        i, j = var_selector.select(model=var_model)
        ii = np.array([i])
        jj = np.array([j])
        yy = distribution.sample_observations(ii, jj)

        var_model.update(ii,jj,yy)
        avg_dist = var_model.eval(n=n_samples, W=W, V=V, z=z, distance=distance)
        results.append({"Query":t, 
                        "Strategy":"Var", 
                        "Objective distance":avg_dist})

        ## MPS query:
        i, j = mps_selector.select(model=mps_model)
        ii = np.array([i])
        jj = np.array([j])
        yy = distribution.sample_observations(ii, jj)

        mps_model.update(ii,jj,yy)
        avg_dist = mps_model.eval(n=n_samples, W=W, V=V, z=z, distance=distance)
        results.append({"Query":t, 
                        "Strategy":"MPS", 
                        "Objective distance":avg_dist})

        ## Random query:
        i = np.random.choice(n_rows)
        j = np.random.choice(n_cols)
        ii = np.array([i])
        jj = np.array([j])
        yy = distribution.sample_observations(ii, jj)

        random_model.update(ii,jj,yy)
        avg_dist = random_model.eval(n=n_samples, W=W, V=V, z=z, distance=distance)
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
    parser.add_argument('--obj', type=str, default='cluster', help="The objective we use.")
    parser.add_argument('--nqueries', type=int, default=100, help="Number of rounds.")

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    nqueries = args.nqueries

    obj = args.obj
    
    ## Folder name
    folder = os.path.join('results', 'momf', obj)
    
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, str(args.seed)+".pkl")

    
    distribution = MOMFDistribution(n_rows=10, n_cols=50, n_features=3, K=2)

    ## Choose distances
    if obj == 'cluster':
        distance = MOMFClusterDistance()
    

    n_samples = 150
    max_triples = 1000

    ## Choose models + selectors
    model = NormMixtureMFModel(n_rows=distribution.n_rows, 
                               n_cols=distribution.n_cols, 
                               n_features=distribution.n_features, 
                               K=distribution.K,
                               sigma_obs=distribution.sigma_obs,
                               sigma_emb=distribution.sigma_emb,
                               sigma_norm=distribution.sigma_norm,
                               thin=10,
                               burnin=1500)

    eig_selector = EIGNormMOMF(n_samples=n_samples, sigma=distribution.sigma_obs)
    dbal_selector = DBALNormMOMF(n_samples=n_samples, dist=distance, max_triples=max_triples)
    var_selector = VarNormMOMF(n_samples=n_samples)
    mps_selector = MPSNormMOMF(n_samples=n_samples, dist=distance, sigma=distribution.sigma_obs)

    result = active_momf(distribution=distribution, 
                               dbal_selector=dbal_selector, 
                               eig_selector=eig_selector, 
                               var_selector=var_selector,
                               mps_selector=mps_selector,
                               model=model, 
                               distance=distance, 
                               nqueries=nqueries,
                               fname=fname)
    