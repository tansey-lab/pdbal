import numpy as np
from dbal import DBALMFSelector, DBALBernMF, DBALNormMF
from eig import EIGMFSelector, EIGBernMF, EIGNormMF
from var import VarMFSelector, VarBernMF, VarNormMF
from mps import MPSMFSelector, MPSNormMF
from dists import MFDistance, MFMSEDistance, MFRowMSEDistance, MFMaxCoordDistance
from mf_models import BayesianMFModel, BayesBernMFModel, BayesNormMFModel
from copy import deepcopy
from numpy.linalg import norm
from scipy.special import expit
from tqdm import trange
import argparse
import os
import pickle
import random

class MFDistribution():
    def __init__(self,
                 W:np.ndarray,
                 V:np.ndarray,
                 bsize:int=1,
                 stdv:float=0.25, 
                 **kwargs):

        self.W = W
        self.V = V
        self.prod =  np.einsum('ik, jk -> ij', self.W, self.V)
        self.stdv = stdv


        ## Chunk up rows and columns into groups of size bsize
        row_indices = np.arange(self.n_rows)
        col_indices = np.arange(self.n_cols)

        row_k = int(len(row_indices)/bsize)
        col_k = int(len(col_indices)/bsize)

        row_groups = np.array_split(row_indices, row_k)
        col_groups = np.array_split(col_indices, col_k)

        self.index_pairs = []
        for rgroup in row_groups:
            for cgroup in col_groups:
                I = np.array(np.meshgrid(rgroup, cgroup)).T.reshape(-1,2)
                self.index_pairs.append( (I[:,0], I[:,1]) )

    def sample_observations(self, ii:np.ndarray, jj:np.ndarray):

        mu = self.prod[ii, jj]
        n = len(mu)
        y = mu + self.stdv*np.random.standard_normal(size=n)

        return(y)


def active_mf(distribution:MFDistribution, 
              dbal_selector:DBALMFSelector, 
              eig_selector:EIGMFSelector,
              var_selector:VarMFSelector,
              mps_selector:MPSMFSelector,
              model:BayesianMFModel, 
              distance:MFDistance,
              nqueries:int,
              fname:str):

    W = distribution.W
    V = distribution.V
    n_rows = W.shape[0]
    n_cols = V.shape[0]
    prod = distribution.prod
    index_pairs = distribution.index_pairs
    n_pairs = len(index_pairs)

    print("Cloning models...")
    eig_model = deepcopy(model)
    dbal_model = deepcopy(model)
    var_model = deepcopy(model)
    mps_model = deepcopy(model)
    random_model = deepcopy(model)
    

    ## Initialize with 10 random points
    n_init = 10
    ii = np.random.choice(n_rows, size=n_init)
    jj = np.random.choice(n_cols, size=n_init)
    yy = distribution.sample_observations(ii, jj)

    eig_model.update(ii,jj,yy)
    dbal_model.update(ii,jj,yy)
    var_model.update(ii,jj,yy)
    random_model.update(ii,jj,yy)


    results = []
    n_samples = 300
    print("Beginning loop")
    for t in trange(nqueries):

        ## EIG query:
        idx = eig_selector.select(model=eig_model, index_pairs=index_pairs)
        ii, jj = index_pairs[idx]
        yy = distribution.sample_observations(ii, jj)

        eig_model.update(ii,jj,yy)
        avg_dist = eig_model.eval(n=n_samples, W=W, V=V, prod=prod, distance=distance)
        results.append({"Query":t, 
                        "Strategy":"EIG", 
                        "Objective distance":avg_dist})


        ## DBAL query:
        idx = dbal_selector.select(model=dbal_model, index_pairs=index_pairs)
        ii, jj = index_pairs[idx]
        yy = distribution.sample_observations(ii, jj)

        dbal_model.update(ii,jj,yy)
        avg_dist = dbal_model.eval(n=n_samples, W=W, V=V, prod=prod, distance=distance)
        results.append({"Query":t, 
                        "Strategy":"DBAL", 
                        "Objective distance":avg_dist})

        ## Variance query:
        idx = var_selector.select(model=var_model, index_pairs=index_pairs)
        ii, jj = index_pairs[idx]
        yy = distribution.sample_observations(ii, jj)

        var_model.update(ii,jj,yy)
        avg_dist = var_model.eval(n=n_samples, W=W, V=V, prod=prod, distance=distance)
        results.append({"Query":t, 
                        "Strategy":"Var", 
                        "Objective distance":avg_dist})

        ## MPS query:
        idx = mps_selector.select(model=mps_model, index_pairs=index_pairs)
        ii, jj = index_pairs[idx]
        yy = distribution.sample_observations(ii, jj)

        mps_model.update(ii,jj,yy)
        avg_dist = mps_model.eval(n=n_samples, W=W, V=V, prod=prod, distance=distance)
        results.append({"Query":t, 
                        "Strategy":"MPS", 
                        "Objective distance":avg_dist})

        ## Random query:
        idx = np.random.choice(n_pairs)
        ii, jj = index_pairs[idx]
        yy = distribution.sample_observations(ii, jj)

        random_model.update(ii,jj,yy)
        avg_dist = random_model.eval(n=n_samples, W=W, V=V, prod=prod, distance=distance)
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
    parser.add_argument('--obj', type=str, default='mse', help="The objective we use.")
    parser.add_argument('--dataset', type=str, default='brenton', help="brenton, marshall or porkka.")
    
    parser.add_argument('--bsize', type=int, default=5, help="Size of row/col groupings.")

    parser.add_argument('--stdv', type=float, default=0.25, help="Gaussian noise for linear regression.")
    parser.add_argument('--nqueries', type=int, default=100, help="Number of rounds.")

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    nqueries = args.nqueries
    bsize = args.bsize

    stdv = args.stdv
    length = args.length
    obj = args.obj
    dataset = args.dataset
    
    ## Folder name
    folder = os.path.join('results', dataset, obj)
    
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, str(args.seed)+".pkl")

    ## Load in the embeddings
    dataset_fname = "../data/"+dataset+"_fit.pkl"
    with open(dataset_fname, 'rb') as io:
        fit = pickle.load(io)
    W = fit['W']
    V = fit['V']
    n_rows, n_features = W.shape
    n_cols, _ = V.shape

    distribution = MFDistribution(W, V, bsize, stdv)

    ## Choose distances
    if obj == 'mse':
        distance = MFMSEDistance()
    elif obj == 'row':
        distance = MFRowMSEDistance(row=0)
    elif obj == 'max':
        distance = MFMaxCoordDistance()

    n_samples = 100
    max_triples = 1000

    ## Choose models + selectors
    model = BayesNormMFModel(n_rows=n_rows, n_cols=n_cols, n_features=n_features, sigma=stdv)
    eig_selector = EIGNormMF(n_samples=n_samples, sigma=stdv)
    dbal_selector = DBALNormMF(n_samples=n_samples, dist=distance, max_triples=max_triples)
    var_selector = VarNormMF(n_samples=n_samples)
    mps_selector = MPSNormMF(n_samples=n_samples, dist=distance, sigma=stdv)

    result = active_mf(distribution=distribution, 
                               dbal_selector=dbal_selector, 
                               eig_selector=eig_selector, 
                               var_selector=var_selector,
                               mps_selector=mps_selector,
                               model=model, 
                               distance=distance, 
                               nqueries=nqueries,
                               fname=fname)
    