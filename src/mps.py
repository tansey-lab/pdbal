import abc
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import entropy
from mf_models import BayesianMFModel, BayesBernMFModel, BayesNormMFModel
from reg_models import BayesianModel, BayesBetaRegression, BayesLinearRegression, BayesLogisticRegression, BayesPoissonRegression
from utils import poisson_probs, beta_probs, continuous_entropy, beta_entropy, norm_probs
from dists import MFDistance


class MPSSelector():
    __metaclass__ = abc.ABCMeta
    def __init__(self, n_samples:int, **kwargs):
        self.n_samples = n_samples
    
    def select(self, model:BayesianModel, X:np.ndarray, **kwargs)->int:
        raise NotImplementedError





class MPSMFSelector():
    __metaclass__ = abc.ABCMeta
    def __init__(self, n_samples:int, dist:MFDistance, n_model_samples:int=50, **kwargs):
        self.n_samples = n_samples
        self.distance = dist
        self.n_model_samples = n_model_samples

    def select(self, model:BayesianMFModel, index_pairs:list, **kwargs)->int:
        raise NotImplementedError



class MPSNormMF(MPSMFSelector):
    def __init__(self, n_samples:int, dist:MFDistance, sigma:float, n_model_samples:int=50, **kwargs):
        super().__init__(n_samples, dist, n_model_samples, **kwargs)
        self.sigma = sigma

    def select(self, model:BayesNormMFModel, index_pairs:list, **kwargs)->int:
        W_list, V_list = model.sample(self.n_samples)

        mu = np.einsum('tik, tjk -> tij', W_list, V_list) ## n x dim1 x dim2
        
        mu_star = mu[0,:,:]
        W_star = W_list[0,:,:]
        V_star = V_list[0,:,:]

        _, r, c = mu.shape
        y = mu_star + self.sigma*np.random.standard_normal(size=(self.n_model_samples, r, c)) ## m x dim1 x dim2

        mu_rest = mu[1:,:,:]
        N, _, _ = mu_rest.shape
        W_rest = W_list[1:,:,:]
        V_rest = V_list[1:,:,:]

        dists = np.array([self.distance.distance(W_rest[i], V_rest[i], mu_rest[i], W_star, V_star, mu_star) for i in range(N)])
        
        log_likelihoods = -np.log(self.sigma*np.sqrt(2.* np.pi) ) - 0.5*np.square((y[np.newaxis,:,:,:]  - mu_rest[:,np.newaxis,:,:])/self.sigma) ## N x m x dim1 x dim2

        ## Combine into groups
        ll_group = np.empty((N, self.n_model_samples, len(index_pairs)))
        for i, (ii,jj) in enumerate(index_pairs):
            ll_group[:, :, i] = np.sum( log_likelihoods[:, :, ii, jj] , axis=-1)

        ll_sums = logsumexp(ll_group, axis=0, keepdims=True)
        ll_group = ll_group - ll_sums
        probs = np.exp(ll_group) ## N x m x pairs

        
        per_y_scores = np.sum(probs*dists[:,np.newaxis, np.newaxis], axis=0) ## m x pairs
        scores = np.mean(per_y_scores, axis=0) ## pairs
        idx = np.argmin(scores)
        return(idx)