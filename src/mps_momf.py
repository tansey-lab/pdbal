import abc
import numpy as np
from scipy.special import logsumexp
from momf_models import NormMixtureMFModel
from dists import MOMFDistance



class MPSNormMOMF():
    def __init__(self, n_samples:int, dist:MOMFDistance, sigma:float, n_model_samples:int=10, **kwargs):
        self.n_samples = n_samples
        self.dist = dist
        self.n_model_samples = n_model_samples
        self.sigma = sigma

    def select(self, model:NormMixtureMFModel, **kwargs)->int:
        W_list, V_list, z_list = model.sample(self.n_samples)

        mu = np.einsum('tik, tjk -> tij', W_list, V_list) ## n x dim1 x dim2
        
        mu_star = mu[0,:,:]
        W_star = W_list[0,:,:]
        V_star = V_list[0,:,:]
        z_star = z_list[0,:]

        _, r, c = mu.shape
        y = mu_star + self.sigma*np.random.standard_normal(size=(self.n_model_samples, r, c)) ## m x dim1 x dim2

        mu_rest = mu[1:,:,:]
        N, _, _ = mu_rest.shape
        W_rest = W_list[1:,:,:]
        V_rest = V_list[1:,:,:]
        z_rest = z_list[1:,:]

        dists = np.array([self.dist.distance(W_rest[i], V_rest[i], z_rest[i], W_star, V_star, z_star) for i in range(N)])
        
        log_likelihoods = -np.log(self.sigma*np.sqrt(2.* np.pi) ) - 0.5*np.square((y[np.newaxis,:,:,:]  - mu_rest[:,np.newaxis,:,:])/self.sigma) ## N x m x dim1 x dim2

        probs = np.exp(log_likelihoods - logsumexp(log_likelihoods, axis=0, keepdims=True) ) ## N x m x dim1 x dim2
        
        per_y_scores = np.sum(probs*dists[:,np.newaxis, np.newaxis, np.newaxis], axis=0) ## m x dim1 x dim2
        scores = np.mean(per_y_scores, axis=0) ## dim1 x dim2

        # ii, jj =  np.where(scores <= np.partition(scores.flatten(), 2)[2])
        ii, jj = np.where(scores <= np.min(scores))
        index = np.random.choice(len(ii))
        i = ii[index]
        j = jj[index]
        return(i, j)