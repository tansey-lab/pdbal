import abc
import numpy as np
from scipy.special import expit
from scipy.stats import entropy
from utils import norm_probs, continuous_entropy
from momf_models import NormMixtureMFModel


class EIGNormMOMF():
    def __init__(self, n_samples:int, sigma:float, **kwargs):
        self.n_samples = n_samples
        self.sigma = sigma

    def select(self, model:NormMixtureMFModel, **kwargs):
        W_list, V_list, z_list = model.sample(self.n_samples)

        mu = np.einsum('tik, tjk -> tij', W_list, V_list) ## n x dim1 x dim2
        
        probs, x = norm_probs(mu, self.sigma)  ## n x dim1 x dim2 x O

        marginal_probs = np.mean(probs, axis=0) ## dim1 x dim2 x O
        marginal_entropies = continuous_entropy(marginal_probs, x, axis=-1) ## dim1 x dim2

        i, j = np.unravel_index(np.argmax(marginal_entropies, axis=None), marginal_entropies.shape)
        return(i,j)