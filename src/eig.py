import abc
import numpy as np
from scipy.special import expit
from scipy.stats import entropy
from mf_models import BayesianMFModel, BayesBernMFModel, BayesNormMFModel
from reg_models import BayesianModel, BayesBetaRegression, BayesLinearRegression, BayesLogisticRegression, BayesPoissonRegression
from utils import poisson_probs, beta_probs, continuous_entropy, beta_entropy, norm_probs

## probs: num samples x num queries x num outcomes 
def eig_finite_outcome(probs:np.ndarray)->int:
    n, m, O = probs.shape 
    marginal_probs = np.mean(probs, axis=0) ## m x O
    marginal_entropies = entropy(marginal_probs, axis=1) ## m

    model_entropies = entropy(probs, axis=2) ## n x m
    conditional_entropies = np.mean(model_entropies, axis=0) ## m

    objective = marginal_entropies - conditional_entropies
    return(np.argmax(objective))


class EIGSelector():
    __metaclass__ = abc.ABCMeta
    def __init__(self, n_samples:int, **kwargs):
        self.n_samples = n_samples
    
    def select(self, model:BayesianModel, X:np.ndarray, **kwargs)->int:
        raise NotImplementedError


class EIGLogisticRegression(EIGSelector):
    def __init__(self, n_samples:int):
        super().__init__(n_samples=n_samples)
    
    def select(self, model:BayesLogisticRegression, X:np.ndarray, **kwargs)->int:
        n, _ = X.shape

        W = model.sample(self.n_samples)

        pos_probs = expit(np.einsum('ik, jk -> ij', W, X) )

        probs = np.empty((self.n_samples, n, 2))
        probs[:,:,0] = pos_probs
        probs[:,:,1] = 1.0-pos_probs

        idx = eig_finite_outcome(probs)

        return(idx)


class EIGLinearRegression(EIGSelector):
    def __init__(self, n_samples:int=0):
        super().__init__(n_samples=n_samples)

    def select(self, model:BayesLinearRegression, X:np.ndarray, **kwargs)->int:
        ## Predicted variances
        pred_vars = np.einsum('ij, ik, jk -> i', X, X, model.cov, optimize='optimal')

        idx = np.argmax(pred_vars)

        return(idx)

class EIGPoissonRegression(EIGSelector):
    def __init__(self, n_samples:int):
        super().__init__(n_samples=n_samples)
    
    def select(self, model:BayesPoissonRegression, X:np.ndarray, **kwargs)->int:
        W = model.sample(self.n_samples)

        lams = np.exp( np.einsum('ik, jk -> ij', W, X) )

        probs = poisson_probs(lams)
        idx = eig_finite_outcome(probs)
        return(idx)

class EIGBetaRegression(EIGSelector):
    def __init__(self, n_samples:int):
        super().__init__(n_samples=n_samples)
        

    def select(self, model:BayesBetaRegression, X:np.ndarray, **kwargs)->int:
        phi = model.phi
        W = model.sample(self.n_samples)

        mu = expit( np.einsum('ik, jk -> ij', W, X) )
        a = phi*mu
        b = phi*(1-mu)

        ## probs: n, m, K
        probs, x = beta_probs(a=a, b=b) 

        # model_entropies = continuous_entropy(probs, x, axis=2) ## n x m
        model_entropies = beta_entropy(a,b)

        marginal_probs = np.mean(probs, axis=0) ## m x O
        marginal_entropies = continuous_entropy(marginal_probs, x, axis=1) ## m

        
        conditional_entropies = np.mean(model_entropies, axis=0) ## m

        objective = marginal_entropies - conditional_entropies

        idx = np.argmax(objective)

        return(idx)




## Matrix-arranged queries are dim1 x dim2
## probs: num samples x dim1 x dim2 x num outcomes 
def eig_mf_finite_outcome(probs:np.ndarray, index_pairs:list)->int:
    n, dim1, dim2, O = probs.shape 
    marginal_probs = np.mean(probs, axis=0) ## dim1 x dim2 x O
    marginal_entropies = entropy(marginal_probs, axis=-1) ## dim1 x dim2

    model_entropies = entropy(probs, axis=-1) ## n x dim1 x dim2
    conditional_entropies = np.mean(model_entropies, axis=0) ## dim1 x dim2

    objective = marginal_entropies - conditional_entropies

    indexed_obj = [np.sum( objective[ii,jj]) for ii,jj in index_pairs]

    return(np.argmax(indexed_obj))



class EIGMFSelector():
    __metaclass__ = abc.ABCMeta
    def __init__(self, n_samples:int, **kwargs):
        self.n_samples = n_samples
    
    def select(self, model:BayesianMFModel, index_pairs:list, **kwargs)->int:
        raise NotImplementedError


class EIGBernMF(EIGMFSelector):
    def __init__(self, n_samples:int, **kwargs):
        self.n_samples = n_samples
    
    def select(self, model:BayesBernMFModel, index_pairs:list, **kwargs)->int:
        W_list, V_list = model.sample(self.n_samples)

        pos_probs = expit(np.einsum('tik, tjk -> tij', W_list, V_list))
        
        _, dim1, dim2 = pos_probs.shape

        probs = np.empty((self.n_samples, dim1, dim2, 2))
        probs[:,:,:,0] = pos_probs
        probs[:,:,:,1] = 1.0-pos_probs

        idx = eig_mf_finite_outcome(probs, index_pairs)
        return(idx)


class EIGNormMF(EIGMFSelector):
    def __init__(self, n_samples:int, sigma:float, **kwargs):
        self.n_samples = n_samples
        self.sigma = sigma

    def select(self, model:BayesNormMFModel, index_pairs:list, **kwargs)->int:
        W_list, V_list = model.sample(self.n_samples)

        mu = np.einsum('tik, tjk -> tij', W_list, V_list) ## n x dim1 x dim2
        
        probs, x = norm_probs(mu, self.sigma)  ## n x dim1 x dim2 x O


        marginal_probs = np.mean(probs, axis=0) ## dim1 x dim2 x O
        marginal_entropies = continuous_entropy(marginal_probs, x, axis=-1) ## dim1 x dim2

        group_ents = [np.sum(marginal_entropies[ii,jj]) for ii,jj in index_pairs]

        idx = np.argmax(group_ents)
        return(idx)

