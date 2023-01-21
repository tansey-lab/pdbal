import abc
import numpy as np
from scipy.special import expit
from reg_models import BayesianModel, BayesLogisticRegression, BayesPoissonRegression, BayesBetaRegression, BayesLinearRegression
from mf_models import BayesianMFModel, BayesBernMFModel, BayesNormMFModel

def var_outcome(conditional_means:np.ndarray, conditional_variances:np.ndarray)->int:

    variances = np.var(conditional_means, axis=0) + np.mean(conditional_variances, axis=0)

    return(np.argmax(variances))


class VarSelector():
    __metaclass__ = abc.ABCMeta
    def __init__(self, n_samples:int, **kwargs):
        self.n_samples = n_samples

    def select(self, model:BayesianModel, X:np.ndarray, **kwargs)->int:
        raise NotImplementedError



class VarLogisticRegression(VarSelector):
    def __init__(self, n_samples:int):
        super().__init__(n_samples=n_samples)
    
    def select(self, model:BayesLogisticRegression, X:np.ndarray, **kwargs)->int:
        n, _ = X.shape

        W = model.sample(self.n_samples)

        pos_probs = expit(np.einsum('ik, jk -> ij', W, X) )

        conditional_variances = pos_probs * (1.0-pos_probs)

        idx = var_outcome(conditional_means=pos_probs, conditional_variances=conditional_variances)
        
        return(idx)


class VarLinearRegression(VarSelector):
    def __init__(self, n_samples:int=0):
        super().__init__(n_samples=n_samples)

    def select(self, model:BayesLinearRegression, X:np.ndarray, **kwargs)->int:
        ## Predicted variances
        pred_vars = np.einsum('ij, ik, jk -> i', X, X, model.cov, optimize='optimal')

        idx = np.argmax(pred_vars)

        return(idx)

class VarPoissonRegression(VarSelector):
    def __init__(self, n_samples:int):
        super().__init__(n_samples=n_samples)
    
    def select(self, model:BayesPoissonRegression, X:np.ndarray, **kwargs)->int:
        W = model.sample(self.n_samples)

        lams = np.exp( np.einsum('ik, jk -> ij', W, X) ) 

        ## Poisson mean = variance = lams
        idx = var_outcome(conditional_means=lams, conditional_variances=lams)

        return(idx)

class VarBetaRegression(VarSelector):
    def __init__(self, n_samples:int):
        super().__init__(n_samples=n_samples)
        

    def select(self, model:BayesBetaRegression, X:np.ndarray, **kwargs)->int:
        phi = model.phi
        W = model.sample(self.n_samples)

        mu = expit( np.einsum('ik, jk -> ij', W, X) )
        a = phi*mu
        b = phi*(1-mu)

        ## probs: n, m, K
        conditional_variances = a*b/(np.square(a + b) * (1. + a + b))
        idx = var_outcome(conditional_means=mu, conditional_variances=conditional_variances)

        return(idx)

class VarMFSelector():
    __metaclass__ = abc.ABCMeta
    def __init__(self, n_samples:int, **kwargs):
        self.n_samples = n_samples
    
    def select(self, model:BayesianMFModel, index_pairs:list, **kwargs)->int:
        raise NotImplementedError


def var_mf_outcome(conditional_means:np.ndarray, conditional_variances:np.ndarray, index_pairs:list)->int:

    variances = np.var(conditional_means, axis=0) + np.mean(conditional_variances, axis=0)

    group_variances = [np.sum(variances[ii, jj])  for ii,jj in index_pairs]

    return(np.argmax(group_variances))


class VarBernMF(VarMFSelector):
    def __init__(self, n_samples:int, **kwargs):
        self.n_samples = n_samples
    
    def select(self, model:BayesBernMFModel, index_pairs:list, **kwargs)->int:
        W_list, V_list = model.sample(self.n_samples)

        pos_probs = expit(np.einsum('tik, tjk -> tij', W_list, V_list))

        conditional_variances = pos_probs * (1.0-pos_probs)

        idx = var_mf_outcome(conditional_means=pos_probs, conditional_variances=conditional_variances, index_pairs=index_pairs)

        return(idx)

class VarNormMF(VarMFSelector):
    def __init__(self, n_samples:int, **kwargs):
        self.n_samples = n_samples

    def select(self, model:BayesNormMFModel, index_pairs:list, **kwargs)->int:
        W_list, V_list = model.sample(self.n_samples)

        conditional_means = np.einsum('tik, tjk -> tij', W_list, V_list) ## n x dim1 x dim2
        
        variances = np.var(conditional_means, axis=0)

        group_variances = [np.sum(variances[ii, jj])  for ii,jj in index_pairs]

        idx = np.argmax(group_variances)
        return(idx)