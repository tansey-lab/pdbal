import abc
import numpy as np
from scipy.special import logsumexp, comb, expit
from scipy.stats import entropy
from scipy.integrate import simpson
from dists import Distance, MFDistance
from reg_models import BayesianModel, BayesLogisticRegression, BayesPoissonRegression, BayesBetaRegression, BayesLinearRegression
from mf_models import BayesianMFModel, BayesBernMFModel
from utils import poisson_probs, beta_probs, beta_entropy

## Flatten tuples
def flatten(d):
    for i in d:
        yield from [i] if not isinstance(i, tuple) else flatten(i)

def iterCombination_(index, n, k):
    '''Yields the items of the single combination that would be at the provided
    (0-based) index in a lexicographically sorted list of combinations of choices
    of k items from n items [0,n), given the combinations were sorted in 
    descending order. Yields in descending order.
    '''
    nCk = 1
    for nMinusI, iPlus1 in zip(range(n, n - k, -1), range(1, k + 1)):
        nCk *= nMinusI
        nCk //= iPlus1
    curIndex = nCk
    for k in range(k, 0, -1):
        nCk *= k
        nCk //= n
        while curIndex - nCk > index:
            curIndex -= nCk
            nCk *= (n - k)
            nCk -= nCk % k
            n -= 1
            nCk //= n
        n -= 1
        yield n
        
def combination_lookup(index, n, k):
    return( tuple(iterCombination_(index, n, k)) )

## probs: num samples x num queries x num outcomes
## log_triple_dists: len of (idx1, idx2, idx3) x num samples
def dbal_finite_outcomes(probs:np.ndarray, log_triple_dists:np.ndarray, idx1:np.ndarray, idx2:np.ndarray, idx3:np.ndarray):
    triple_likelihood = np.sum(probs[idx1,:,:]*probs[idx2,:,:]*probs[idx3,:,:], axis=2) ## n_triples x m
    triple_dists = np.exp(log_triple_dists)
    scores = np.mean(triple_likelihood*triple_dists, axis=0)
    return(np.argmin(scores))


## probs: num samples x num queries x num sampled space
## x: num sampled space
## log_triple_dists: len of (idx1, idx2, idx3) x num samples
def dbal_continuous_outcomes(probs:np.ndarray,  x:np.ndarray, log_triple_dists:np.ndarray, idx1:np.ndarray, idx2:np.ndarray, idx3:np.ndarray):
    y = probs[idx1,:,:]*probs[idx2,:,:]*probs[idx3,:,:] ## n_triples x m x O
    triple_likelihood = simpson(y, x, axis=2) ## n_triples x m
    triple_dists = np.exp(log_triple_dists)
    scores = np.mean(triple_likelihood*triple_dists, axis=0)
    return(np.argmin(scores))

class DBALSelector():
    __metaclass__ = abc.ABCMeta
    def __init__(self, n_samples:int, dist:Distance, max_triples:int=10000,  dfactor:float=1.0, **kwargs):
        self.n_samples = n_samples
        self.dist = dist
        self.max_triples = max_triples
        self.dfactor = dfactor
        self.entropy_adjust = True

    def select(self, model:BayesianModel, X:np.ndarray, **kwargs)->int:
        raise NotImplementedError

    def sample_indices(self):
        ncombs = comb(self.n_samples, 3, exact=True)
        n_triples = min(ncombs, self.max_triples)
        unpacked_indices = np.random.choice(ncombs, size=n_triples, replace=False)
        idx1, idx2, idx3 = zip(*[combination_lookup(ind, self.n_samples, 3) for ind in unpacked_indices])
        return(idx1, idx2, idx3)

    def calc_log_triple_dists(self, W, idx1, idx2, idx3, entropies=None):
        ## Calculate distance matrix
        dists = self.dist.square_form(W)

        d12 = dists[idx1, idx2]
        d23 = dists[idx2, idx3]
        d13 = dists[idx1, idx3]
        if self.entropy_adjust:
            assert entropies is not None, "entropy adjustment selected, but no model entropies provided."
            exp_entropies = np.exp(2*entropies)
            triple_dists = exp_entropies[idx3,:]*d12[:,np.newaxis] + exp_entropies[idx2,:]*d13[:,np.newaxis] + exp_entropies[idx1,:]*d23[:,np.newaxis]
        else:
            triple_dists = d12 + d23 + d13
            triple_dists = triple_dists[:, np.newaxis]
        
        ## Handle 0 values
        with np.errstate(divide='ignore'):
            log_triple_dists = self.dfactor*np.log(triple_dists)
        
        return(log_triple_dists)

class DBALLinearRegression(DBALSelector):
    def __init__(self, n_samples: int, dist: Distance, max_triples: int = 10000, dfactor: float = 1, **kwargs):
        super().__init__(n_samples, dist, max_triples, dfactor, **kwargs)
        self.entropy_adjust = False ## Entropy adjustment does not factor in for Gaussian models with fixed variance

    def select(self, model:BayesLinearRegression, X:np.ndarray, **kwargs)->int:
        W = model.sample(self.n_samples) ## draw 300 samples
        mean_preds = np.einsum('ij, kj -> ik', W, X)
        n, m = mean_preds.shape

        ## Subsample indices
        idx1, idx2, idx3 = self.sample_indices()

        log_triple_dists = self.calc_log_triple_dists(W, idx1, idx2, idx3)

        ## Calculate score of every query
        d12 = np.square(mean_preds[idx1,:] - mean_preds[idx2,:])
        d13 = np.square(mean_preds[idx1,:] - mean_preds[idx3,:])
        d23 = np.square(mean_preds[idx2,:] - mean_preds[idx3,:])
        ll =  -(1./18.)*(d12 + d13 + d23) ## n_triples x m
        scores = logsumexp(ll + log_triple_dists, axis=0)

        ## Choose minimizer
        idx = np.argmin(scores)
        return(idx)


class DBALLogisticRegression(DBALSelector):
    def __init__(self, n_samples: int, dist: Distance, max_triples: int = 10000, dfactor: float = 1, **kwargs):
        super().__init__(n_samples, dist, max_triples, dfactor, **kwargs)
        self.entropy_adjust = True

    def select(self, model:BayesLogisticRegression, X:np.ndarray, **kwargs)->int:
        m, _ = X.shape
        W = model.sample(self.n_samples)

        pos_probs = expit(np.einsum('ik, jk -> ij', W, X) )

        probs = np.empty((self.n_samples, m, 2))
        probs[:,:,0] = pos_probs
        probs[:,:,1] = 1.0-pos_probs

        ## Subsample indices
        idx1, idx2, idx3 = self.sample_indices()

        ## Entropy adjustment
        entropies = None
        if self.entropy_adjust:
            entropies = entropy(probs, axis=2)

        log_triple_dists = self.calc_log_triple_dists(W, idx1, idx2, idx3, entropies=entropies)

        idx = dbal_finite_outcomes(probs=probs, log_triple_dists=log_triple_dists, idx1=idx1, idx2=idx2, idx3=idx3)
        return(idx)

###################
class DBALPoissonRegression(DBALSelector):
    def __init__(self, n_samples: int, dist: Distance, max_triples: int = 10000, dfactor: float = 1, **kwargs):
        super().__init__(n_samples, dist, max_triples, dfactor, **kwargs)
        self.entropy_adjust = True

    def select(self, model:BayesPoissonRegression, X:np.ndarray, **kwargs)->int:
        m, _ = X.shape
        W = model.sample(self.n_samples)

        lams = np.exp( np.einsum('ik, jk -> ij', W, X) )

        probs = poisson_probs(lams) ## n x m x O

        ## Subsample indices
        idx1, idx2, idx3 = self.sample_indices()

        ## Entropy adjustment
        entropies = None
        if self.entropy_adjust:
            entropies = entropy(probs, axis=2)

        log_triple_dists = self.calc_log_triple_dists(W, idx1, idx2, idx3, entropies=entropies)

        idx = dbal_finite_outcomes(probs=probs, log_triple_dists=log_triple_dists, idx1=idx1, idx2=idx2, idx3=idx3)
        return(idx)


###################
class DBALBetaRegression(DBALSelector):
    def __init__(self, n_samples: int, dist: Distance, max_triples: int = 10000, dfactor: float = 1, **kwargs):
        super().__init__(n_samples, dist, max_triples, dfactor, **kwargs)
        self.entropy_adjust = True

    def select(self, model:BayesBetaRegression, X:np.ndarray, **kwargs)->int:
        phi = model.phi
        n, _ = X.shape

        W = model.sample(self.n_samples)

        mu = np.clip(expit( np.einsum('ik, jk -> ij', W, X) ), a_min=0.001, a_max=0.999 ) 
        a = phi*mu
        b = phi*(1-mu)

        ## probs: n, m, K
        probs, x = beta_probs(a=a, b=b) 
        
        ## Subsample indices
        idx1, idx2, idx3 = self.sample_indices()

        ## Entropy adjustment
        entropies = None
        if self.entropy_adjust:
            entropies = beta_entropy(a, b) ## n x m

        log_triple_dists = self.calc_log_triple_dists(W, idx1, idx2, idx3, entropies=entropies)
        
        idx = dbal_continuous_outcomes(probs=probs, x=x, log_triple_dists=log_triple_dists, idx1=idx1, idx2=idx2, idx3=idx3)

        return(idx)



## probs: num samples x dim1 x dim2 x num outcomes
## log_triple_dists: len of (idx1, idx2, idx3) x num samples
def dbal_mf_finite_outcomes(probs:np.ndarray, log_triple_dists:np.ndarray, idx1:np.ndarray, idx2:np.ndarray, idx3:np.ndarray, index_pairs:list):
    log_triple_likelihood = np.log(np.sum(probs[idx1,:,:]*probs[idx2,:,:]*probs[idx3,:,:], axis=-1)) ## n_triples x dim1 x dim2
    n = len(idx1)
    m = len(index_pairs)
    log_triple_group_likelihood = np.empty((n, m))
    for i, (ii,jj) in enumerate(index_pairs):
        log_triple_group_likelihood[:, i] = np.sum( log_triple_likelihood[:, ii, jj] )

    scores = logsumexp(log_triple_group_likelihood + log_triple_dists, axis=0)
    return(np.argmin(scores))

class DBALMFSelector():
    __metaclass__ = abc.ABCMeta
    def __init__(self, n_samples:int, dist:MFDistance, max_triples:int=1000,  dfactor:float=1.0, **kwargs):
        self.n_samples = n_samples
        self.dist = dist
        self.max_triples = max_triples
        self.dfactor = dfactor
        self.entropy_adjust = True

    def select(self, model:BayesianMFModel, index_pairs:list, **kwargs)->int:
        raise NotImplementedError

    def sample_indices(self):
        ncombs = comb(self.n_samples, 3, exact=True)
        n_triples = min(ncombs, self.max_triples)
        unpacked_indices = np.random.choice(ncombs, size=n_triples, replace=False)
        idx1, idx2, idx3 = zip(*[combination_lookup(ind, self.n_samples, 3) for ind in unpacked_indices])
        return(idx1, idx2, idx3)

    def calc_log_triple_dists(self, W_list, V_list, prod_list, idx1, idx2, idx3, entropies=None):
        ## Calculate distance matrix
        dists = self.dist.square_form(W_list, V_list, prod_list)

        d12 = dists[idx1, idx2]
        d23 = dists[idx2, idx3]
        d13 = dists[idx1, idx3]
        if self.entropy_adjust:
            assert entropies is not None, "entropy adjustment selected, but no model entropies provided."
            exp_entropies = np.exp(2*entropies)
            triple_dists = exp_entropies[idx3,:]*d12[:,np.newaxis] + exp_entropies[idx2,:]*d13[:,np.newaxis] + exp_entropies[idx1,:]*d23[:,np.newaxis]
        else:
            triple_dists = d12 + d23 + d13
            triple_dists = triple_dists[:, np.newaxis]
        
        ## Handle 0 values
        with np.errstate(divide='ignore'):
            log_triple_dists = self.dfactor*np.log(triple_dists)
        
        return(log_triple_dists)



class DBALBernMF(DBALMFSelector):
    def __init__(self, n_samples:int, dist:MFDistance, max_triples:int=1000,  dfactor:float=1.0, **kwargs):
        super().__init__(n_samples, dist, max_triples, dfactor, **kwargs)

    
    def select(self, model: BayesianMFModel, index_pairs: list, **kwargs) -> int:
        W_list, V_list = model.sample(self.n_samples)
        prod_list = np.einsum('tik, tjk -> tij', W_list, V_list)
        
        _, dim1, dim2 = prod_list.shape

        probs = np.empty((self.n_samples, dim1, dim2, 2))
        probs[:,:,:,0] = expit(prod_list)
        probs[:,:,:,1] = 1.0-probs[:,:,:,0]

        ## Subsample indices
        idx1, idx2, idx3 = self.sample_indices()

        ## Entropy adjustment
        entropies = None
        if self.entropy_adjust:
            entropies = entropy(probs, axis=-1)

        log_triple_dists = self.calc_log_triple_dists(W_list, V_list, prod_list, idx1, idx2, idx3, entropies=entropies)

        idx = dbal_mf_finite_outcomes(probs=probs, log_triple_dists=log_triple_dists, idx1=idx1, idx2=idx2, idx3=idx3, index_pairs=index_pairs)
        return(idx)