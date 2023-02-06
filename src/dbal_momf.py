import abc
import numpy as np
from scipy.special import logsumexp, comb, expit
from scipy.stats import entropy
from scipy.integrate import simpson
from dists import MOMFDistance
from momf_models import NormMixtureMFModel

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




class DBALNormMOMF():
    def __init__(self, n_samples:int, dist:MOMFDistance, max_triples:int=1000,  dfactor:float=1.0, **kwargs):
        self.n_samples = n_samples
        self.dist = dist
        self.max_triples = max_triples
        self.dfactor = dfactor


    def sample_indices(self):
        ncombs = comb(self.n_samples, 3, exact=True)
        n_triples = min(ncombs, self.max_triples)
        unpacked_indices = np.random.choice(ncombs, size=n_triples, replace=False)
        idx1, idx2, idx3 = zip(*[combination_lookup(ind, self.n_samples, 3) for ind in unpacked_indices])
        return(idx1, idx2, idx3)


    def calc_log_triple_dists(self, W_list, V_list, z_list, idx1, idx2, idx3):
        ## Calculate distance matrix
        dists = self.dist.square_form(W_list, V_list, z_list)

        d12 = dists[idx1, idx2]
        d23 = dists[idx2, idx3]
        d13 = dists[idx1, idx3]

        triple_dists = d12 + d23 + d13
        triple_dists = triple_dists[:,np.newaxis,np.newaxis]
        
        ## Handle 0 values
        with np.errstate(divide='ignore'):
            log_triple_dists = self.dfactor*np.log(triple_dists)
        
        return(log_triple_dists)

    def select(self, model: NormMixtureMFModel, **kwargs) -> int:
        W_list, V_list, z_list = model.sample(self.n_samples)
        prod_list = np.einsum('tik, tjk -> tij', W_list, V_list)
        
        ## Subsample indices
        idx1, idx2, idx3 = self.sample_indices()

        log_triple_dists = self.calc_log_triple_dists(W_list, V_list, z_list, idx1, idx2, idx3)

        ## Calculate score of every query
        d12 = np.square(prod_list[idx1,:,:] - prod_list[idx2,:,:])
        d13 = np.square(prod_list[idx1,:,:] - prod_list[idx3,:,:])
        d23 = np.square(prod_list[idx2,:,:] - prod_list[idx3,:,:])
        ll =  -(1./18.)*(d12 + d13 + d23) ## n_triples x dim1 x dim2

        scores = logsumexp(ll + log_triple_dists, axis=0)

        ## Choose minimizer
        # i, j = np.unravel_index(np.argmin(scores, axis=None), scores.shape)
        # ii, jj =  np.where(scores <= np.partition(scores.flatten(), 2)[2])
        ii, jj = np.where(scores <= np.min(scores))

        nmins = len(ii)
        print("Number of minimizers:", nmins)
        if nmins > 1:
            print("scores:", scores)
            print("ll:", ll)
            print("log_triple_dists:", log_triple_dists)

        index = np.random.choice(nmins)
        i = ii[index]
        j = jj[index]
        return(i,j)