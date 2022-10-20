import abc
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform, euclidean
from scipy.stats import spearmanr, kendalltau


class Distance():
    __metaclass__ = abc.ABCMeta
    def __init__(self, **kwargs):
        return
    
    def distance(self, u:np.ndarray, v:np.ndarray):
        raise NotImplementedError

    def __call__(self, u:np.ndarray, v:np.ndarray):
        return(self.distance(u=u, v=v))
    
    def square_form(self, X:np.ndarray):
        return(squareform(pdist(X, metric=self.distance)))

    def average_distance(self, X:np.ndarray, x:np.ndarray):
        avg_dist = np.mean( cdist(X,x[np.newaxis, :], metric=self.distance) )
        return(avg_dist)

class CoordSignDistance(Distance):
    def __init__(self, idx:int=0):
        super().__init__()
        self.idx = idx

    def distance(self, u:np.ndarray, v:np.ndarray):
        return( float(u[self.idx]*v[self.idx] <= 0) )

class MaxCoordDistance(Distance):
    def __init__(self):
        super().__init__()

    def distance(self, u:np.ndarray, v:np.ndarray):
        return( float(np.argmax(np.abs(u)) != np.argmax(np.abs(v))) )

class EuclideanDistance(Distance):
    def __init__(self):
        super().__init__()

    def distance(self, u:np.ndarray, v:np.ndarray):
        return(euclidean(u, v))


class KendallDistance(Distance):
    def __init__(self):
        super().__init__()

    def distance(self, u:np.ndarray, v:np.ndarray):
        rho, _ = kendalltau( np.abs(u), np.abs(v) )
        dist = 0.5*(1.0 - rho)
        return(dist)


def sign_distance( u:np.ndarray, v:np.ndarray):
    return(np.mean( np.sign(u) !=  np.sign(v) ))


class CoordInfluenceDistance(Distance):
    def __init__(self, X_data:np.ndarray, coords:np.ndarray):
        super().__init__()
        self.X_coords = X_data[:, coords]
        self.coords = coords

    def distance(self, u:np.ndarray, v:np.ndarray):
        u_ = np.dot(self.X_coords, u[self.coords])
        v_ = np.dot(self.X_coords, v[self.coords])
        return(sign_distance(u_, v_))

    def square_form(self, X:np.ndarray):
        V = np.einsum('ik, jk -> ij', X[:,self.coords], self.X_coords) ## n samples x m data points
        return(squareform(pdist(V, metric=sign_distance)))

    def average_distance(self, X:np.ndarray, x:np.ndarray):
        V = np.einsum('ik, jk -> ij', X[:,self.coords], self.X_coords) ## n samples x m data points
        v = np.dot(self.X_coords, x[self.coords])
        avg_dist = np.mean( cdist(V,v[np.newaxis, :], metric=self.distance))
        return(avg_dist)