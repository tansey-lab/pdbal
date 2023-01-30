import abc
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform, euclidean
from scipy.stats import kendalltau
from sklearn.metrics.cluster import normalized_mutual_info_score

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


class MFDistance():
    __metaclass__ = abc.ABCMeta
    def __init__(self, **kwargs):
        return
    
    def distance(self, W1:np.ndarray, V1:np.ndarray, prod1:np.ndarray,  W2:np.ndarray, V2:np.ndarray, prod2:np.ndarray):
        raise NotImplementedError

    def square_form(self, W_list:np.ndarray, V_list:np.ndarray, prod_list:np.ndarray):
        N = W_list.shape[0]
        M = np.zeros((N,N))
        for i in range(1,N):
            for j in range(i):
                M[i,j] = self.distance(W_list[i], V_list[i], prod_list[i], W_list[j], V_list[j], prod_list[j])
                M[j,i] = M[i,j]
        return(M)

    def average_distance(self, W_list:np.ndarray, V_list:np.ndarray, prod_list:np.ndarray,  W:np.ndarray, V:np.ndarray, prod:np.ndarray):
        N = W_list.shape[0]
        avg_dist = np.mean( [self.distance(W_list[i], V_list[i], prod_list[i], W,V,prod) for i in range(N)  ])
        return(avg_dist)


class MFMSEDistance(MFDistance):
    def __init__(self):
        super().__init__()

    def distance(self, W1:np.ndarray, V1:np.ndarray, prod1:np.ndarray,  W2:np.ndarray, V2:np.ndarray, prod2:np.ndarray):
        dist = np.mean(np.square(prod1 - prod2))
        return(dist)

class MFRowMSEDistance(MFDistance):
    def __init__(self, row:int):
        super().__init__()
        self.row = row

    def distance(self, W1:np.ndarray, V1:np.ndarray, prod1:np.ndarray,  W2:np.ndarray, V2:np.ndarray, prod2:np.ndarray):
        dist = np.mean(np.square(prod1[self.row,:] - prod2[self.row,:]))
        return(dist)


class MFMaxCoordDistance(MFDistance):
    def __init__(self):
        super().__init__()

    def distance(self, W1:np.ndarray, V1:np.ndarray, prod1:np.ndarray,  W2:np.ndarray, V2:np.ndarray, prod2:np.ndarray):
        idx1 = np.argmax(prod1)
        idx2 = np.argmax(prod2)
        dist = float(idx1 != idx2)
        return(dist)

class MFMKendallDistance(MFDistance):
    def __init__(self):
        super().__init__()

    def distance(self, W1:np.ndarray, V1:np.ndarray, prod1:np.ndarray,  W2:np.ndarray, V2:np.ndarray, prod2:np.ndarray):
        rho, _ = kendalltau( prod1, prod2 )
        dist = 0.5*(1.0 - rho)
        return(dist)

class MFMRegretPerRowDistance(MFDistance):
    def __init__(self):
        super().__init__()

    def distance(self, W1:np.ndarray, V1:np.ndarray, prod1:np.ndarray,  W2:np.ndarray, V2:np.ndarray, prod2:np.ndarray):
        n,_ = prod1.shape
        idx1 = np.argmin(prod1, axis=1)
        idx2 = np.argmin(prod2, axis=1)

        onevtwo = prod1[np.arange(n), idx2] - prod1[np.arange(n), idx1] ## regret of choosing 2 over 1 (in 1's world)
        twovone = prod2[np.arange(n), idx1] - prod2[np.arange(n), idx2] ## regret of choosing 1 over 2 (in 2's world)

        dist = np.mean(0.5*onevtwo + 0.5*twovone)
        return(dist)



class MOMFDistance():
    __metaclass__ = abc.ABCMeta
    def __init__(self, **kwargs):
        return
    
    def distance(self, W1:np.ndarray, V1:np.ndarray, z1:np.ndarray,  W2:np.ndarray, V2:np.ndarray, z2:np.ndarray):
        raise NotImplementedError

    def square_form(self, W_list:np.ndarray, V_list:np.ndarray, z_list:np.ndarray):
        N = W_list.shape[0]
        M = np.zeros((N,N))
        for i in range(1,N):
            for j in range(i):
                M[i,j] = self.distance(W_list[i], V_list[i], z_list[i], W_list[j], V_list[j], z_list[j])
                M[j,i] = M[i,j]
        return(M)

    def average_distance(self, W_list:np.ndarray, V_list:np.ndarray, z_list:np.ndarray,  W:np.ndarray, V:np.ndarray, z:np.ndarray):
        N = W_list.shape[0]
        avg_dist = np.mean( [self.distance(W_list[i], V_list[i], z_list[i], W, V, z) for i in range(N)])
        return(avg_dist)


class MOMFClusterDistance(MOMFDistance):
    def __init__(self):
        super().__init__()

    def distance(self, W1: np.ndarray, V1: np.ndarray, z1: np.ndarray, W2: np.ndarray, V2: np.ndarray, z2: np.ndarray):
        dist = 1.0 - normalized_mutual_info_score(z1, z2)
        return(dist)