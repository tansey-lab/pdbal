import numpy as np
import abc
from dists import Distance

class ParamBatch(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, **kwargs):
        return

class BayesianModel(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, **kwargs):
        self.X = []
        self.y = []

    def sample(self, n:int)->ParamBatch:
        raise NotImplementedError

    def update(self, x:np.ndarray, y:float, **kwargs):
        self.X.append(x)
        self.y.append(y)

    def eval(self, n:int, w_star:np.ndarray, distance:Distance, **kwargs):
        W = self.sample(n)

        avg_dist = distance.average_distance(W, w_star)

        return(avg_dist)
