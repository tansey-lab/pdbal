import abc
import numpy as np
from momf_models import NormMixtureMFModel


class VarNormMOMF():
    def __init__(self, n_samples:int, **kwargs):
        self.n_samples = n_samples

    def select(self, model:NormMixtureMFModel, **kwargs)->int:
        W_list, V_list, z_list = model.sample(self.n_samples)

        conditional_means = np.einsum('tik, tjk -> tij', W_list, V_list) ## n x dim1 x dim2
        
        variances = np.var(conditional_means, axis=0)

        i, j = np.unravel_index(np.argmax(variances, axis=None), variances.shape)
        return(i,j)