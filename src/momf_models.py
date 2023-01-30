import numpy as np
import abc
from dists import MOMFDistance
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from fast_mvn import sample_mvn_from_precision
from tqdm import trange

class NormMixtureMFModel(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, n_rows:int, n_cols:int, n_features:int, K:int, sigma_obs:float, sigma_emb:float, sigma_norm:float, thin:int=10, burnin:int=100, **kwargs):
        self.ii = np.array([], dtype=np.int32)
        self.jj = np.array([], dtype=np.int32)
        self.y = np.array([], dtype=np.float32)

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_features = n_features
        self.K = K
        self.sigma_obs = sigma_obs
        self.sigma_emb = sigma_emb
        self.sigma_norm = sigma_norm

        self.var_0 = np.square(sigma_norm)
        self.var_emb = np.square(self.sigma_emb)
        self.var_obs = np.square(self.sigma_obs)

        self.burnin = burnin
        self.thin = thin

        self.initialize()
    
    def initialize(self):
        self.V = np.random.standard_normal(size=(self.n_cols, self.n_features))
        self.W = np.random.standard_normal(size=(self.n_rows, self.n_features))
        self.mu = np.random.standard_normal(size=(self.K, self.n_features))
        self.z = np.random.choice(self.K, size=self.n_rows, replace=True)

    def z_step(self):
        log_probs = -0.5*cdist(self.W, self.mu, metric='sqeuclidean')/np.square(self.sigma_emb) ## n_rows x K
        probs = np.exp(log_probs - logsumexp(log_probs, axis=1, keepdims=True))
        for i in range(self.n_rows):
            self.z[i] = np.random.choice(self.K, p=probs[i,:])

    def mu_step(self):
        for k in range(self.K):
            mask = self.z == k
            n = np.sum(mask)
            if n == 0:
                self.mu[k,:] = self.sigma_norm * np.random.standard_normal(size=self.n_features)
            else:
                mu_k = np.mean(self.W[mask,:], axis=0) * n * self.var_0/( self.var_emb + n * self.var_0)
                sigma_k = np.sqrt(self.var_0 * self.var_emb /(self.var_emb + n * self.var_0))

                self.mu[k,:] = mu_k + sigma_k * np.random.standard_normal(size=self.n_features)

    def V_step(self):
        for j in range(self.n_cols):
            if j not in self.col_lookup:
                self.V[j,:] = self.sigma_norm * np.random.standard_normal(size=self.n_features)
            else:
                idx = self.col_lookup[j]
                X = self.W[self.ii[idx],:]
                y = self.y[idx]
                P_n = (1./self.var_0)*np.eye(self.n_features) + (1./self.var_obs)*np.dot(X.T, X)
                mu_part = (1./self.var_obs)*np.dot(X.T, y)
                
                self.V[j,:] = sample_mvn_from_precision(Q=P_n, mu_part=mu_part)
    
    def W_step(self):
        for i in range(self.n_rows):
            mu = self.mu[self.z[i],:]

            if i not in self.row_lookup:
                self.W[i,:] = mu + self.sigma_emb * np.random.standard_normal(size=self.n_features)
            else:
                idx = self.row_lookup[i]
                X = self.V[self.jj[idx],:]
                y = self.y[idx]
                P_n = (1./self.var_emb)*np.eye(self.n_features) + (1./self.var_obs)*np.dot(X.T, X)
                mu_part = (mu/self.var_emb) + (1./self.var_obs)*np.dot(X.T, y)
                
                self.W[i,:] = sample_mvn_from_precision(Q=P_n, mu_part=mu_part)
    
    def sample(self, n:int):
        self.initialize()
        
        W = np.empty((n, self.n_rows, self.n_features))
        V = np.empty((n, self.n_cols, self.n_features))
        z = np.empty((n, self.n_rows), dtype=np.int32)

        total_steps = self.burnin + n * self.thin
        i = 0
        for t in trange(total_steps):
            self.V_step()
            self.W_step()
            self.mu_step()
            self.z_step()

            if (t >= self.burnin) and ((t - self.burnin)%self.thin == 0):
                W[i,:,:] = self.W.copy()
                V[i,:,:] = self.V.copy()
                z[i,:] = self.z.copy()
                i += 1
        assert i==n, "Did not calculate samples correctly!"
        return(W, V, z)

    def update(self, ii_new:np.ndarray, jj_new:np.ndarray, y_new:np.ndarray, **kwargs):
        self.ii = np.append(self.ii, ii_new)
        self.jj = np.append(self.jj, jj_new)
        self.y = np.append(self.y, y_new)

        ## Preprocessing
        self.row_lookup = {}
        self.col_lookup = {}

        for idx, (i, j) in enumerate(zip(self.ii, self.jj)):
            if i in self.row_lookup:
                self.row_lookup[i].append(idx)
            else:
                self.row_lookup[i] = [idx]
            
            if j in self.col_lookup:
                self.col_lookup[j].append(idx)
            else:
                self.col_lookup[j] = [idx]
        
        for i, c in self.row_lookup.items():
            self.row_lookup[i] = np.array(c, dtype=np.int32)

        for j, c in self.col_lookup.items():
            self.col_lookup[j] = np.array(c, dtype=np.int32)

    def eval(self, n:int, W:np.ndarray, V:np.ndarray, z:np.ndarray, distance:MOMFDistance, **kwargs):
        W_list, V_list, z_list = self.sample(n) ## n x r x dim, n x c x dim
        avg_dist = distance.average_distance(W_list, V_list, z_list, W, V, z)
        return(avg_dist)

