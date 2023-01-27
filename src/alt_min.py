import numpy as np
from tqdm import trange
from scipy.linalg import lstsq
from sklearn.linear_model import Ridge, RidgeCV

def alternating_minimization(ii, jj, y, n_features=5, n_epochs=100):
    np.random.seed(100)
    n_rows = np.max(ii)+1
    n_cols = np.max(jj)+1

    ## Preprocessing
    row_lookup = {}
    col_lookup = {}

    for idx, (i, j) in enumerate(zip(ii, jj)):
        if i in row_lookup:
            row_lookup[i].append(idx)
        else:
            row_lookup[i] = [idx]
        
        if j in col_lookup:
            col_lookup[j].append(idx)
        else:
            col_lookup[j] = [idx]
    
    for i, c in row_lookup.items():
        row_lookup[i] = np.array(c)

    for j, c in col_lookup.items():
        col_lookup[j] = np.array(c)

    W = np.random.standard_normal(size=(n_rows, n_features))
    V = np.random.standard_normal(size=(n_cols, n_features))

    clf = Ridge(fit_intercept=False)
    for _ in trange(n_epochs):
        for i, idx in row_lookup.items():
            # W[i,:], *_ = lstsq(V[jj[idx],:], y[idx])
            clf.fit(V[jj[idx],:], y[idx])
            W[i,:] = clf.coef_

        for j,idx in col_lookup.items():
            # V[j,:], *_ = lstsq(W[ii[idx],:], y[idx])
            clf.fit(W[ii[idx],:], y[idx])
            V[j,:] = clf.coef_

    return(W, V)