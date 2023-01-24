import numpy as np
from tqdm import trange
from scipy.linalg import lstsq

def alternating_minimization(ii, jj, y, n_features=5, n_epochs=100):
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
        col_lookup[i] = np.array(c)

    W = np.random.standard_normal(shape=(n_rows, n_features))
    V = np.random.standard_normal(shape=(n_cols, n_features))

    for _ in trange(n_epochs):
        for i, idx in row_lookup.items():
            W[i,:] = lstsq(V[idx,:], y[idx])

        for j,idx in col_lookup.items():
            V[j,:] = lstsq(W[idx,:], y[idx])
    return(W, V)