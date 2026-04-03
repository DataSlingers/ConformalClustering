import sys
sys.path.append('../../')

import numpy as np  
from conformal_clustering.utils import ConformalClustering, GMMCutoff, sample_gmm

import pickle
import joblib
from joblib import Parallel, delayed


K = 5
d = 50

# generate means in 2D by equispaced points on a circle
means_2d = 4 * np.vstack([np.cos(np.pi * np.linspace(0, 2, K, endpoint=False)),
                      np.sin(np.pi * np.linspace(0, 2, K, endpoint=False))]).T

# embedding the 2D means into higher dimensions by padding with zeros
means = np.zeros((K, d))
means[:, :2] = means_2d

base_covariances = np.array([np.eye(d) for _ in range(K)])
weights = np.ones(K) / K



def run_experiment(n, var, seed):
    covariances = base_covariances * var
    n_al = np.min([100, n//10]) # 10% or 100 points for alignment
    n_total = n + n_al + 1 # total points needed
    X, y = sample_gmm(means, covariances, weights, n_samples=n_total, seed=seed)

    X_tr, y_tr = X[:(n//2)], y[:(n//2)] # training
    X_cal, y_cal = X[(n//2):n], y[(n//2):n] # calibration
    X_al, y_al = X[n:(n+n_al)], y[n:(n+n_al)] # alignment
    X_val, y_val = X[(n+n_al):(n+n_al+1)], y[(n+n_al):(n+n_al+1)] # validation

    cc = ConformalClustering(X_tr, X_cal)
    cc.set_classifier('SVC', random_state=0, probability=True)
    # repeat clustering until K clusters are found (change seed)
    found_K = False
    attempt = 0
    while not found_K:
        try:
            cc.fit('GMM', n_components=K, covariance_type='diag', max_iter=200, n_init=5, random_state=attempt)
            found_K = True
        except ValueError:
            attempt += 1
    res_GMM = cc.validate(X_val, y_val, X_al, y_al, method='narc')

    cc = ConformalClustering(X_tr, X_cal)
    cc.set_classifier('SVC', random_state=0, probability=True)
    found_K = False
    while not found_K:
        try:
            cc.fit('GMMS', n_components=K, covariance_type='diag', max_iter=200, n_init=5, random_state=attempt)
            found_K = True
        except ValueError:
            attempt += 1
    res_GMMS = cc.validate(X_val, y_val, X_al, y_al, method='narc')
    
    gc = GMMCutoff(X)
    found_K = False
    attempt = 0
    while not found_K:
        try:
            gc.fit(n_components=K, covariance_type='diag', max_iter=200, n_init=5, random_state=attempt)
            found_K = True
        except ValueError:
            attempt += 1
    res_GMMC = gc.validate(X_val, y_val, X_al, y_al)

    results = {
        'n': n,
        'var': var,
        'seed': seed,
        'GMM': res_GMM,
        'GMMS': res_GMMS,
        'GMM-Cutoff': res_GMMC
    }


    return results


if __name__ == '__main__':
    var = 2.8
    B = 4000
    np.random.seed(0)
    seeds = np.random.permutation(10*B)[:B]
    n_range = [300, 500, 800, 1200, 2000, 3200, 5000, 8000]

    print(f'Available CPUs by joblib: {joblib.cpu_count()}')
    results = Parallel(n_jobs=-1, verbose=1)(
            delayed(run_experiment)(n, var, seed)
            for n in n_range
            for seed in seeds
        )

    with open(f'GMM50D5v1_asymp.pkl', 'wb') as f:
        pickle.dump(results, f)
