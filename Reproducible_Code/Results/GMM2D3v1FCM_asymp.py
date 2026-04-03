import sys
sys.path.append('../../')

import numpy as np  
from conformal_clustering.utils import ConformalClustering, GMMCutoff, sample_gmm

import pickle
import joblib
from joblib import Parallel, delayed


K = 3
means = np.array([[1, 1], [3, 4], [4, 1]])
base_covariances = np.array([np.eye(2) for _ in range(K)])
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
            cc.fit('FCMS', c=K, m=1.4, error=0.001, maxiter=1000, seed=attempt)
            found_K = True
        except ValueError:
            attempt += 1
    res_FCMS1 = cc.validate(X_val, y_val, X_al, y_al, method='narc')

    cc = ConformalClustering(X_tr, X_cal)
    cc.set_classifier('SVC', random_state=0, probability=True)
    found_K = False
    attempt = 0
    while not found_K:
        try:
            cc.fit('FCMS', c=K, m=1.7, error=0.001, maxiter=1000, seed=attempt)
            found_K = True
        except ValueError:
            attempt += 1
    res_FCMS2 = cc.validate(X_val, y_val, X_al, y_al, method='narc')

    cc = ConformalClustering(X_tr, X_cal)
    cc.set_classifier('SVC', random_state=0, probability=True)
    found_K = False
    attempt = 0
    while not found_K:
        try:
            cc.fit('FCMS', c=K, m=2, error=0.001, maxiter=1000, seed=attempt)
            found_K = True
        except ValueError:
            attempt += 1
    res_FCMS3 = cc.validate(X_val, y_val, X_al, y_al, method='narc')

    results = {
        'n': n,
        'var': var,
        'seed': seed,
        'FCMS1': res_FCMS1,
        'FCMS2': res_FCMS2,
        'FCMS3': res_FCMS3
    }


    return results


if __name__ == '__main__':
    var = 1.5
    B = 4000
    np.random.seed(0)
    seeds = np.random.permutation(10*B)[:B]
    n_range = [30, 60, 100, 200, 400, 1000, 2000, 4000]

    print(f'Available CPUs by joblib: {joblib.cpu_count()}')
    results = Parallel(n_jobs=-1, verbose=1)(
            delayed(run_experiment)(n, var, seed)
            for n in n_range
            for seed in seeds
        )

    with open(f'GMM2D3v1FCM_asymp.pkl', 'wb') as f:
        pickle.dump(results, f)
