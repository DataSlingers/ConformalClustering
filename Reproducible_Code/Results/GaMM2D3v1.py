import sys
sys.path.append('../../')

import numpy as np  
from conformal_clustering.utils import ConformalClustering, GaMMCutoff, sample_gamm

import pickle
import joblib
from joblib import Parallel, delayed


K = 3
means = np.array([[4, 9], [9, 4], [2, 2]])

base_covariances = np.array([np.eye(2) for _ in range(K)])
base_shapes = np.zeros((K, 2))
base_scales = np.zeros((K, 2))

for k in range(K):
    base_scales[k] = base_covariances[k].diagonal() / means[k]
    base_shapes[k] = means[k] / base_scales[k]

weights = np.ones(K) / K



def run_experiment(n, var, seed):
    shapes = base_shapes / var
    scales = base_scales * var
    n_al = np.min([100, n//10]) # 10% or 100 points for alignment
    n_total = n + n_al + 1 # total points needed
    X, y = sample_gamm(shapes, scales, weights, n_samples=n_total, seed=seed)

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
            cc.fit('GaMM', n_components=K, max_iter=1000, n_init=25, random_state=attempt)
            found_K = True
        except ValueError:
            attempt += 1
    res_GaMM = cc.validate(X_val, y_val, X_al, y_al, method='narc')

    cc = ConformalClustering(X_tr, X_cal)
    cc.set_classifier('SVC', random_state=0, probability=True)
    found_K = False
    while not found_K:
        try:
            cc.fit('GaMMS', n_components=K, max_iter=1000, n_init=25, random_state=attempt)
            found_K = True
        except ValueError:
            attempt += 1
    res_GaMMS = cc.validate(X_val, y_val, X_al, y_al, method='narc')
    
    gc = GaMMCutoff(X)
    found_K = False
    attempt = 0
    while not found_K:
        try:
            gc.fit(n_components=K, max_iter=1000, n_init=25, random_state=attempt)
            found_K = True
        except ValueError:
            attempt += 1
    res_GaMMC = gc.validate(X_val, y_val, X_al, y_al)

    results = {
        'n': n,
        'var': var,
        'seed': seed,
        'GaMM': res_GaMM,
        'GaMMS': res_GaMMS,
        'GaMM-Cutoff': res_GaMMC
    }


    return results


if __name__ == '__main__':
    n = 1000
    B = 4000
    np.random.seed(0)
    seeds = np.random.permutation(10*B)[:B]
    var_range = np.linspace(2, 11, 10)

    print(f'Available CPUs by joblib: {joblib.cpu_count()}')
    results = Parallel(n_jobs=-1, verbose=1)(
            delayed(run_experiment)(n, var, seed)
            for var in var_range
            for seed in seeds
        )

    with open(f'GaMM2D3v1.pkl', 'wb') as f:
        pickle.dump(results, f)
