# Import clustering models
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
import skfuzzy as fuzz

# Import classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Import other utilities
import warnings
import numpy as np
from scipy.sparse import csr_array, csr_matrix
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.special import gammaln, psi, polygamma, logsumexp
from sklearn.neighbors import kneighbors_graph
from sklearn.exceptions import ConvergenceWarning


def conformal_arc(cal_prob, Y_cal, test_prob, alpha=0.1):
    """
    Based on Romano et al. (2020) for classification - adaptive and reliable classification (ARC)
        cal_prob: calibration probabilities (n_cal, K)
        Y_cal: calibration labels (n_cal,)
        test_prob: test probabilities (n_test, K)
    """
    n_cal, K = cal_prob.shape
    n_test = test_prob.shape[0]

    # sort the calibration probabilities
    sorted_cal_prob_ids = np.argsort(-cal_prob, axis=1) # descending order (r inverse in my notes)
    sorted_cal_probs = np.take_along_axis(cal_prob, sorted_cal_prob_ids, axis=1) # sorted probabilities
    sorted_cal_probs_cumsum = np.cumsum(sorted_cal_probs, axis=1) # cumulative sum of sorted probabilities
    
    # ranking (map r() in my notes)
    cal_prob_ranks = np.argsort(sorted_cal_prob_ids, axis=1) # rank of each class in each calibration sample

    # compute the ranks corresponding to the labels in the calibration set
    rank_Y_cal = cal_prob_ranks[np.arange(n_cal), Y_cal] # rank of the true class in each calibration sample

    # randomization
    U_cal = np.random.random_sample(n_cal)  # uniform random numbers for each calibration sample
    
    # scores (E in the paper)
    cal_scores = sorted_cal_probs_cumsum[np.arange(n_cal), rank_Y_cal] - sorted_cal_probs[np.arange(n_cal), rank_Y_cal]*U_cal

    # get quantiles
    sorted_cal_scores = np.sort(cal_scores)
    q_rank = int(np.ceil((n_cal + 1) * (1 - alpha)))
    q = sorted_cal_scores[q_rank-1] # q_rank-th smallest value among the scores

    # test scores
    sorted_test_prob_ids = np.argsort(-test_prob, axis=1) # descending order
    sorted_test_probs = np.take_along_axis(test_prob, sorted_test_prob_ids, axis=1) # sorted probabilities
    sorted_test_probs_cumsum = np.cumsum(sorted_test_probs, axis=1) # cumulative sum of sorted probabilities

    # ranking 
    test_prob_ranks = np.argsort(sorted_test_prob_ids, axis=1) # rank of each class in each test sample

    # randomization
    U_test = np.random.random_sample(n_test)  # uniform random numbers for each test sample
    test_scores = sorted_test_probs_cumsum - sorted_test_probs*U_test[:, np.newaxis] # each row is ascending

    # compute the conformal set
    prediction_set_rank = test_scores <= q # based on the ranks
    prediction_set = np.take_along_axis(prediction_set_rank, test_prob_ranks, axis=1)   # rank to the original indices

    return prediction_set   # (n_test, K) with each row = prediction set as a boolean vector


def conformal_arc_ne(cal_prob, Y_cal, test_prob, alpha=0.1):
    """
    conformal_arc with non-empty guarantee (include the most probable label if empty)
        cal_prob: calibration probabilities (n_cal, K)
        Y_cal: calibration labels (n_cal,)
        test_prob: test probabilities (n_test, K)
    """
    n_cal, K = cal_prob.shape
    n_test = test_prob.shape[0]

    # sort the calibration probabilities
    sorted_cal_prob_ids = np.argsort(-cal_prob, axis=1) # descending order (r inverse in my notes)
    sorted_cal_probs = np.take_along_axis(cal_prob, sorted_cal_prob_ids, axis=1) # sorted probabilities
    sorted_cal_probs_cumsum = np.cumsum(sorted_cal_probs, axis=1) # cumulative sum of sorted probabilities

    # ranking (map r() in my notes)
    cal_prob_ranks = np.argsort(sorted_cal_prob_ids, axis=1) # rank of each class in each calibration sample

    # compute the ranks corresponding to the labels in the calibration set
    rank_Y_cal = cal_prob_ranks[np.arange(n_cal), Y_cal] # rank of the true class in each calibration sample

    # randomization
    U_cal = np.random.random_sample(n_cal)  # uniform random numbers for each calibration sample

    # scores (E in the paper)
    cal_scores = sorted_cal_probs_cumsum[np.arange(n_cal), rank_Y_cal] - sorted_cal_probs[np.arange(n_cal), rank_Y_cal]*U_cal

    # get quantiles
    sorted_cal_scores = np.sort(cal_scores)
    q_rank = int(np.ceil((n_cal + 1) * (1 - alpha)))
    q = sorted_cal_scores[q_rank-1] # q_rank-th smallest value among the scores

    # test scores
    sorted_test_prob_ids = np.argsort(-test_prob, axis=1) # descending order
    sorted_test_probs = np.take_along_axis(test_prob, sorted_test_prob_ids, axis=1) # sorted probabilities
    sorted_test_probs_cumsum = np.cumsum(sorted_test_probs, axis=1) # cumulative sum of sorted probabilities

    # ranking
    test_prob_ranks = np.argsort(sorted_test_prob_ids, axis=1) # rank of each class in each test sample

    # randomization
    U_test = np.random.random_sample(n_test)  # uniform random numbers for each test sample
    test_scores = sorted_test_probs_cumsum - sorted_test_probs*U_test[:, np.newaxis] # each row is ascending

    # compute the conformal set
    prediction_set_rank = test_scores <= q # based on the ranks
    prediction_set = np.take_along_axis(prediction_set_rank, test_prob_ranks, axis=1)   # rank to the original indices

    # ensure non-empty sets (include the most probable label if empty)
    most_probable_labels = np.argmax(test_prob, axis=1)
    prediction_set[np.arange(n_test), most_probable_labels] = True
    
    return prediction_set   # (n_test, K) with each row = prediction set as a boolean vector


def conformal_narc(cal_prob, Y_cal, test_prob, alpha=0.1):
    """
    conformal_arc without randomization (most probable label always included)
        cal_prob: calibration probabilities (n_cal, K)
        Y_cal: calibration labels (n_cal,)
        test_prob: test probabilities (n_test, K)
    """
    n_cal, K = cal_prob.shape
    n_test = test_prob.shape[0]

    # sort the calibration probabilities
    sorted_cal_prob_ids = np.argsort(-cal_prob, axis=1) # descending order (r inverse in my notes)
    sorted_cal_probs = np.take_along_axis(cal_prob, sorted_cal_prob_ids, axis=1) # sorted probabilities
    sorted_cal_probs_cumsum = np.cumsum(sorted_cal_probs, axis=1) # cumulative sum of sorted probabilities
    
    # ranking (map r() in my notes)
    cal_prob_ranks = np.argsort(sorted_cal_prob_ids, axis=1) # rank of each class in each calibration sample

    # compute the ranks corresponding to the labels in the calibration set
    rank_Y_cal = cal_prob_ranks[np.arange(n_cal), Y_cal] # rank of the true class in each calibration sample

    # scores (E in the paper)
    cal_scores = sorted_cal_probs_cumsum[np.arange(n_cal), rank_Y_cal] - sorted_cal_probs[np.arange(n_cal), rank_Y_cal]

    # get quantiles
    sorted_cal_scores = np.sort(cal_scores)
    q_rank = int(np.ceil((n_cal + 1) * (1 - alpha)))
    q = sorted_cal_scores[q_rank-1] # q_rank-th smallest value among the scores

    # test scores
    sorted_test_prob_ids = np.argsort(-test_prob, axis=1) # descending order
    sorted_test_probs = np.take_along_axis(test_prob, sorted_test_prob_ids, axis=1) # sorted probabilities
    sorted_test_probs_cumsum = np.cumsum(sorted_test_probs, axis=1) # cumulative sum of sorted probabilities

    # ranking 
    test_prob_ranks = np.argsort(sorted_test_prob_ids, axis=1) # rank of each class in each test sample
    test_scores = sorted_test_probs_cumsum - sorted_test_probs # each row is ascending

    # compute the conformal set
    prediction_set_rank = test_scores <= q # based on the ranks
    prediction_set = np.take_along_axis(prediction_set_rank, test_prob_ranks, axis=1)   # rank to the original indices

    return prediction_set   # (n_test, K) with each row = prediction set as a boolean vector


def conformal_clp(cal_prob, Y_cal, test_prob, alpha=0.1):
    """
    Compute the conformal set using the classification probability
        cal_prob: calibration probabilities (n_cal, K)
        Y_cal: calibration labels (n_cal,)
        test_prob: test probabilities (n_test, K)
    """
    n_cal, K = cal_prob.shape
    n_test = test_prob.shape[0]

    # compute the classification scores = pi_hat(Y_i | X_i) for i in cal
    # conformal scores = negative probabilities
    cal_scores = -cal_prob[np.arange(n_cal), Y_cal]

    # get quantiles
    sorted_cal_scores = np.sort(cal_scores)
    q_rank = int(np.ceil((n_cal + 1) * (1 - alpha)))
    q = sorted_cal_scores[q_rank-1] # q_rank-th smallest value among the scores

    # compute the conformal set
    test_scores = -test_prob
    prediction_set = test_scores <= q

    return prediction_set   # (n_test, K) with each row = prediction set as a boolean vector


def conformal_clp_ne(cal_prob, Y_cal, test_prob, alpha=0.1):
    """
    conformal_clp with non-empty guarantee (include the most probable label if empty)
        cal_prob: calibration probabilities (n_cal, K)
        Y_cal: calibration labels (n_cal,)
        test_prob: test probabilities (n_test, K)
    """
    n_cal, K = cal_prob.shape
    n_test = test_prob.shape[0]

    # compute the classification scores = pi_hat(Y_i | X_i) for i in cal
    # conformal scores = negative probabilities
    cal_scores = -cal_prob[np.arange(n_cal), Y_cal]

    # get quantiles
    sorted_cal_scores = np.sort(cal_scores)
    q_rank = int(np.ceil((n_cal + 1) * (1 - alpha)))
    q = sorted_cal_scores[q_rank-1] # q_rank-th smallest value among the scores

    # compute the conformal set
    test_scores = -test_prob
    prediction_set = test_scores <= q

    # ensure non-empty sets (include the most probable label if empty)
    most_probable_labels = np.argmax(test_prob, axis=1)
    prediction_set[np.arange(n_test), most_probable_labels] = True

    return prediction_set   # (n_test, K) with each row = prediction set as a boolean vector


def label_alignment(t, s, K):
    """
    Get the alignment between two labels t = (t_1, ..., t_n) and s = (s_1, ..., s_n)
    
    Parameters:
    - t: array of true labels (n,)
    - s: array of obtained labels (n,)
    - K: number of clusters
    
    Returns:
    - sigma: desired permutation such that t_i = sigma(s_i) as many i's as possible
    """    

    if len(t) != len(s):
        raise ValueError("Labels must have the same length")
    
    n = len(t)
    all_ones = np.ones(n)
    arange_n = np.arange(n)
    
    # create the sparse matrices 
    t_matrix = csr_array((all_ones, (arange_n, t)), shape=(n, K))  # rows = e_{t_i}
    s_matrix = csr_array((all_ones, (arange_n, s)), shape=(n, K))  # rows = e_{s_i}

    # solve the linear assignment problem
    Q = -s_matrix.T @ t_matrix # (K, K) matrix = small matrix
    Q = Q.toarray()  # convert to dense array for linear_sum_assignment ()
    _, col_ind = linear_sum_assignment(Q)

    return col_ind

def sample_gmm(means, covariances, weights, n_samples, seed=None):
    """
    Efficiently samples from a Gaussian Mixture Model.

    Parameters:
    - means: (K, D) array of component means.
    - covariances: (K, D, D) array of component covariance matrices.
    - weights: (K,) array of component mixing proportions (must sum to 1).
    - n_samples: int, number of samples to generate.
    - seed: int or None, random seed for reproducibility.

    Returns:
    - X: (n_samples, D) array of generated samples.
    - y: (n_samples,) array of component labels (0 to K-1) corresponding to X.
    """
    rng = np.random.default_rng(seed)
    
    # Ensure inputs are numpy arrays
    means = np.asarray(means)
    covariances = np.asarray(covariances)
    weights = np.asarray(weights)
    
    # Validate shapes
    K, D = means.shape
    
    # 1. Determine how many samples to draw from each component
    # This vectorizes the component selection process
    n_samples_per_component = rng.multinomial(n_samples, weights)
    
    # Pre-allocate memory for speed
    X = np.empty((n_samples, D))
    y = np.empty(n_samples, dtype=int)
    
    # 2. Generate samples for each component in batches
    current_idx = 0
    for k, count in enumerate(n_samples_per_component):
        if count > 0:
            # Generate 'count' samples efficiently using optimized C-backend
            samples = rng.multivariate_normal(means[k], covariances[k], size=count)
            
            # Fill the pre-allocated arrays
            X[current_idx : current_idx + count] = samples
            y[current_idx : current_idx + count] = k
            
            current_idx += count
            
    # 3. Shuffle the result to mix components randomly
    # (Otherwise, output is sorted by component 0, then 1, etc.)
    perm = rng.permutation(n_samples)
    
    return X[perm], y[perm]

def sample_gamm(shapes, scales, weights, n_samples, seed=None):
    """
    Efficiently sample from a Gamma Mixture Model.

    Parameters:
    - shapes : (K, D) array of shapes (alpha) per component and dimension. Must be > 0.
    - scales : (K, D) array of scales (theta) per component and dimension. Must be > 0.
    - weights : (K,) array of component mixing proportions (must sum to 1)
    - n_samples : int, number of samples to generate.
    - seed : int or None, random seed for reproducibility.

    Returns:
    - X : (n_samples, D) array of generated samples.
    - y : (n_samples,) array of component labels (0 to K-1) corresponding to X.
    """
    rng = np.random.default_rng(seed)

    # Ensure inputs are numpy arrays
    shapes = np.asarray(shapes)
    scales = np.asarray(scales)
    weights = np.asarray(weights)

    # Validate shapes
    K, D = shapes.shape

    # 1. Determine how many samples to draw from each component
    # This vectorizes the component selection process
    n_samples_per_component = rng.multinomial(n_samples, weights)

    # Pre-allocate memory for speed
    X = np.empty((n_samples, D))
    y = np.empty(n_samples, dtype=int)

    # 2. Generate samples for each component in batches
    current_idx = 0
    for k, count in enumerate(n_samples_per_component):
        if count > 0:
            # Generate 'count' samples efficiently using optimized C-backend
            samples = rng.gamma(shape=shapes[k], scale=scales[k], size=(count, D))

            # Fill the pre-allocated arrays
            X[current_idx:current_idx+count] = samples
            y[current_idx:current_idx+count] = k

            current_idx += count

    # 3. Shuffle the result to mix components randomly
    # (Otherwise, output is sorted by component 0, then 1, etc.)
    perm = rng.permutation(n_samples)

    return X[perm], y[perm]

# Stochastic version of GaussianMixture
class GaussianMixtureS(GaussianMixture):
    def predict(self, X, seed=0):
        probs = self.predict_proba(X)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        
        cum_probs = probs.cumsum(axis=1)
        cum_probs[:, -1] = 1.0 # force the last column to be exactly 1.0 to prevent numerical issues
        u = rng.random((probs.shape[0], 1)) # draw one uniform random number per row
        y = (u < cum_probs).argmax(axis=1) # smallest index where u < cum_probs = find wher u falls in the cumulative distribution

        return y

# Older version of GaussianMixtureS
class GaussianMixtureGen(GaussianMixtureS):
    pass

# Gamma Mixture Model
class GammaMixture:
    def __init__(self, n_components=3, max_iter=1000, tol=1e-3, n_init=20, init_max_iter=15, random_state=None):
        """
        Parameters:
        -----------
        n_components : int
            Number of mixture components.
        max_iter : int
            Maximum iterations for the final deep exploitation run.
        tol : float
            Convergence tolerance.
        n_init : int
            Number of exploratory initializations (Stage 1).
        init_max_iter : int
            Number of EM iterations to run for each exploratory initialization.
        random_state : int, optional
            Controls the random seed.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init_max_iter = init_max_iter
        self.random_state = random_state
        
        self.weights = None
        self.shapes = None  
        self.scales = None  
        self.converged_ = False
        self.lower_bound_ = -np.inf 

    def _estimate_log_prob(self, X, log_X, shapes, scales, weights):
        const_terms = np.sum(shapes * np.log(scales) + gammaln(shapes), axis=1)
        term_logx = np.dot(log_X, (shapes - 1.0).T)
        term_x = np.dot(X, (1.0 / scales).T)
        return (np.log(weights + 1e-10) + term_logx - term_x - const_terms)

    def _solve_shape(self, y, current_shape, n_iter=5):
        shape = current_shape.copy()
        for _ in range(n_iter):
            f_val = np.log(shape) - psi(shape) - y
            f_prime = 1.0 / shape - polygamma(1, shape)
            f_prime = np.minimum(f_prime, -1e-12)
            shape -= f_val / f_prime
            shape = np.clip(shape, 1e-3, 2000.0)
        return shape

    def _run_em(self, X, log_X, curr_shapes, curr_scales, curr_weights, max_iterations):
        """Core EM loop, abstracted so it can be run for short or long durations."""
        N, d = X.shape
        log_likelihood_history = []
        converged = False
        
        for it in range(max_iterations):
            # E-Step
            log_prob = self._estimate_log_prob(X, log_X, curr_shapes, curr_scales, curr_weights)
            log_prob_norm = logsumexp(log_prob, axis=1)
            
            if np.any(np.isnan(log_prob_norm)):
                log_likelihood_history.append(-np.inf)
                break

            current_ll = np.sum(log_prob_norm)
            log_likelihood_history.append(current_ll)
            
            resp = np.exp(log_prob - log_prob_norm[:, np.newaxis])

            # M-Step
            Nk = resp.sum(axis=0) + 1e-10
            curr_weights = Nk / N
            
            w_mean_x = np.dot(resp.T, X) / Nk[:, np.newaxis]
            w_mean_log_x = np.dot(resp.T, log_X) / Nk[:, np.newaxis]
            
            y = np.log(w_mean_x) - w_mean_log_x
            curr_shapes = self._solve_shape(y, curr_shapes)
            curr_scales = w_mean_x / curr_shapes
            
            # Convergence Check
            if it > 0 and abs(current_ll - log_likelihood_history[-2]) < self.tol:
                converged = True
                break
                
        final_ll = log_likelihood_history[-1] if log_likelihood_history else -np.inf
        return final_ll, curr_shapes, curr_scales, curr_weights, converged

    def fit(self, X):
        X = np.maximum(X, 1e-6) 
        N, d = X.shape
        log_X = np.log(X)
        
        rng = np.random.default_rng(self.random_state)
        
        # Base statistics for initialization
        global_mean = np.mean(X, axis=0)
        global_var = np.var(X, axis=0)
        base_shape = global_mean**2 / global_var
        base_scale = global_var / global_mean
        
        best_init_ll = -np.inf
        best_shapes = None
        best_scales = None
        best_weights = None

        # ==========================================
        # STAGE 1: Short Exploratory Runs
        # ==========================================
        for init_idx in range(self.n_init):
            # Generate random starting points
            curr_weights = np.ones(self.n_components) / self.n_components
            curr_shapes = np.tile(base_shape, (self.n_components, 1)) * rng.uniform(0.6, 1.4, (self.n_components, d))
            curr_scales = np.tile(base_scale, (self.n_components, 1)) * rng.uniform(0.6, 1.4, (self.n_components, d))

            # Run a short burst of EM
            ll, shapes, scales, weights, _ = self._run_em(
                X, log_X, curr_shapes, curr_scales, curr_weights, self.init_max_iter
            )
            
            # Keep track of the best basin of attraction
            if ll > best_init_ll:
                best_init_ll = ll
                best_shapes = shapes
                best_scales = scales
                best_weights = weights

        # ==========================================
        # STAGE 2: Deep Exploitation
        # ==========================================
        if best_shapes is None:
             raise ValueError("All initializations failed due to numerical instability.")
             
        final_ll, final_shapes, final_scales, final_weights, converged = self._run_em(
            X, log_X, best_shapes, best_scales, best_weights, self.max_iter
        )
        
        # Save final state
        self.lower_bound_ = final_ll
        self.shapes = final_shapes
        self.scales = final_scales
        self.weights = final_weights
        self.converged_ = converged
        
        if not self.converged_:
            warnings.warn(
                f"GammaMixture did not converge after {self.max_iter} iterations.",
                ConvergenceWarning
            )
            
        return self

    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted.")
        X = np.maximum(X, 1e-6)
        log_prob = self._estimate_log_prob(X, np.log(X), self.shapes, self.scales, self.weights)
        log_prob_norm = logsumexp(log_prob, axis=1)
        return np.exp(log_prob - log_prob_norm[:, np.newaxis])

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted.")
        X = np.maximum(X, 1e-6)
        log_prob = self._estimate_log_prob(X, np.log(X), self.shapes, self.scales, self.weights)
        return log_prob.argmax(axis=1)


# Stochastic version of GammaMixture
class GammaMixtureS(GammaMixture):
    def predict(self, X, seed=0):
        probs = self.predict_proba(X)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        
        cum_probs = probs.cumsum(axis=1)
        cum_probs[:, -1] = 1.0 # force the last column to be exactly 1.0 to prevent numerical issues
        u = rng.random((probs.shape[0], 1)) # draw one uniform random number per row
        y = (u < cum_probs).argmax(axis=1) # smallest index where u < cum_probs = find wher u falls in the cumulative distribution

        return y

class FCM:
    def __init__(self, **kargs):
        self.kwargs = kargs

    def fit(self, X):
        self.fcm_result = fuzz.cluster.cmeans(X.T, **self.kwargs)
        self.centers_ = self.fcm_result[0] # cluster centers (n_clusters, n_features)
        self.probs = self.fcm_result[1].T  # membership degrees (n_samples, n_clusters)

    def labels_(self):
        return np.argmax(self.probs, axis=1)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_()

# Stochastic version of FCM
class FCMS(FCM):
    def predict(self, seed=0):
        probs = self.probs
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        
        cum_probs = probs.cumsum(axis=1)
        cum_probs[:, -1] = 1.0 # force the last column to be exactly 1.0 to prevent numerical issues
        u = rng.random((probs.shape[0], 1)) # draw one uniform random number per row
        y = (u < cum_probs).argmax(axis=1) # smallest index where u < cum_probs = find wher u falls in the cumulative distribution

        return y 
    
# Older version of FCMS
class FCMGen(FCMS):
    pass

# Stochastic version of spectral clustering with GMM on the spectral embedding
class SpectralGMMS:
    def __init__(self, n_neighbors, mutual=True, **kwargs):
        self.clst = GaussianMixtureS(**kwargs)
        self.K = kwargs['n_components']
        self.n_neighbors = n_neighbors
        self.mutual = mutual

    def fit(self, X):
        self.X = X
        self.G_dnn = kneighbors_graph(X, n_neighbors=self.n_neighbors) # directed graph
        self.G_mnn = self.G_dnn.multiply(self.G_dnn.T).toarray()

        if self.mutual==True:
            self.G = self.G_mnn
        else:
            self.G_nn = (self.G_dnn + self.G_dnn.T > 0).astype(int).toarray()
            self.G = self.G_nn

        self.L = np.diag(self.G.sum(axis=1)) - self.G
        self.n = self.L.shape[0]
        if self.n <= 1000:
            eigvals_all, eigvecs_all = eigh(self.L)
            self.eigvals = eigvals_all[:self.K]
            self.eigvecs = eigvecs_all[:, :self.K]
        else:
            n_eig = int(np.sqrt(self.L.shape[0]-1))  # number of eigenvalues to compute
            self.eigvals, self.eigvecs = eigsh(csr_matrix(self.L), k=n_eig, which='SM')
        self.X_embd = self.eigvecs[:, :self.K]

        self.clst.fit(self.X_embd)

    def predict(self, seed=0):
        return self.clst.predict(self.X_embd, seed=seed)

# Older version of SpectralGMMS
class SpectralGMMG(SpectralGMMS):
    pass

class ConformalClustering:
    def __init__(self, X_tr, X_cal, alpha=0.1):
        self.X_tr = X_tr
        self.X_cal = X_cal
        self.alpha = alpha

    def set_classifier(self, classifier, **kwargs):
        if classifier == 'RF':
            self.clf = RandomForestClassifier(**kwargs)
        elif classifier == 'SVC':
            self.clf = SVC(**kwargs) # probability=True should be set in kwargs
            if not self.clf.probability:
                raise ValueError("For SVC, probability=True must be set")
        else:
            raise ValueError("Invalid classification method. Choose from ['RF', 'SVC']")
        
    def fit(self, clustering, gen_seed_tr=0, gen_seed_cal=0, **kwargs):
        if clustering == 'GMM':
            self.clst_tr = GaussianMixture(**kwargs)
            self.y_tr = self.clst_tr.fit_predict(self.X_tr)
            self.clst_cal = GaussianMixture(**kwargs)
            self.y_cal = self.clst_cal.fit_predict(self.X_cal)
            self.K = kwargs['n_components']

        elif clustering == 'GaMM':
            self.clst_tr = GammaMixture(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.y_tr = self.clst_tr.predict(self.X_tr)
            self.clst_cal = GammaMixture(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.y_cal = self.clst_cal.predict(self.X_cal)
            self.K = kwargs['n_components']

        elif clustering == 'KM':
            self.clst_tr = KMeans(**kwargs)
            self.y_tr = self.clst_tr.fit_predict(self.X_tr)
            self.clst_cal = KMeans(**kwargs)
            self.y_cal = self.clst_cal.fit_predict(self.X_cal)
            self.K = kwargs['n_clusters']

        elif clustering == 'FCM':
            self.clst_tr = FCM(**kwargs)
            self.y_tr = self.clst_tr.fit_predict(self.X_tr)
            self.clst_cal = FCM(**kwargs)
            self.y_cal = self.clst_cal.fit_predict(self.X_cal)
            self.K = kwargs['c']

        elif clustering == 'Hier':
            self.clst_tr = AgglomerativeClustering(**kwargs)
            self.y_tr = self.clst_tr.fit_predict(self.X_tr)
            self.clst_cal = AgglomerativeClustering(**kwargs)
            self.y_cal = self.clst_cal.fit_predict(self.X_cal)
            self.K = kwargs['n_clusters']

        # Stochastic clustering methods (seeds for sampling)
        elif clustering in ['GMMS', 'GMM-Gen']:
            self.clst_tr = GaussianMixtureS(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.y_tr = self.clst_tr.predict(self.X_tr, seed=gen_seed_tr)
            self.clst_cal = GaussianMixtureS(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.y_cal = self.clst_cal.predict(self.X_cal, seed=gen_seed_cal)
            self.K = kwargs['n_components']

        elif clustering == 'GaMMS':
            self.clst_tr = GammaMixtureS(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.y_tr = self.clst_tr.predict(self.X_tr, seed=gen_seed_tr)
            self.clst_cal = GammaMixtureS(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.y_cal = self.clst_cal.predict(self.X_cal, seed=gen_seed_cal)
            self.K = kwargs['n_components']

        elif clustering in ['FCMS', 'FCM-Gen']:
            self.clst_tr = FCMS(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.y_tr = self.clst_tr.predict(seed=gen_seed_tr)
            self.clst_cal = FCMS(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.y_cal = self.clst_cal.predict(seed=gen_seed_cal)
            self.K = kwargs['c']

        elif clustering in ['SpectralGMMS', 'SpectralGMM-Gen']:
            self.clst_tr = SpectralGMMS(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.y_tr = self.clst_tr.predict(seed=gen_seed_tr)
            self.clst_cal = SpectralGMMS(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.y_cal = self.clst_cal.predict(seed=gen_seed_cal)
            self.K = kwargs['n_components']

        else:
            raise ValueError("Invalid clustering method. Choose from ['GMM', 'GMMS', 'GMM-Gen', 'GaMM', 'GaMMS', 'KM', 'FCM', 'FCMS', 'FCM-Gen', 'Hier', 'SpectralGMMS', 'SpectralGMM-Gen']")
        
        if self.K != len(np.unique(self.y_tr)): # number of clusters learned does not match
            raise ValueError("Number of clusters in clustering does not match the specified number of clusters")
        
        # Fit the classifier
        self.clf.fit(self.X_tr, self.y_tr)
        if self.clf.classes_.shape[0] != self.K:
            raise ValueError("Number of classes in classifier does not match number of clusters")

        # Get calibration probabilities
        self.cal_prob = self.clf.predict_proba(self.X_cal) # (n_cal, K)
        
        # Align calibration labels
        y_cal_hat = self.clf.predict(self.X_cal)
        align_to_tr = label_alignment(y_cal_hat, self.y_cal, self.K)
        self.y_cal_al = align_to_tr[self.y_cal] # aligned calibration labels


    def conformal_set(self, X_test, method='narc'):
        """
        Parameters:
        - X_test: test points (n_test, K)
        Returns:
        - prediction_set: array of shape (n_test, K) with each row = prediction (boolean vector)
        """
        if method == 'arc':
            _conformal = conformal_arc
        elif method == 'arc_ne':
            _conformal = conformal_arc_ne
        elif method == 'narc':
            _conformal = conformal_narc
        elif method == 'clp':
            _conformal = conformal_clp
        elif method == 'clp_ne':
            _conformal = conformal_clp_ne
        else:
            raise ValueError("Invalid method. Choose from ['arc', 'arc_ne', 'narc', 'clp', 'clp_ne']")


        test_prob = self.clf.predict_proba(X_test)

        return _conformal(self.cal_prob, self.y_cal_al, test_prob, alpha=self.alpha)
            

    def validate(self, X_val, y_val, X_al, y_al, method='narc'):
        """
        Parameters:
        - X_val: validation data (n_val, p)
        - y_val: true labels of the validation data (n_val,)
        - X_al: alignment data (n_al, p)
        - y_al: true labels of the alignment data (n_al,)
        Returns:
        """

        if method not in ['arc', 'arc_ne', 'narc', 'clp', 'clp_ne']:
            raise ValueError("Invalid method. Choose from ['arc', 'arc_ne', 'narc', 'clp', 'clp_ne']")
        
        # Oracle alignment using the alignment data
        C_bool = self.conformal_set(X_al, method=method) # conformal sets as boolean (n_al, K)
        C = csr_array(C_bool.astype(int)) # convert to sparse matrix

        n_al = len(y_al)
        y_mat = csr_array(  
            (np.ones(n_al), (np.arange(n_al), y_al)),
            shape=(n_al, self.K)
        ) # labels to one-hot encoding (row = e_{y_i})

        Q = -y_mat.T @ C # (K, K) cost matrix
        _, align_true_to_conformal = linear_sum_assignment(Q.toarray()) 

        # Compute conformal sets for the validation data
        conformal_sets_bool = self.conformal_set(X_val, method=method) # (n_val, K)
        conformal_sets = [np.where(row)[0] for row in conformal_sets_bool]  # list of arrays
        sizes = [len(s) for s in conformal_sets]

        # Align true labels to conformal sets
        y_val_aligned = align_true_to_conformal[y_val]
        n_val = len(y_val)

        # Check coverage and set sizes
        if n_val == 1:
            coverage = int(y_val_aligned[0] in conformal_sets[0])
            return coverage, sizes[0]
        else:
            coverage = np.array([int(y_val_aligned[i] in conformal_sets[i]) for i in range(n_val)])
            return coverage, sizes

class GMMCutoff:
    def __init__(self, X, alpha=0.1):
        self.X = X
        self.alpha = alpha

    def fit(self, **kwargs):
        self.gmm = GaussianMixture(**kwargs)
        self.y_clst = self.gmm.fit_predict(self.X)
        self.K = kwargs['n_components']
        if self.K != len(np.unique(self.y_clst)):
            raise ValueError("Number of clusters in GMM does not match the specified number of clusters")

    def cutoff_set(self, X_test):
        test_prob = self.gmm.predict_proba(X_test)
        return cutoff(test_prob, self.alpha)
    
    def validate(self, X_val, y_val, X_al, y_al):
        # Oracle alignment using the alignment data
        C_bool = self.cutoff_set(X_al) # conformal sets as boolean (n_al, K)
        C = csr_array(C_bool.astype(int)) # convert to sparse matrix

        n_al = len(y_al)
        y_mat = csr_array(  
            (np.ones(n_al), (np.arange(n_al), y_al)),
            shape=(n_al, self.gmm.n_components)
        ) # labels to one-hot encoding (row = e_{y_i})

        Q = -y_mat.T @ C # (K, K) cost matrix
        _, align_true_to_cutoff = linear_sum_assignment(Q.toarray()) 

        # Compute cutoff sets for the validation data
        cutoff_sets_bool = self.cutoff_set(X_val) # (n_val, K)
        cutoff_sets = [np.where(row)[0] for row in cutoff_sets_bool]  # list of arrays
        sizes = [len(s) for s in cutoff_sets]

        # Align true labels to cutoff sets
        y_val_aligned = align_true_to_cutoff[y_val]
        n_val = len(y_val)

        # Check coverage and set sizes
        if n_val == 1:
            coverage = int(y_val_aligned[0] in cutoff_sets[0])
            return coverage, sizes[0]
        else:
            coverage = np.array([int(y_val_aligned[i] in cutoff_sets[i]) for i in range(n_val)])
            return coverage, sizes
    
class GaMMCutoff:
    def __init__(self, X, alpha=0.1):
        self.X = X
        self.alpha = alpha

    def fit(self, **kwargs):
        self.gamm = GammaMixture(**kwargs)
        self.gamm.fit(self.X)
        self.y_clst = self.gamm.predict(self.X)
        self.K = kwargs['n_components']
        if self.K != len(np.unique(self.y_clst)):
            raise ValueError("Number of clusters in GaMM does not match the specified number of clusters")

    def cutoff_set(self, X_test):
        test_prob = self.gamm.predict_proba(X_test)
        return cutoff(test_prob, self.alpha)
    
    def validate(self, X_val, y_val, X_al, y_al):
        # Oracle alignment using the alignment data
        C_bool = self.cutoff_set(X_al) # conformal sets as boolean (n_al, K)
        C = csr_array(C_bool.astype(int)) # convert to sparse matrix

        n_al = len(y_al)
        y_mat = csr_array(  
            (np.ones(n_al), (np.arange(n_al), y_al)),
            shape=(n_al, self.gamm.n_components)
        ) # labels to one-hot encoding (row = e_{y_i})

        Q = -y_mat.T @ C # (K, K) cost matrix
        _, align_true_to_cutoff = linear_sum_assignment(Q.toarray()) 

        # Compute cutoff sets for the validation data
        cutoff_sets_bool = self.cutoff_set(X_val) # (n_val, K)
        cutoff_sets = [np.where(row)[0] for row in cutoff_sets_bool]  # list of arrays
        sizes = [len(s) for s in cutoff_sets]

        # Align true labels to cutoff sets
        y_val_aligned = align_true_to_cutoff[y_val]
        n_val = len(y_val)

        # Check coverage and set sizes
        if n_val == 1:
            coverage = int(y_val_aligned[0] in cutoff_sets[0])
            return coverage, sizes[0]
        else:
            coverage = np.array([int(y_val_aligned[i] in cutoff_sets[i]) for i in range(n_val)])
            return coverage, sizes

def cutoff(test_prob, alpha=0.1):
    sorted_ids = np.argsort(-test_prob, axis=1)  # descending order
    sorted_probs = np.take_along_axis(test_prob, sorted_ids, axis=1)
    test_prob_rank = np.argsort(sorted_ids, axis=1)
    sorted_probs_cumsum = np.cumsum(sorted_probs, axis=1)
    cutoff = sorted_probs_cumsum < 1-alpha
    cutoff[np.arange(len(test_prob)),np.argmin(cutoff, axis=1)] = True
    prediction_set = np.take_along_axis(cutoff, test_prob_rank, axis=1)
    return prediction_set