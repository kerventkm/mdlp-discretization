import cupy as cp
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y, column_or_1d, check_random_state
from sklearn.utils.validation import check_is_fitted


def gpu_slice_entropy(y_gpu, start, end):
    """Compute entropy of a slice on GPU"""
    y_slice = y_gpu[start:end]
    counts = cp.bincount(y_slice)
    probs = counts.astype(cp.float32) / (end - start)
    # Remove zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    entropy = -cp.sum(probs * cp.log(probs))
    return float(entropy), int(cp.sum(probs != 0))


def gpu_find_cut(y_gpu, start, end):
    """Find the best cut point using GPU acceleration"""
    length = end - start
    if length <= 1:
        return -1
    
    # Get consecutive differences to find potential cut points
    diff = cp.diff(y_gpu[start:end])
    potential_cuts = cp.nonzero(diff)[0] + start + 1
    
    if len(potential_cuts) == 0:
        return -1
        
    # Initialize variables for best cut
    min_entropy = float('inf')
    best_k = -1
    
    # Evaluate entropy for each potential cut point
    for k in potential_cuts:
        first_half = float(k - start) / length * gpu_slice_entropy(y_gpu, start, k)[0]
        second_half = float(end - k) / length * gpu_slice_entropy(y_gpu, k, end)[0]
        curr_entropy = first_half + second_half
        
        if curr_entropy < min_entropy:
            min_entropy = curr_entropy
            best_k = int(k)
    
    return best_k


def gpu_reject_split(y_gpu, start, end, k):
    """Determine whether to reject a split using MDL criterion on GPU"""
    if k == -1:
        return True
        
    N = float(end - start)
    entropy1, k1 = gpu_slice_entropy(y_gpu, start, k)
    entropy2, k2 = gpu_slice_entropy(y_gpu, k, end)
    whole_entropy, k0 = gpu_slice_entropy(y_gpu, start, end)
    
    # Calculate gain
    part1 = 1 / N * ((k - start) * entropy1 + (end - k) * entropy2)
    gain = whole_entropy - part1
    
    # Calculate delta
    entropy_diff = k0 * whole_entropy - k1 * entropy1 - k2 * entropy2
    delta = np.log(pow(3, k0) - 2) - entropy_diff
    
    return gain <= 1 / N * (np.log(N - 1) + delta)


class MDLP_GPU(BaseEstimator, TransformerMixin):
    """GPU-accelerated version of MDLP discretization algorithm.
    
    This implementation uses CuPy for GPU acceleration of the core computations.
    
    Parameters
    ----------
    continuous_features : array-like or None, default=None
        Indices of continuous features to discretize. If None, all features
        are treated as continuous.
    
    min_depth : int, default=0
        Minimum depth of the discretization tree.
        
    random_state : int or None, default=None
        Random state for reproducibility.
        
    min_split : float, default=1e-3
        Minimum fraction of samples required to split a node.
        
    dtype : type, default=int
        Output dtype for discretized values.
    """
    
    def __init__(self, continuous_features=None, min_depth=0, random_state=None,
                 min_split=1e-3, dtype=int):
        self.continuous_features = continuous_features
        self.min_depth = min_depth
        self.random_state = random_state
        self.min_split = min_split
        self.dtype = dtype
        
    def fit(self, X, y):
        """Fit the discretizer to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, force_all_finite=True, ensure_2d=True, dtype=np.float64)
        y = column_or_1d(y)
        y = check_array(y, ensure_2d=False, dtype=int)
        X, y = check_X_y(X, y)
        
        if len(X.shape) != 2:
            raise ValueError("Invalid input dimension for X. Expected 2D array.")
            
        # Initialize random state and permute data
        state = check_random_state(self.random_state)
        perm = state.permutation(len(y))
        X = X[perm]
        y = y[perm]
        
        # Determine continuous features
        if self.continuous_features is None:
            self.continuous_features_ = np.arange(X.shape[1])
        else:
            continuous_features = np.array(self.continuous_features)
            if continuous_features.dtype == np.bool:
                continuous_features = np.arange(len(continuous_features))[continuous_features]
            else:
                continuous_features = continuous_features.astype(int, casting='safe')
                assert np.max(continuous_features) < X.shape[1] and np.min(continuous_features) >= 0
            self.continuous_features_ = continuous_features
            
        self.cut_points_ = [None] * X.shape[1]
        self.mins_ = np.min(X, axis=0)
        self.maxs_ = np.max(X, axis=0)
        
        # Transfer y to GPU
        y_gpu = cp.asarray(y)
        
        # Process each continuous feature
        for index in self.continuous_features_:
            col = X[:, index]
            order = cp.argsort(cp.asarray(col))
            col_sorted = col[cp.asnumpy(order)]
            y_sorted_gpu = y_gpu[order]
            
            cut_points = set()
            search_intervals = [(0, len(col), 0)]  # (start, end, depth)
            
            while search_intervals:
                start, end, depth = search_intervals.pop()
                if end - start <= self.min_split * len(col):
                    continue
                    
                k = gpu_find_cut(y_sorted_gpu, start, end)
                
                if (k == -1) or (depth >= self.min_depth and 
                                gpu_reject_split(y_sorted_gpu, start, end, k)):
                    front = float("-inf") if start == 0 else (col_sorted[start-1] + col_sorted[start]) / 2
                    back = float("inf") if end == len(col) else (col_sorted[end-1] + col_sorted[end]) / 2
                    
                    if front == back:
                        continue
                    if front != float("-inf"):
                        cut_points.add(front)
                    if back != float("inf"):
                        cut_points.add(back)
                    continue
                
                search_intervals.append((start, k, depth + 1))
                search_intervals.append((k, end, depth + 1))
            
            self.cut_points_[index] = np.sort(list(cut_points))
            
        return self
        
    def transform(self, X):
        """Transform the data by discretizing continuous features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to discretize.
            
        Returns
        -------
        X_discrete : array-like of shape (n_samples, n_features)
            The discretized data.
        """
        X = check_array(X, force_all_finite=True, ensure_2d=False)
        check_is_fitted(self, "cut_points_")
        
        output = X.copy()
        for i in self.continuous_features_:
            if self.cut_points_[i] is not None:
                output[:, i] = np.searchsorted(self.cut_points_[i], X[:, i])
        return output.astype(self.dtype)
        
    def cat2intervals(self, X, index):
        """Convert categorical data back to intervals.
        
        Parameters
        ----------
        X : array-like
            The discretized array
            
        index : int
            Feature index to convert
            
        Returns
        -------
        intervals : list of tuples
            List of (start, end) intervals for each category
        """
        cp_indices = X[:, index]
        return self.assign_intervals(cp_indices, index)
        
    def assign_intervals(self, cp_indices, index):
        """Assign cut point indices to intervals.
        
        Parameters
        ----------
        cp_indices : array-like
            Cut point indices
            
        index : int
            Feature index
            
        Returns
        -------
        intervals : list of tuples
            List of (start, end) intervals
        """
        cut_points = self.cut_points_[index]
        if cut_points is None:
            raise ValueError(f"The given index {index} has not been discretized!")
            
        non_zero_mask = cp_indices[cp_indices - 1 != -1].astype(int) - 1
        fronts = np.zeros(cp_indices.shape)
        fronts[cp_indices == 0] = float("-inf")
        fronts[cp_indices != 0] = cut_points[non_zero_mask]
        
        n_cuts = len(cut_points)
        backs = np.zeros(cp_indices.shape)
        non_n_cuts_mask = cp_indices[cp_indices != n_cuts].astype(int)
        backs[cp_indices == n_cuts] = float("inf")
        backs[cp_indices != n_cuts] = cut_points[non_n_cuts_mask]
        
        return [(front, back) for front, back in zip(fronts, backs)] 