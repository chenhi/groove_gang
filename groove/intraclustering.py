from matplotlib.patches import Ellipse
import scipy, math, sklearn
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Get Matplotlib patches for various elliptical confidence regions
# Only 2D plots - if dimension > 2, need to specify orthogonal components shape (num_components, vector) and mean shift
def get_ellipse_patch(center, cov, components=None, shift=0, confidence=0.95):
    if type(components) is np.ndarray:
        proj = components.T @ np.linalg.inv(components @ components.T)
        cov = proj.T @ cov @ proj
        center = (center - shift) @ proj

    # Figure out number of stdevs for confidence
    scale = math.sqrt(scipy.stats.chi2.ppf(q=confidence, df=center.shape[0]))
    
    u, s, vh = np.linalg.svd(cov)
    # If det < 0, reflect either u_1 or u_2 to its negative, may as well do u_2
    # For arccos, need y-value of u_1 to be positive
    if u[0,1] < 0:
        u[:,0] = -u[:,0]
    th = np.arccos(np.dot(np.array([1, 0]), u[:,0])) * 180 / np.pi
    #print(np.sqrt(np.linalg.det(cov)))
    return Ellipse((center[0], center[1]), np.sqrt(s[0]) * scale * 2, np.sqrt(s[1])* scale * 2, angle=th, alpha=0.5, color='grey')



# Check if the confidence region contains a list of points
# center shape (n,), covariance shape (n,n), points shape (p, n) where p is number of points
# Returns shape (n, 1) boolean
def ellipse_contains_points(center, covariance, points, confidence=0.95):
    u, s, vh = np.linalg.svd(covariance)
    a = u @ np.diag(np.sqrt(s))
    return np.linalg.norm((points - center) @ np.linalg.inv(a).T, axis=1) ** 2 < scipy.stats.chi2.ppf(q=confidence, df=center.shape[0])
    

# Gets patches and containment data for a given fitted GaussianMixture object
# If dimension > 2, need a PCA object to reduce dimensions to 2
# Returns an list of patches, a Numpy array of containment data
def get_patches(gm, confidence, pts, pca=None, how_reduce='top'):
    assert gm.means_[0].flatten().shape[0] <= 2 or type(pca) is sklearn.decomposition._pca.PCA
    patches = []
    contains = []
    for i in range(gm.means_.shape[0]):
        if pca != None:
            e = get_ellipse_patch(gm.means_[i], gm.covariances_[i], confidence=confidence, components=pca.components_[0:2], shift=pca.mean_)
        else:
            e = get_ellipse_patch(gm.means_[i], gm.covariances_[i], confidence=confidence)
        patches.append(e)
        contains.append(ellipse_contains_points(gm.means_[i], gm.covariances_[i], pts, confidence))

    return patches, np.stack(contains, axis=0)


# Algorithm for determining the number of components according to our winnowing criteria: no more than 10% overlaps, and each 95% confidence region should contain 20% of all data points
# Deprecated in favor of bic_winnow_gm_components
def winnow_gm_components(data, simulations_per_level=3, confidence_limit=0.80, overlap_allowance = 0.1, cluster_threshold = 0.3, use_weights = False, start = None, verbose=False):
    # Add 1 in case of rounding error
    if start == None:
        start = 10 if cluster_threshold == 0. else min(int(1 / cluster_threshold) + 1, 10)

    if start > data.shape[0]:
        start = data.shape[0]

    for i in range(start, 0, -1):
        sims = []
        scores = []
        for _ in range(simulations_per_level):
            gm = GaussianMixture(n_components=i)
            gm.fit(data)
            sims.append(gm)
            scores.append(gm.score(data))
            if verbose:
                print(f"Score: {scores[-1]}")
                print(f"Num components: {i}")
        best_index = np.array(scores).argmax()
        gm = sims[best_index]

        # If i = 1, then we are done anyway, so break
        if i == 1:
            break

        contains = []
        for j in range(i):
            contains.append(ellipse_contains_points(gm.means_[j], gm.covariances_[j], data, confidence=confidence_limit))
        contains = np.stack(contains)    

        # Check if 10% or more of data points are contained in overlaps
        overlaps = (contains.sum(axis=0) > 1).sum().item() / data.shape[0]
        if verbose:
            print(f"Overlaps {overlaps}")
        if overlaps > overlap_allowance:
            if verbose:
                print(f"Failed because overlaps exceeded allowance {overlap_allowance}.")
            continue
        
        # If use_weights, check if any ellipses have weight under threshold
        if use_weights and (gm.means_ >= cluster_threshold).prod().item() != 1:
            continue
        # If use_weights is off, then check if any ellipses contain less than 10% of data points
        containment = contains.sum(axis=1) / data.shape[0]
        if verbose:
            print(f"Containment {containment}")
        if not use_weights and (containment >= cluster_threshold).prod().item() != 1:
            if verbose:
                print(f"Failed because some clusters beneath required coverage {cluster_threshold}.")
            continue
        # Otherwise, we are done

        return gm
        
    return gm


def bic_winnow_gm_components(data, max_clusters = 5, simulations_per_level=3, pca_variance_threshold = .95, verbose=False):
    
    # Maximum clusters cannot exceed number of samples
    max_clusters = min(max_clusters, data.shape[0])

    # First, do PCA to reduce to 95% of explained variance
    n_features = min(data.shape[0], data.shape[1])
    total_pca = PCA(n_components=n_features)
    total_pca.fit(data)
    if verbose:
        _, axs = plt.subplots(figsize=(18, 8), nrows=2, ncols=1)
        axs[0].plot(range(1, n_features+1), total_pca.explained_variance_)
        axs[1].plot(range(1, n_features+1), np.cumsum(total_pca.explained_variance_ratio_))

    pca_cutoff = (np.cumsum(total_pca.explained_variance_ratio_) > pca_variance_threshold).argmax() + 1

    pca = PCA(n_components=pca_cutoff)
    pca.fit(data)
    red_data = pca.transform(data)

    # Now, on reduced data:
    # for each number of clusters, fit Gaussian mixture max_clusters times and pick the best fit
    # between number of clusters, pick model with best BIC
    gms = []
    bics = []
    for i in range(max_clusters):
        level_gm = []
        level_score = []
        for _ in range(simulations_per_level):
            gm = GaussianMixture(n_components=i+1)
            gm.fit(red_data)
            level_gm.append(gm)
            level_score.append(gm.score(red_data))
            if verbose:
                print(f"Num components: {i+1}")
                print(f"Converged? {gm.converged_}")
                print(f"Log-likelihood: {level_score[-1]}")
        level_score = np.array(level_score)
        best_index = level_score.argmax()
        if verbose:
            print(f"BIC: {level_score}, selecting index {best_index}")
        gms.append(level_gm[best_index])
        bics.append(level_gm[best_index].bic(red_data))

    best_index = np.array(bics).argmin()
    if verbose:
        print(f"Final BICs: {bics}\nSelecting {best_index + 1} components")

    # Refit on original data
    gm = GaussianMixture(n_components=best_index + 1)
    gm.fit(data)
    return gm


# how indicates whether to return only the top cluster or all clusters
def get_primary_gaussian_mean(data, how='top', max_clusters=5, simulations_per_level=3, pca_variance_threshold=.95):
    gm = bic_winnow_gm_components(data, max_clusters=max_clusters, simulations_per_level=simulations_per_level, pca_variance_threshold=pca_variance_threshold)
    if how == 'all':
        return gm.means_
    else:
        return gm.means_[gm.weights_.argmax(keepdims=True)[0]]


# Finds the closest data point to the given means
def find_closest_indices(means, data):
    outs = []
    for i in range(means.shape[0]):
        distances = np.linalg.norm(means[i] - data, axis=1)
        outs.append(distances.argmin(keepdims=True)[0])
    return np.stack(outs)