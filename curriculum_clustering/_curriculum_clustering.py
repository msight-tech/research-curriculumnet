# -*- coding: utf-8 -*-
"""
Curriculum clustering algorithm

When teaching a particular concept, it can be useful to break down the learning into lessons of increasing complexity.
Consider a curriculum, of various subsets of difficulty, for machine learning. Instead of presenting samples
in random order, imagine starting with the easiest examples first, and the most complex ones later.
In this way, the machine can develop a solid foundation in a particular concept, before being exposed to more
sophistication. This can help enable a machine to learn from noisy (complex) samples, by learning easy (simple) ones
first. In terms of technique, this forms the basic principle of a weakly supervised learning approach called
Curriculum Learning (CurriculumNet).

The input is a set of feature vectors and corresponding concept (category) labels. Normally clustering algorithms
do not need labels, however this algorithm does require (possibly noisy) labels because they represent the concepts
that should be learned over a curriculum of increasing complexity. The number of subsets of the curriculum is set by
a subsets parameter. The algorithm will cluster *each* concept category into N subsets using distribution density
in an unsupervised manner.

Consider the application to computer vision, where each vector represents an image. A curriculum subset with a high
density value means all images are close to each other in feature space, suggesting that these images have a strong
similarity. We define this subset as a clean one, by assuming most of the labels are correct. The subset with a small
density value means the images have a large diversity in visual appearance, which may include more irrelevant images
with incorrect labels. This subset is considered as noisy data. Therefore, we generate a number of subsets in each
category, arranged from clean, noisy, to highly noisy ones, which are ordered with increasing complexity.

If you'd like to learn more please refer to, and / or if you find this work useful for your research, please cite:

    CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images
    S. Guo, W. Huang, H. Zhang, C. Zhuang, D. Dong, M. R. Scott, D. Huang
    European Conference on Computer Vision (ECCV), 2018 (arXiv:1808.01097)

"""

# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import time
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import check_array, check_consistent_length, gen_batches


def cluster_curriculum_subsets(X, y, n_subsets=3, method='default', density_t=0.6, verbose=False,
                               dim_reduce=256, batch_max=500000, random_state=None, calc_auxiliary=False):
    """Perform Curriculum Clustering from vector array or distance matrix.

    The algorithm will cluster each category into n subsets by analyzing distribution density.

    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array (the embedding space of the samples.)

    y : array-like, size=[n_samples]
        category labels (the curriculum will be learned for each of these categories into N subsets)

    verbose : bool, optional, default False
        Whether to print progress messages to stdout.

    density_t : float, optional
        The density threshold for neighbors to be clustered together into a subset

    n_subsets : int, optional (default = 3)
        The number of subsets to cluster each category into. For example, if set to 3, then the categories
        outputted will be assigned a label 0, 1, or 2. Where 0 contains the simplest (most similar) samples,
        1 contains middle level (somewhat similar samples), and 2 contains most complex (most diverse) samples.

    random_state : int, RandomState instance or None (default), optional
        The generator used to make random selections within the algorithm. Use an int to make the
        randomness deterministic.

    method : {'default', 'gaussian'}, optional
        The algorithm to be used to calculate local density values. The default algorithm
        follows the approach outlined in the scientific paper referenced below.

    dim_reduce : int, optional (default = 256)
        The dimensionality to reduce the feature vector to, prior to calculating distance.
        Lower dimension is more efficient, but degrades performance, and visa-versa.

    batch_max : int, optional (default = 500000)
        The maximum batch of feature vectors to process at one time (loaded into memory).

    calc_auxiliary : bool, optional (default = False)
        Provide auxiliary including delta centers and density centers.
        This can be useful information for visualization, and debugging, amongst other use-cases.
        The downside is the processing time significantly increases if turned on.

    Returns
    -------
    all_clustered_labels : array [n_samples]
        Clustered labels for each point. Labels are integers ordered from most simple to most complex.
        E.g. if curriculum subsets=3, then label=0 is simplest, labels=1 is harder, and label=n is hardest.

    auxiliary_info : list
        If calc_auxiliary is set to True, this list contains collected auxiliary information
        during the clustering process, including delta centers, which can be useful for visualization.

    References
    ----------
    S. Guo, W. Huang, H. Zhang, C. Zhuang, D. Dong, M. R. Scott, D. Huang,
    "CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images".
    In: Proceedings of the European Conference on Computer Vision (ECCV),
    Munich, Germany, 2018. (arXiv:1808.01097)
    """

    if not density_t > 0.0:
        raise ValueError("density_thresh must be positive.")
    X = check_array(X, accept_sparse='csr')
    check_consistent_length(X, y)

    unique_categories = set(y)
    t0 = None
    pca = None
    auxiliary_info = []
    if X.shape[1] > dim_reduce:
        pca = PCA(n_components=dim_reduce, copy=False, random_state=random_state)

    # Initialize all labels as negative one which represents un-clustered 'noise'.
    # Post-condition: after clustering, there should be no negatives in the label output.
    all_clustered_labels = np.full(len(y), -1, dtype=np.intp)

    for cluster_idx, current_category in enumerate(unique_categories):
        if verbose:
            t0 = time.time()

        # Collect the "learning material" for this particular category
        dist_list = [i for i, label in enumerate(y) if label == current_category]

        for batch_range in gen_batches(len(dist_list), batch_size=batch_max):
            batch_dist_list = dist_list[batch_range]

            # Load data subset
            subset_vectors = np.zeros((len(batch_dist_list), X.shape[1]), dtype=np.float32)
            for subset_idx, global_idx in enumerate(batch_dist_list):
                subset_vectors[subset_idx, :] = X[global_idx, :]

            # Calc distances
            if pca:
                subset_vectors = pca.fit_transform(subset_vectors)
            m = np.dot(subset_vectors, np.transpose(subset_vectors))
            t = np.square(subset_vectors).sum(axis=1)
            distance = np.sqrt(np.abs(-2 * m + t + np.transpose(np.array([t]))))

            # Calc densities
            if method == 'gaussian':
                densities = np.zeros((len(subset_vectors)), dtype=np.float32)
                distance = distance / np.max(distance)
                for i in xrange(len(subset_vectors)):
                    densities[i] = np.sum(1 / np.sqrt(2 * np.pi) * np.exp((-1) * np.power(distance[i], 2) / 2.0))
            else:
                densities = np.zeros((len(subset_vectors)), dtype=np.float32)
                flat_distance = distance.reshape(distance.shape[0] * distance.shape[1])
                dist_cutoff = np.sort(flat_distance)[int(distance.shape[0] * distance.shape[1] * density_t)]
                for i in xrange(len(batch_dist_list)):
                    densities[i] = len(np.where(distance[i] < dist_cutoff)[0]) - 1  # remove itself
            if len(densities) < n_subsets:
                raise ValueError("Cannot cluster into {} subsets due to lack of density diversification,"
                                 " please try a smaller n_subset number.".format(n_subsets))

            # Optionally, calc auxiliary info
            if calc_auxiliary:
                # Calculate deltas
                deltas = np.zeros((len(subset_vectors)), dtype=np.float32)
                densities_sort_idx = np.argsort(densities)
                for i in xrange(len(densities_sort_idx) - 1):
                    larger = densities_sort_idx[i + 1:]
                    larger = larger[np.where(larger != densities_sort_idx[i])]  # remove itself
                    deltas[i] = np.min(distance[densities_sort_idx[i], larger])

                # Find the centers and package
                center_id = np.argmax(densities)
                center_delta = np.max(distance[np.argmax(densities)])
                center_density = densities[center_id]
                auxiliary_info.append((center_id, center_delta, center_density))

            model = KMeans(n_clusters=n_subsets, random_state=random_state)
            model.fit(densities.reshape(len(densities), 1))
            clusters = [densities[np.where(model.labels_ == i)] for i in xrange(n_subsets)]
            n_clusters_made = len(set([k for j in clusters for k in j]))
            if n_clusters_made < n_subsets:
                raise ValueError("Cannot cluster into {} subsets, please try a smaller n_subset number, such as {}.".
                                 format(n_subsets, n_clusters_made))

            cluster_mins = [np.min(c) for c in clusters]
            bound = np.sort(np.array(cluster_mins))

            # Distribute into curriculum subsets, and package into global adjusted returnable array, optionally aux too
            other_bounds = xrange(n_subsets - 1)
            for i in xrange(len(densities)):

                # Check if the most 'clean'
                if densities[i] >= bound[n_subsets - 1]:
                    all_clustered_labels[batch_dist_list[i]] = 0
                # Else, check the others
                else:
                    for j in other_bounds:
                        if bound[j] <= densities[i] < bound[j + 1]:
                            all_clustered_labels[batch_dist_list[i]] = len(bound) - j - 1

        if verbose:
            print "Clustering {} of {} categories into {} curriculum subsets ({:.2f} secs).".format(
                cluster_idx + 1, len(unique_categories), n_subsets, time.time() - t0)

    if (all_clustered_labels > 0).all():
        raise ValueError("A clustering error occurred: incomplete labels detected.")

    return all_clustered_labels, auxiliary_info


class CurriculumClustering(BaseEstimator, ClusterMixin):
    """Perform Curriculum Clustering from vector array or distance matrix.

    The algorithm will cluster *each* category into N subsets using distribution density in an unsupervised manner.
    The subsets can be thought of stages in an educational curriculum, going from easiest to hardest learning material.
    For information, please see see CurriculumNet, a weakly supervised learning approach that leverages this technique.

    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array (the embedding space of the samples.)

    y : array-like, size=[n_samples]
        category labels (the curriculum will be learned for each of these categories into N subsets)

    verbose : bool, optional, default False
        Whether to print progress messages to stdout.

    density_t : float, optional
        The density threshold for neighbors to be clustered together

    n_subsets : int, optional (default = 3)
        The number of subsets to cluster each category into. For example, if set to 3, then the categories
        outputted will be assigned a label 0, 1, or 2. Where 0 contains the simplest (most similar) samples,
        1 contains middle level (somewhat similar samples), and 2 contains most complex (most diverse) samples.

    random_state : int, RandomState instance or None (default), optional
        The generator used to make random selections within the algorithm. Use an int to make the
        randomness deterministic.

    method : {'default', 'gaussian'}, optional
        The algorithm to be used to calculate local density values. The default algorithm
        follows the approach outlined in the scientific paper referenced below.

    dim_reduce : int, optional (default = 256)
        The dimensionality to reduce the feature vector to, prior to calculating distance.
        Lower dimension is more efficient, but degrades performance, and visa-versa.

    batch_max : int, optional (default = 500000)
        The maximum batch of feature vectors to process at one time (loaded into memory).

    calc_auxiliary : bool, optional (default = False)
        Provide auxiliary including delta centers and density centers.
        This can be useful information for visualization, and debugging, amongst other use-cases.
        The downside is the processing time significantly increases if turned on.

    Returns
    -------
    all_clustered_labels : array [n_samples]
        Clustered labels for each point. Labels are integers ordered from most simple to most complex.
        E.g. if curriculum subsets=3, then label=0 is simplest, labels=1 is harder, and label=n is hardest.

    auxiliary_info : list
        If calc_auxiliary is set to True, this list contains collected auxiliary information
        during the clustering process, including delta centers, which can be useful for visualization.

    References
    ----------
    S. Guo, W. Huang, H. Zhang, C. Zhuang, D. Dong, M. R. Scott, D. Huang,
    "CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images".
    In: Proceedings of the European Conference on Computer Vision (ECCV),
    Munich, Germany, 2018. (arXiv:1808.01097)
    """

    def __init__(self, n_subsets=3, method='default', density_t=0.6, verbose=False,
                 dim_reduce=256, batch_max=500000, random_state=None, calc_auxiliary=False):
        self.n_subsets = n_subsets
        self.method = method
        self.density_t = density_t
        self.verbose = verbose
        self.output_labels = None
        self.random_state = random_state
        self.dim_reduce = dim_reduce
        self.batch_max = batch_max
        self.calc_auxiliary = calc_auxiliary

    def fit(self, X, y):
        """Perform curriculum clustering.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array (the embedding space of the samples.)

        y : array-like, size=[n_samples]
            category labels (the curriculum will be learned for each of these categories into N subsets)

        """
        X = check_array(X, accept_sparse='csr')
        check_consistent_length(X, y)
        self.output_labels, _ = cluster_curriculum_subsets(X, y, **self.get_params())
        return self

    def fit_predict(self, X, y=None):
        """Performs curriculum clustering on X and returns clustered labels (subsets).

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array (the embedding space of the samples.)

        y : array-like, size=[n_samples]
            category labels (the curriculum will be learned for each of these categories into N subsets)

        Returns
        -------
        all_clustered_labels : array [n_samples]
            Clustered labels for each point. Labels are integers ordered from most simple to most complex.
            E.g. if curriculum subsets=3, then label=0 is simplest, labels=1 is harder, and label=n is hardest.

        auxiliary_info : list
            If calc_auxiliary is set to True, this list contains collected auxiliary information
            during the clustering process, including delta centers, which can be useful for visualization.
        """
        self.fit(X, y)
        return self.output_labels
