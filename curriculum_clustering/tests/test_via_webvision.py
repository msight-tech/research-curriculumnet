# -*- coding: utf-8 -*-
"""
Test curriculum_clustering via a subset of the WebVision dataset v1.0

You can find more information about the dataset here:
https://www.vision.ee.ethz.ch/webvision/2017/

The testing dataset contains extracted features and labels for the first 10 classes of the WebVision dataset 1.0

The class names are local to this repository, but since the features are a large file, it has been made available here:
https://sai-pub.s3.cn-north-1.amazonaws.com.cn/malong-research/curriculumnet/webvision_cls0-9.npy

The test will download the file automatically if it is not available at test-data/input/webvision_cls0-9.npy

"""

# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import os
import shutil
import tempfile
import urllib
import numpy as np
from curriculum_clustering import CurriculumClustering


def test_curriculum_cluster():
    X, y, metadata = load_webvision_data()
    cc = CurriculumClustering(n_subsets=3, verbose=True, random_state=0)
    cc.fit(X, y)
    verify_webvision_expected_clusters(labels=cc.output_labels, n_subsets=cc.n_subsets, metadata=metadata)


def load_webvision_data():
    # Load a subset of WebVision data (classes 0 to 9, features and names)

    # X: features
    features_file = 'test-data/input/webvision_cls0-9.npy'
    download_data_if_not_local(features_file)
    X = np.load(features_file)

    # y: labels
    cluster_list = 'test-data/input/webvision_cls0-9.txt'  # imagenet train list
    with open(cluster_list) as f:
        metadata = [x.strip().split(' ') for x in f]
    y = [int(item[1]) for item in metadata]

    return X, y, metadata


def verify_webvision_expected_clusters(labels, n_subsets, metadata):
    # Create a place to write results for possible forensics
    test_dir = tempfile.mkdtemp()

    # combine back to labels with metadata using the indexes from the clustering result
    # Create output files - depending on the levels, 1.txt would be 'clean', 2.txt is medium, and 3.txt is dirty
    clustered_by_levels = [list() for _ in xrange(n_subsets)]
    for idx, _ in enumerate(metadata):
        clustered_by_levels[labels[idx]].append(idx)
    for idx, level_output in enumerate(clustered_by_levels):
        with open("{}/{}.txt".format(test_dir, idx + 1), 'w') as f:
            for i in level_output:
                f.write("{}\n".format(str.join(' ', metadata[i])))

    # Verify matches expectation
    for i in range(1, n_subsets + 1):
        with open('{}/{}.txt'.format(test_dir, i), 'r') as file1:
            with open('test-data/output-expected/{}.txt'.format(i), 'r') as file2:
                diff = set(file1).difference(file2)

        if len(diff) != 0:
            with open('{}/err.txt'.format(test_dir), 'w') as file_out:
                for label_lines in diff:
                    file_out.write(label_lines)
            raise RuntimeError(
                "ERROR: Found {} differences in expected output file {} See {}/err.txt.".format(len(diff), i, test_dir))

    # Clean up
    shutil.rmtree(test_dir)
    print "Test is successful."


def download_data_if_not_local(features_file):
    features_url = "https://sai-pub.s3.cn-north-1.amazonaws.com.cn/malong-research/curriculumnet/webvision_cls0-9.npy"
    expected_features_size = 105447552

    if not os.path.exists(features_file) or os.path.getsize(features_file) != expected_features_size:
        print "This test requires WebVision subset features which haven't been found at {}.".format(features_file)
        print "Retrieving from {} ({}MB). Downloading...".format(features_url, expected_features_size / 1024 / 1024)
        urllib.urlretrieve(features_url, features_file)
        if os.path.getsize(features_file) != expected_features_size:
            raise IOError("Failed to download {}, size mismatch (expected: {} vs retrieved {})".format(
                features_url, expected_features_size, os.path.getsize(features_file)))
        print "Successfully downloaded to {}.".format(features_file)


if __name__ == "__main__":
    test_curriculum_cluster()
