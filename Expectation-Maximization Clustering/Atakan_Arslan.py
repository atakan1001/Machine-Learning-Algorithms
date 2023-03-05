#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:07:23 2023

@author: atakanarslan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
from scipy.stats import multivariate_normal

initial_centroids = np.genfromtxt("hw05_initial_centroids.csv", delimiter=",")
data_set = np.genfromtxt("hw05_data_set.csv", delimiter=",")

initial_means = np.array([[0,5.5],[-5.5,0],[0,0],[5.5,0],[0,-5.5]])
initial_covariances = np.array([[[4.8,0],[0,0.4]],[[0.4,0],[0,2.8]],[[2.4,0],[0,2.4]],[[0.4,0],[0,2.8]],[[4.8,0],[0,0.4]]])
class_sizes = np.array([275,150,150,150,275])

K = class_sizes.shape[0];
N = data_set.shape[0]


# Extract x1 and x2 columns from the data set
x1 = data_set[:, 0]
x2 = data_set[:, 1]

# Create a figure and plot the data points

fig, ax = plt.subplots(figsize = (10, 10))
ax.plot(x1, x2, "k.", markersize = 12)

# Add labels to the axes
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")

# Show the plot
plt.show()

 
def update(centroids, X):
    num_centroids = centroids.shape[0]
    num_samples = X.shape[0]
    memberships = np.zeros(num_samples)
    for i in range(num_samples):
        min_distance = float('inf')
        for j in range(num_centroids):
            distance = np.linalg.norm(centroids[j] - X[i])
            if distance < min_distance:
                min_distance = distance
                memberships[i] = j
    return memberships


def eStep(Fi, data_set):
    centroids = Fi[0]
    class_covariances = Fi[1]
    probabilities = Fi[2]
    num_classes = len(centroids)
    num_samples = data_set.shape[0]

    class_probabilities = np.zeros((num_samples, num_classes))
    for c in range(num_classes):
        current_gaussian = multivariate_normal.pdf(data_set, mean=centroids[c, :], cov=class_covariances[c, :, :])
        class_probabilities[:, c] = current_gaussian * probabilities[c]

    class_probabilities /= np.sum(class_probabilities, axis=1, keepdims=True)
    return class_probabilities


def mStep(X, memberships_probabilities):
    K = memberships_probabilities.shape[1]
    N = memberships_probabilities.shape[0]

    centroids = np.zeros((K, X.shape[1]))
    class_covariances = np.zeros((K, X.shape[1], X.shape[1]))
    probabilities = np.zeros(K)

    for k in range(K):
        memberships_k = memberships_probabilities[:,k]
        N_k = np.sum(memberships_k)
        probabilities[k] = N_k / N
        centroids[k] = np.sum(memberships_k[:, np.newaxis] * X, axis=0) / N_k
        centered_X = X - centroids[k]
        class_covariances[k] = centered_X.T @ (memberships_k[:, np.newaxis] * centered_X) / N_k

    return centroids, class_covariances, probabilities


def plot_current_state(centroids, memberships, X, class_covariances, initial_means, initial_covariances):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"])

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    x1_interval = np.linspace(-8, +8, 1601)
    x2_interval = np.linspace(-8, +8, 1601)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    intervals = np.dstack((x1_grid, x2_grid))

    for k in range(K):
        EM_data_set = multivariate_normal(centroids[k], class_covariances[k]).pdf(intervals)
        plt.contour(x1_grid, x2_grid, EM_data_set, colors=cluster_colors[k], levels=[0.05])
        given_data_set = multivariate_normal(initial_means[k], initial_covariances[k]).pdf(intervals)
        plt.contour(x1_grid, x2_grid, given_data_set, linestyles='dashed', levels=[0.05], colors='k')

    if memberships is None:
        plt.plot(X[:, 0], X[:, 1], "k.", markersize=10)
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
                     color=cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], markersize=12,
                 markerfacecolor=cluster_colors[c], markeredgecolor="black")








centroids = initial_centroids
memberships = update(centroids, data_set)


class_covariances = []

for k in range(K):
    class_data = data_set[memberships == k, :]
    centered_data = class_data - centroids[k, :]
    covariance = np.dot(centered_data.T, centered_data) / class_sizes[k]
    class_covariances.append(covariance)
class_covariances = np.array(class_covariances)

probabilities = (class_sizes / N)

iteration = 1
Fi = []

for i in range(1,101):
    # print("Iteration#{}:".format(iteration))
    print("Iteration ",i)
    Fi = (centroids, class_covariances, probabilities)
    membership_probabilities = eStep(Fi, data_set)

    Fi = mStep(data_set, membership_probabilities)
    centroids = Fi[0]
    class_covariances = Fi[1]
    probabilities = Fi[2]

    iteration = iteration + 1
    
print("Means")

print(centroids)
memberships = np.argmax(membership_probabilities, axis=1)
plt.figure(figsize = (10, 10))
plot_current_state(centroids, memberships, data_set, class_covariances, initial_means, initial_covariances)
plt.show()