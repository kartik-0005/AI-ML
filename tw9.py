#TW9

import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def initialize_centroids(X, k):
    # Randomly choose k data points as initial centroids
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) == 0: # If a cluster is empty, reinitialize its centroid randomly
            new_centroid = X[np.random.choice(X.shape[0])]
        else:
            new_centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.all(np.abs(new_centroids - centroids) <= tol):
            break
        centroids = new_centroids
    return centroids, clusters

# Generate sample data
np.random.seed(42)
X = np.vstack([np.random.randn(100, 2) + np.array([3, 3]),
               np.random.randn(100, 2) + np.array([-3, -3]),
               np.random.randn(100, 2) + np.array([-3, 3]),
               np.random.randn(100, 2) + np.array([3, -3])])

# Apply K-means algorithm
k = 4
centroids, clusters = kmeans(X, k)

# # Plotting the sample data
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1])
# plt.title("Sample Data for Clustering")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

# Plotting K-means clustering result
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b', 'y']
for i in range(k):
    cluster_points = X[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids', marker='X')
plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

print("Final Centroids:")
print(centroids)
