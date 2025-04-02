import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

file_name = 'data2.csv'

# Load data
data = []
with open(file_name) as file:
    reader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        data.append(row)
data = np.array(data)

# K-Means parameters
K = 2
max_iterations = 100
tolerance = 1e-4

# Randomly pick points for initial centroids
np.random.seed(0)
initial_indicies = np.random.choice(len(data), K, replace=False)
centroids = data[initial_indicies]

# Run K-Means
for iteration in range(max_iterations):
    distances = cdist(data, centroids)
    labels = np.argmin(distances, axis=1)
    updated_centroids = np.array([data[labels==i].mean(axis=0) for i in range(K)])

    if (np.all(np.linalg.norm(updated_centroids - centroids, axis=1) < tolerance)):
        print(f"Converged in {iteration+1} iterations")
        break

    centroids = updated_centroids

# Plot
colors = ['blue', 'red']
for i in range(K):
    cluster_points = data[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}')
    plt.scatter(centroids[i, 0], centroids[i, 1], color=colors[i], edgecolors='black', marker='*', s=200, label=f'Centroid {i+1}')

plt.title('K-Means clustering on ' + file_name)
plt.legend()
plt.show()
