from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

# Load the mandrill image as an NxNx3 array. Values range from 0.0 to 255.0.
mandrill = plt.imread('mandrill.png')[:,:,:3]
if mandrill.max() <= 1.0:
    mandrill *= 255.0
mandrill = mandrill.astype(float)
N = int(mandrill.shape[0])
M = 2
K = 64

# Store each MxM block of the image as a row vector of X
X = np.zeros((N**2//M**2, 3*M**2))
for i in range(N//M):
    for j in range(N//M):
        X[i*N//M+j,:] = mandrill[i*M:(i+1)*M,j*M:(j+1)*M,:].reshape(3*M**2)

# K-Means
def k_means(X, K, max_iterations=100, tolerance=1e-4):
    np.random.seed(0)
    initial_indicies = np.random.choice(len(X), K, replace=False)
    centroids = X[initial_indicies]
    objective_function_values = []

    for iteration in range(max_iterations):
        distances = cdist(X, centroids)
        labels = np.argmin(distances, axis=1)
        updated_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(K)
        ])

        objective_function = np.sum((X - centroids[labels])**2)
        objective_function_values.append(objective_function)

        if (np.all(np.linalg.norm(updated_centroids - centroids, axis=1) < tolerance)):
            print(f"Converged in {iteration+1} iterations")
            break

        centroids = updated_centroids

    return centroids, labels, objective_function_values

centroids, labels, objective_function_values = k_means(X, K)

# Reconstruct compressed image
def reconstruct_image(centroids, labels, N, M):
    blocks_per_row = N // M
    compressed = np.zeros((N, N, 3))

    for idx, label in enumerate(labels):
        i = idx // blocks_per_row
        j = idx % blocks_per_row
        block = centroids[label].reshape(M, M, 3)
        compressed[i*M:(i+1)*M,j*M:(j+1)*M,:] = block

    return compressed

compressed = reconstruct_image(centroids, labels, N, M)
difference = np.abs(mandrill - compressed)

# Calculate relative mean absolute error
rmae = np.mean(difference) / 255
print(f"Relative Mean Absolute Error: {rmae}")

# Compute difference between compressed and original images
difference_image = np.clip(difference + 128, 0, 255).astype(np.uint8)

# Plot K-means object function value vs iterations
plt.figure()
plt.plot(objective_function_values)
plt.xlabel("Iteration")
plt.ylabel("K-means objective function value")
plt.title("Objective function value v.s. Iterations")
plt.grid(True)
plt.show()

# Show original image
plt.figure()
plt.imshow(mandrill.astype(np.uint8))
plt.title("Original Image")
plt.axis("off")
plt.show()

# Show compressed image
plt.figure()
plt.imshow(compressed.astype(np.uint8))
plt.title("Compressed Image")
plt.axis("off")
plt.show()

# Show difference of the two images (compressed and original)
plt.figure()
plt.imshow(difference_image)
plt.title("Difference Image")
plt.axis("off")
plt.show()
