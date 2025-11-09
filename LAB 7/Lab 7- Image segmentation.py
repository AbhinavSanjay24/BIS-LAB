import numpy as np
from skimage import data, img_as_float
import matplotlib.pyplot as plt

# Load the color astronaut image
img = img_as_float(data.astronaut())
height, width, channels = img.shape

# Parameters
n_clusters = 4
max_iter = 50
mutation_rate = 0.05

# Initialize random cluster labels
labels = np.random.randint(0, n_clusters, (height, width))
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# ---- Compute cluster means for color image ----
def compute_cluster_means(labels):
    means = np.zeros((n_clusters, channels))
    for k in range(n_clusters):
        mask = (labels == k)
        if np.any(mask):
            means[k] = np.mean(img[mask], axis=0)
    return means

# ---- Compute fitness (intensity + spatial smoothness) ----
def compute_fitness(labels, means):
    # Compute color distance per pixel
    diff = img - means[labels]
    fit_intensity = 1 - np.linalg.norm(diff, axis=2)

    # Smoothness term
    smooth = np.zeros_like(fit_intensity)
    for dy, dx in neighbors:
        shifted = np.roll(np.roll(labels, dy, axis=0), dx, axis=1)
        smooth += (shifted == labels)
    fit_smooth = smooth / len(neighbors)

    # Combine both
    return 0.5 * fit_intensity + 0.5 * fit_smooth

# ---- Show initial state ----
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(labels, cmap='nipy_spectral')
plt.title("Initial Random Labels")
plt.axis('off')
plt.show()

# ---- Evolutionary segmentation loop ----
for it in range(max_iter):
    means = compute_cluster_means(labels)
    fitness = compute_fitness(labels, means)

    dy, dx = neighbors[np.random.randint(len(neighbors))]
    neighbor_labels = np.roll(np.roll(labels, dy, axis=0), dx, axis=1)

    # Crossover
    child_labels = np.where(np.random.rand(height, width) < 0.5, labels, neighbor_labels)

    # Mutation
    mutation_mask = np.random.rand(height, width) < mutation_rate
    child_labels[mutation_mask] = np.random.randint(0, n_clusters, np.count_nonzero(mutation_mask))

    # Evaluate child
    child_means = compute_cluster_means(child_labels)
    child_fitness = compute_fitness(child_labels, child_means)

    # Selection
    labels = np.where(child_fitness > fitness, child_labels, labels)

# ---- Display final segmentation ----
segmented_img = np.zeros_like(img)
means = compute_cluster_means(labels)
for k in range(n_clusters):
    segmented_img[labels == k] = means[k]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_img)
plt.title("Final Segmentation (Color Evolution Result)")
plt.axis('off')
plt.show()