import numpy as np

user_input = []
for _ in range(6):
    user_input.append(float(input()))

cluster_centers = np.array([(user_input[i], user_input[i + 1]) for i in range(0, len(user_input), 2)])

# Dataset
dataset = np.array([
    (0.22, 0.33), (0.45, 0.76), (0.73, 0.39), (0.25, 0.35), (0.51, 0.69),
    (0.69, 0.42), (0.41, 0.49), (0.15, 0.29), (0.81, 0.32), (0.50, 0.88),
    (0.23, 0.31), (0.77, 0.30), (0.56, 0.75), (0.11, 0.38), (0.81, 0.33),
    (0.59, 0.77), (0.10, 0.89), (0.55, 0.09), (0.75, 0.35), (0.44, 0.55)
])

cluster_dataset_mapping = {i: [] for i in range(len(cluster_centers))}

distances = np.linalg.norm(dataset[:, np.newaxis] - cluster_centers, axis=2) ** 2

has_converged = False
while not has_converged:
    cluster_labels = np.argmin(distances, axis=1)
    for i, label in enumerate(cluster_labels):
        cluster_dataset_mapping[label].append(i)

    new_cluster_centers = np.array([np.mean(dataset[indices], axis=0) for indices in cluster_dataset_mapping.values() if indices])

    if (len(cluster_centers) == len(new_cluster_centers)) and np.allclose(cluster_centers, new_cluster_centers):
        has_converged = True
    else:
        cluster_centers = new_cluster_centers
        cluster_dataset_mapping = {i: [] for i in range(len(cluster_centers))}
        distances = np.linalg.norm(dataset[:, np.newaxis] - cluster_centers, axis=2) ** 2

sse = 0
for cluster_idx, cluster_datapoints_idx in cluster_dataset_mapping.items():
    cluster_datapoints = dataset[cluster_datapoints_idx]
    cluster_sse = np.sum(np.linalg.norm(cluster_datapoints - cluster_centers[cluster_idx], axis=1) ** 2)
    sse += cluster_sse

print(round(sse, 4))