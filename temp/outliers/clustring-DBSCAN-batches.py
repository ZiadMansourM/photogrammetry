import pickle
import numpy as np
from sklearn.cluster import DBSCAN

print("Started exec")

with open("bak/snow-man/point-cloud.pkl", 'rb') as f:
    points_cloud: np.ndarray = pickle.load(f)

print("Started processing...")
batch_size = 100_000
n_points = len(points_cloud)
core_points = []
outlier_points = []
for i in range(0, n_points, batch_size):
    print(f"Started processing batch number {i}...")
    batch = points_cloud[i:i+batch_size]
    dbscan = DBSCAN(eps=0.5, min_samples=10).fit(batch)
    labels = dbscan.labels_
    core_indices = np.where(labels != -1)[0]
    core_points_batch = batch[core_indices, :]
    core_points.append(core_points_batch)
    outlier_indices = np.where(labels == -1)[0]
    outlier_points_batch = batch[outlier_indices, :]
    outlier_points.append(outlier_points_batch)
    
core_points = np.vstack(core_points)
outlier_points = np.vstack(outlier_points)

print("Done")
print("Number of clusters:", len(np.unique(labels)) - 1)
print("Number of outlier points:", len(outlier_points))
