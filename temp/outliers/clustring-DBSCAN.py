import pickle
import numpy as np

print("started exec")

with open("bak/snow-man/point-cloud.pkl", 'rb') as f:
    points_cloud: np.ndarray = pickle.load(f)

print("started processing....")
from sklearn.cluster import DBSCAN
# Cluster the points using DBSCAN algorithm
dbscan = DBSCAN(eps=0.5, min_samples=10).fit(points_cloud)

print("Doneeeee")

# Get the cluster labels for each point
labels = dbscan.labels_

# Get the indices of the core points (i.e., points that are part of a dense region)
core_indices = np.where(labels != -1)[0]

# Get the coordinates of the core points
core_points = points_cloud[core_indices, :]

# Get the indices of the outlier points (i.e., points that are not part of any dense region)
outlier_indices = np.where(labels == -1)[0]

# Get the coordinates of the outlier points
outlier_points = points_cloud[outlier_indices, :]

# Print the number of clusters and the number of outlier points
print("Number of clusters:", len(np.unique(labels)) - 1)
print("Number of outlier points:", len(outlier_indices))