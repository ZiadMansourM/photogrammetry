import numpy as np
import open3d as o3d

# Define the number of points in the point cloud
N = 1_000_000

# Generate random x, y, and z coordinates between -0.5 and 0.5
x = np.random.rand(N) - 0.5
y = np.random.rand(N) - 0.5
z = np.random.rand(N) - 0.5

# Scale the coordinates to create a cube with sides of length 1 meter
x *= 1.0
y *= 1.0
z *= 1.0

# Combine the coordinates into a single NumPy array
point_cloud = np.column_stack((x, y, z))

# Print max and min values of each axis:
print(f"X: {point_cloud[:,0].min()} to {point_cloud[:,0].max()}")
print(f"Y: {point_cloud[:,1].min()} to {point_cloud[:,1].max()}")
print(f"Z: {point_cloud[:,2].min()} to {point_cloud[:,2].max()}")

import numpy as np
import matplotlib.pyplot as plt

# Generate point cloud
# point_cloud = np.random.rand(10000, 3)

# Define scaling factor
scale_factor = 3.0

# Scale the point cloud
scaled_point_cloud = point_cloud * scale_factor

print(f"X: {scaled_point_cloud[:,0].min()} to {scaled_point_cloud[:,0].max()}")
print(f"Y: {scaled_point_cloud[:,1].min()} to {scaled_point_cloud[:,1].max()}")
print(f"Z: {scaled_point_cloud[:,2].min()} to {scaled_point_cloud[:,2].max()}")

# Plot histogram of points_cloud
plt.hist(point_cloud.flatten(), bins=50, alpha=0.5, label='Before Scaling')
# Plot histogram of scaled_point_cloud
plt.hist(scaled_point_cloud.flatten(), bins=50, alpha=0.5, label='After Scaling')

# Add titles and labels to the plot
plt.title('Point Cloud Histogram')
plt.xlabel('Point Value')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

# Show the plot
plt.show()

# Save the point cloud to a file
# np.savetxt('point_cloud.txt', point_cloud)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])

o3d.visualization.draw_geometries([pcd])

# Save it as a.STL file
o3d.io.write_point_cloud("point_cloud.ply", pcd)