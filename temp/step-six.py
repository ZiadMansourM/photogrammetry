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

# Save the point cloud to a file
# np.savetxt('point_cloud.txt', point_cloud)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])

o3d.visualization.draw_geometries([pcd])

# Save it as a.STL file
# o3d.io.write_point_cloud("point_cloud.ply", pcd)