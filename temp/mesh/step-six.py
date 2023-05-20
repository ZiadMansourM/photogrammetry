import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

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

# Create an Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])

# # Save it as a.STL file
o3d.io.write_point_cloud("cube_point_cloud.ply", pcd)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# Estimate the normals for the point cloud
pcd.estimate_normals()

# Apply the Ball-Pivoting Algorithm to create a mesh
radii = [0.005, 0.01, 0.02, 0.04]
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii)
)

# Save the mesh as an STL file
o3d.io.write_triangle_mesh("cube_point_cloud.stl", bpa_mesh)

# Visualize the point cloud
o3d.visualization.draw_geometries([bpa_mesh])