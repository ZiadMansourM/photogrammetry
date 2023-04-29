import numpy as np
import pyvista as pv

# Define the number of points in the point cloud
N = 10000

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
np.savetxt('point_cloud.txt', point_cloud)

points = pv.PolyData(point_cloud)

# save polydata as STL file
points.save('point_cloud.stl')