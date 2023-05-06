import numpy as np
import matplotlib.pyplot as plt

# Generate point cloud
point_cloud = np.random.rand(10000, 3)

# Define scaling factor
scale_factor = 2.0

# Scale the point cloud
scaled_point_cloud = point_cloud * scale_factor

# Plot histogram of point_cloud
plt.hist(point_cloud.flatten(), bins=50, alpha=0.5, label='Before Scaling')
# Plot histogram of scaled_point_cloud
# plt.hist(scaled_point_cloud.flatten(), bins=50, alpha=0.5, label='After Scaling')

# Add titles and labels to the plot
plt.title('Point Cloud Histogram')
plt.xlabel('Point Value')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

# Show the plot
plt.show()
