import exifread
import numpy as np
import pickle

# Open the image file
image = open(r"D:\CUFE\Year 4\Semester 2\GP\Project\photogrammetry\src\images\snow-man\20201115_115903.jpg", "rb")

# Read the EXIF data
exif = exifread.process_file(image)

# Debug
# print(exif.keys())

# Extract the intrinsic parameters
focal_length = exif['EXIF FocalLength'].values[0]
sensor_width = exif['EXIF ExifImageWidth'].values[0]
sensor_height = exif['EXIF ExifImageLength'].values[0]
principal_point_x = exif['EXIF ExifImageWidth'].values[0] / 2
principal_point_y = exif['EXIF ExifImageLength'].values[0] / 2
# distortion_coefficients = exif['EXIF MakerNote'].values[0]


# Calculate the scaling factor for the K-matrix
scaling_factor = 1.0

# Create the K-matrix
K = np.array([[float(focal_length), 0, principal_point_x],
              [0, float(focal_length), principal_point_y],
              [0, 0, scaling_factor]])

# Print the intrinsic parameters
print("Intrinsic parameters:")
print('Focal Length:', focal_length)
print('Sensor Width:', sensor_width)
print('Sensor Height:', sensor_height)
print('Principal Point (X):', principal_point_x)
print('Principal Point (Y):', principal_point_y)
# print('Distortion Coefficients:', distortion_coefficients)

# Print the K-matrix
print("K-matrix:")
print(K)

# Pickle the K-matrix object
with open('k_matrix.pkl', 'wb') as file:
    pickle.dump(K, file)

print("K-matrix object pickled successfully.")