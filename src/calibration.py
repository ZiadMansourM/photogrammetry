import time
import numpy as np
import cv2
import glob

from exceptions import CalibrationError

# logging.basicConfig(filename='logs/davinci.log', level=logging.INFO)

# def calibrate_camera(image_set_name, pattern_size, pattern_type='chessboard'):
#     # Define the dimensions of the calibration pattern or geometry
#     if not pattern_size:
#         pattern_size = (9, 6)

#     # Define the object points of the calibration pattern or geometry
#     objp = np.zeros((np.prod(pattern_size), 3), np.float32)
#     objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

#     # Define the arrays to store object points and image points for all images
#     objpoints = []
#     imgpoints = []

#     # Define the path to the directory containing the images
#     img_dir = f'images/{image_set_name}'

#     # Get a list of the image filenames in the directory
#     img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

#     # Read each image, detect calibration pattern or geometry, and add object and image points to the arrays
#     for img_name in img_names:
#         # Read the image
#         img = cv2.imread(os.path.join(img_dir, img_name))

#         # Convert the image to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Find the corners of the calibration pattern or geometry
#         ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

#         # If corners are found, add the object and image points to the arrays
#         if ret == True:
#             objpoints.append(objp)
#             imgpoints.append(corners)

#             if __name__ == '__main__':
#                 # Draw and display the corners on the image
#                 img = cv2.drawChessboardCorners(img, pattern_size, corners, ret)
#                 cv2.imshow('img', img)
#                 cv2.waitKey(500)

#     # Perform camera calibration using the object and image points
#     ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#     if not os.path.exists(f'parameters/{image_set_name}'):
#         os.makedirs(f'parameters/{image_set_name}')

#     if not ret:
#         logging.error(f'Camera calibration failed {image_set_name}!')
#         raise CalibrationError('Camera calibration failed')

#     logging.info(f'Camera calibration successful {image_set_name}')
#     # Print the intrinsic parameters of the camera
#     logging.info(f'Camera matrix {image_set_name}:\n{mtx}')
#     logging.info(f'Distortion coefficients {image_set_name}:\n{dist}')
#     # Save the intrinsic parameters as a text file
#     np.savetxt(f'parameters/{image_set_name}/intrinsics.txt', mtx)
#     np.savetxt(f'parameters/{image_set_name}/distortion.txt', dist)
#     return mtx, dist

def timeit_helper(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        with open("log.txt", "a") as f:
            f.write(
                f"Function '{func.__name__}' took {end_time - start_time} seconds to execute.\n"
            )
        return result
    return wrapper

@timeit_helper
def calibrate_camera(image_set_name, pattern_type='chessboard'):
    # Define the size of the chessboard pattern used for calibration
    pattern_size = (6, 9)

    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    # Create empty arrays to store object points and image points from all the calibration images
    objpoints = []
    imgpoints = []

    # Load calibration images from the "images/snow-man" directory
    images = glob.glob("images/snow-man/*.jpg")

    # Loop through all images
    for fname in images:
        # Load the image
        img = cv2.imread(fname)
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners in the image
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # If the corners are found, add object points and image points to the lists
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Calibrate the camera and find the camera matrix K
    ret, K_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        raise CalibrationError('Camera calibration failed')

    return K_matrix