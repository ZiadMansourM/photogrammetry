import calibration

def davinci_run(image_set_name):
    # 1. Load Images
    # 2. Feature Extraction
    # 3. Image Matching
    # 4. Feature Matching
    # 5. Camera Calibration
    calibration.calibrate(image_set_name, (9, 6), 'chessboard')
    # 6. Triangulation
    # 7. generate 3D point cloud
    # 8. output .obj file

if __name__ == '__main__':
    davinci_run()