
# Importing Required Modules
from rembg import remove
import cv2 as OpenCV
  
# Store path of the image in the variable input_path
# input_path =  r'E:\Project\photogrammetry\src\data\fountain\images\2.jpg'
input_path =  r'E:\Project\photogrammetry\src\data\snow-man\images\1.jpg'
  
# Store path of the masked image in the variable output_path
mask_path = r'E:\Project\photogrammetry\src\data\fountain\images\mask_2.jpg'
# Processing the image
rgb_image = OpenCV.cvtColor(OpenCV.imread(input_path), OpenCV.COLOR_BGR2RGB)

# Removing the background from the given Image
output = remove(rgb_image)
  
# Convert the output image from memoryview to NumPy array
output = OpenCV.cvtColor(output, OpenCV.COLOR_RGB2GRAY)

# convert all non zero values to 255 
output[output > 0] = 255
OpenCV.imwrite(mask_path, output)
