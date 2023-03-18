# import cv2
# import glob
# from collections import Counter

# image_dir = 'images/snow-man'
# sizes = {}

# # Loop through all images in the directory and get their sizes
# images = glob.glob(f"{image_dir}/*.jpg")
# for fname in images:
#     img = cv2.imread(fname)
#     size = tuple(img.shape[:2])
#     if size not in sizes:
#         sizes[size] = 1
#     else:
#         sizes[size] += 1

# # Count the frequency of each size
# freq = Counter(sizes)

# # Print the sizes and their frequency
# for size, count in freq.items():
#     print(f'Size: {size}, Frequency: {count}')

import cv2 as OpenCV
import glob

def get_image_sizes(directory):
    sizes = {}
    images = glob.glob(f"{directory}/*.jpg")
    
    for fname in images:
        img = OpenCV.imread(fname)
        size = tuple(img.shape[:2])
        if size not in sizes:
            sizes[size] = [fname]
        else:
            sizes[size].append(fname)

    return sizes
image_dir = 'images/snow-man'
sizes_dict = get_image_sizes(image_dir)

for size, image_list in sizes_dict.items():
    print(f'Size: {size}, Frequency: {len(image_list)}')
