import time
import os

import cv2 as OpenCV
from matplotlib import pyplot as plt

def timeit_utils(func):
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

def display_images(image, title = None) -> None:
    if image.ndim == 2:
        plt.gray()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

def read_images_rbg(imagePath: str):
    return OpenCV.cvtColor(OpenCV.imread(imagePath), OpenCV.COLOR_BGR2RGB)

@timeit_utils
def rgp_to_gray(images):
    return [OpenCV.cvtColor(image, OpenCV.COLOR_RGB2GRAY) for image in images]

@timeit_utils
def read_images(folderPath):
    files = sorted(os.listdir(folderPath))
    return [
        read_images_rbg(f"{folderPath}/{file}")
        for file in files
        if ".jpg" in file
    ]