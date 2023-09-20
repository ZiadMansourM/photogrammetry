import cv2 as OpenCV
import numpy as np

from data_structures import Image, Images
from utils import log_to_file, timeit


@timeit
def compute_keypoints_descriptors(images: list[Image], SIFT: OpenCV.SIFT, **kwargs) -> None:
    image_set_name = kwargs['image_set_name']
    data_path = f"../../data/{image_set_name}"
    for img in images.images:
        keypoints: list[OpenCV.KeyPoint]
        descriptors: np.ndarray
        dialated_image = OpenCV.bitwise_and(img.gray_image, img.gray_image, mask=img.mask)
        keypoints, descriptors = SIFT.detectAndCompute(dialated_image, None)
        img.keypoints = keypoints
        img.descriptors = descriptors
        log_to_file(f"{data_path}/logs/tune.log", f"Img({img.img_id}, {img.path}) has {len(img.keypoints)} keypoints and {len(img.descriptors)} descriptors.")
