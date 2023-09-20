import cv2 as OpenCV
import numpy as np

from data_structures import FeatureMatches, Images
from utils import log_to_file, timeit

@timeit
def feature_matching(
        img_one_descriptors: np.ndarray, 
        img_two_descriptors: np.ndarray,
        **kwargs
    ) -> list[OpenCV.DMatch]:
    matcher = OpenCV.BFMatcher(crossCheck=True)
    return matcher.match(img_one_descriptors, img_two_descriptors)

@timeit
def apply_ransac(matches, keypoints1, keypoints2, threshold = 3.0, **kwargs):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = OpenCV.findHomography(src_pts, dst_pts, OpenCV.RANSAC, threshold)
    matches_mask = mask.ravel().tolist()
    return [m for m, keep in zip(matches, matches_mask) if keep]


@timeit
def data_feature_matching(images: Images, **kwargs) -> None:
    import itertools
    image_set_name = kwargs['image_set_name']
    data_path: str = f"../../data/{image_set_name}"
    for key, values in images.similar_images.items():
        log_to_file(f"{data_path}/logs/tune.log", f"Started Feature Match for cluster number {key}:")
        for image, matched_image in itertools.combinations(values, 2):
            feature_matching_output = feature_matching(image.descriptors, matched_image.descriptors, **kwargs)
            ransac_output = apply_ransac(feature_matching_output, image.keypoints, matched_image.keypoints, threshold=150, **kwargs)
            images.feature_matches.append(FeatureMatches(image, matched_image, ransac_output))
            log_to_file(f"{data_path}/logs/tune.log", f"({image.img_id}, {matched_image.img_id}) with {len(ransac_output)} / {len(feature_matching_output)}.")