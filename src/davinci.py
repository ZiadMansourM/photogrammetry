import logging
import os
import pickle
import time
from typing import Final, List, Tuple

import cv2 as OpenCV
import joblib
import numpy as np
import trimesh
from numpy.linalg import norm
from scipy.cluster.vq import kmeans, vq
from scipy.spatial import Delaunay


import utils
import calibration

logging.basicConfig(
    filename='tune.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def timeit(func):
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


@timeit
def get_images_keypoints(GRAY_IMAGES, SIFT):
    keypoints, descriptors = [], []
    for i in range(len(GRAY_IMAGES)):
        keyPoint, descriptor = SIFT.detectAndCompute(GRAY_IMAGES[i], None)
        keypoints.append(np.array(keyPoint))
        descriptors.append(np.array(descriptor))
    return keypoints, descriptors


def convert_keypoints_to_tuples(
        keypoints: List[List[OpenCV.KeyPoint]]
        ) -> List[List[Tuple[float, float, float, float, int, int]]]:
    """
    Converts a list of lists of cv2.KeyPoint objects to a list of lists of tuples, where each tuple contains the point
    coordinates, size, angle, response, octave, and class ID of a keypoint.
    That is done because cv2.KeyPoint objects are not serializable. And we aim to use pickle to save the keypoints.

    Args:
        keypoints: A list of lists of cv2.KeyPoint objects.

    Returns:
        A list of lists of tuples, where each tuple contains the point coordinates, size, angle, response, octave, and
        class ID of a keypoint.

    """
    print(f"{keypoints[0]=}")
    return [
        [
            (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) 
            for kp in kp_list
        ] for kp_list in keypoints 
    ]

@timeit
def load_sift_features(path: str):
        with open(path, 'rb') as f:
            keypoints_tuple, descriptors = pickle.load(f)
        keypoints = convert_tuples_to_keypoints(keypoints_tuple)
        return keypoints, descriptors

@timeit
def dump_sift_features(path: str, keypoints, descriptors):
    keypoints_tuples = convert_keypoints_to_tuples(keypoints)
    with open(path, 'wb') as f:
        pickle.dump((keypoints_tuples, descriptors), f)

def convert_tuples_to_keypoints(
        keypoints_tuple: List[List[Tuple[float, float, float, float, int, int]]]
        ) -> List[List[OpenCV.KeyPoint]]:
    """
    Converts a list of lists of tuples containing point coordinates, size, angle, response, octave, and class ID of a
    keypoint to a list of lists of cv2.KeyPoint objects.
    That is done because cv2.KeyPoint objects are not serializable. And we aim to use pickle to save/load the keypoints.

    Args:
        keypoints_tuple: A list of lists of tuples containing point coordinates, size, angle, response, octave, and
        class ID of a keypoint.

    Returns:
        A list of lists of cv2.KeyPoint objects.

    """
    return [
        [
            OpenCV.KeyPoint(x=kp[0][0], y=kp[0][1], size=kp[1], angle=kp[2], response=kp[3], octave=kp[4], class_id=kp[5])
            for kp in kp_array
        ] for kp_array in keypoints_tuple
    ]



@timeit
def match_images(descriptors):
    all_descriptors = np.concatenate(descriptors)
    CLUSTER_COUNT: Final = 400
    ITER: Final = 2
    centroids, variance = kmeans(all_descriptors, CLUSTER_COUNT, ITER)
    return centroids, variance, CLUSTER_COUNT

@timeit
def load_image_matching(path: str):
    return joblib.load(path)

@timeit
def dump_image_matching(path: str, CLUSTER_COUNT, centroids):
    joblib.dump((CLUSTER_COUNT, centroids), path, compress = 3)


def get_visual_words(descriptors, centroids):
    visual_words = []
    for descriptor in descriptors:
        words, _ = vq(descriptor, centroids)
        visual_words.append(words)
    return visual_words


def get_frequency_vectors(visual_words, CLUSTER_COUNT):
    frequency_vectors = []
    for img_words in visual_words:
        histogram = np.zeros(CLUSTER_COUNT)
        for word in img_words:
            histogram[word] += 1
        frequency_vectors.append(histogram)
    return np.stack(frequency_vectors)


def get_tf_idf(frequency_vectors, IMAGES_COUNT):
    df = np.sum(frequency_vectors > 0, axis = 0)
    idf = np.log(IMAGES_COUNT/df)
    return frequency_vectors * idf


def search_matches(i, top_clusters, tf_idf):
    b = tf_idf
    a = tf_idf[i]
    b_subset = b[:tf_idf.shape[0]]
    cosine_similarity = np.dot(a, b_subset.T)/(norm(a) * norm(b_subset, axis=1))
    idx = np.argsort(-cosine_similarity)[:top_clusters]
    return list(zip(idx, cosine_similarity[idx]))

@timeit
def get_matches_ids(descriptors, centroids, gray_images, images):
    visual_words = get_visual_words(descriptors, centroids)
    frequency_vectors = get_frequency_vectors(visual_words, centroids.shape[0])
    """ tf_idf: Term Frequency-Inverse Document Frequency """
    tf_idf = get_tf_idf(frequency_vectors, len(gray_images))
    return [
        search_matches(i, 10, tf_idf) 
        for i in range(len(images))
    ]

def feature_matching(img_id_one, img_id_two, descriptors):
    matcher = OpenCV.BFMatcher()
    return matcher.match(descriptors[img_id_one], descriptors[img_id_two])


@timeit
def data_feature_matching(matchesIDs, Sdescriptors):
    num_images = len(Sdescriptors)
    checked = np.zeros((num_images, num_images), dtype=int)
    feature_matches_list = []
    for imageID in range(len(matchesIDs)):
        logging.info(f"---------- START Matches for: {str(imageID)}")
        for i, (matchedID, probability) in enumerate(matchesIDs[imageID]):
            if ((checked[imageID][matchedID] == 0 or checked[matchedID][imageID] == 0) and imageID != matchedID and probability > 0.93):
                start_time = time.time()
                feature_matches_list.append([imageID, matchedID, feature_matching(imageID, matchedID, Sdescriptors)])
                checked[imageID][matchedID], checked[matchedID][imageID] = 1, 1
                logging.info(f"done [{i}/{len(matchesIDs[imageID])}] in {(time.time() - start_time):.4f}: {str(imageID)} - {str(matchedID)}")
        # Flush the log file force write to disk
        logging.shutdown()
    return feature_matches_list

def convert_matches_to_dicts(matches):
    match_dicts = []
    for match in matches:
        match_dict = {'queryIdx': match.queryIdx, 'trainIdx': match.trainIdx, 'distance': match.distance}
        match_dicts.append(match_dict)
    return match_dicts

@timeit
def load_feature_matching(path: str):
    with open(path, 'rb') as f:
        feature_matches_dicts = pickle.load(f)
    feature_matches = []
    for match_dict in feature_matches_dicts:
        matches = [
            OpenCV.DMatch(
                match['queryIdx'], 
                match['trainIdx'], 
                match['distance']
            ) for match in match_dict[2]
        ]
        feature_matches.append([match_dict[0], match_dict[1], matches])
    return feature_matches


@timeit
def dump_feature_matching(path: str, feature_matches):
    matches_dicts = [
        [
            match[0],
            match[1], 
            convert_matches_to_dicts(match[2])
        ] for match in feature_matches
    ]
    with open(path, 'wb') as f:
        pickle.dump(matches_dicts, f)


# def triangulate_points(img_points, P):
#     A = []
#     for point in img_points:
#         x, y = point[0], point[1]
#         p1 = P[0][0] - P[0][2]*x
#         p2 = P[0][1] - P[0][2]*y
#         p3 = P[1][0] - P[1][2]*x
#         p4 = P[1][1] - P[1][2]*y
#         A.append([p1, p2, p3, p4])
#     A = np.array(A)
#     _, _, vt = np.linalg.svd(A)
#     v = vt[-1]
#     return v/v[-1]


# def triangulate(i, j, matches, P_matrices):
#     img_points_i, img_points_j = [], []
#     for match in matches:
#         img_points_i.append(keypoints[i][match.queryIdx].pt)
#         img_points_j.append(keypoints[j][match.trainIdx].pt)
#     img_points_i, img_points_j = np.array(img_points_i), np.array(img_points_j)
#     world_points = []
#     for p1, p2 in zip(img_points_i, img_points_j):
#         X = triangulate_points([p1, p2], [P_matrices[i], P_matrices[j]])
#         world_points.append(X)
#     return np.array(world_points)


# def triangulation(feature_matches, P_matrices):
#     point_cloud = []
#     for i, j, matches in feature_matches:
#         world_points = triangulate(i, j, matches, P_matrices)
#         point_cloud.append(world_points)
#     return np.concatenate(point_cloud, axis=0)

@timeit
def triangulatePoints(P1, P2, pts1, pts2):
    """
    Triangulates the given matching points from two images using the given camera matrices.

    Parameters:
    P1 (numpy.ndarray): 3x4 camera matrix of the first image.
    P2 (numpy.ndarray): 3x4 camera matrix of the second image.
    pts1 (numpy.ndarray): Nx2 matrix containing the coordinates of matching points in the first image.
    pts2 (numpy.ndarray): Nx2 matrix containing the coordinates of matching points in the second image.

    Returns:
    numpy.ndarray: Nx3 matrix containing the triangulated 3D points.
    """
    pts4D = OpenCV.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts4D /= pts4D[3]
    return pts4D[:3].T


@timeit
def generate_point_cloud(feature_matches_list, K_matrix):
    """
    Generates a cloud of 3D points using triangulation from feature matches and camera calibration matrix.

    Parameters:
    feature_matches_list (list): List of feature matches between images.
    K_matrix (numpy.ndarray): 3x3 camera calibration matrix.

    Returns:
    numpy.ndarray: Nx3 matrix containing the cloud of 3D points.
    """
    point_cloud = []
    for match in feature_matches_list:
        img1, img2, matches = match
        pts1 = np.float32([kp.pt for kp in matches[0]])
        pts2 = np.float32([kp.pt for kp in matches[1]])
        E, _ = OpenCV.findEssentialMat(pts1, pts2, K_matrix)
        R1, R2, t = OpenCV.decomposeEssentialMat(E)

        for i in range(len(R1)):
            P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = np.hstack((R1[i], t))
            pts_3d = triangulatePoints(K_matrix.dot(P1), K_matrix.dot(P2), pts1, pts2)
            point_cloud.append(pts_3d)
    return np.concatenate(point_cloud, axis=0)


@timeit
def davinci_run(image_set_name):
    print("Welcome ScanMate...")

    # 1. Load Images
    images = utils.read_images(f"images/{image_set_name}")
    print("Images loaded successfully")
    gray_images = utils.rgp_to_gray(images)
    print("Gray Images created successfully")
    
    # 2. Feature Extraction: SIFT
    sift = OpenCV.SIFT_create()
    if os.path.isfile(f"bak/{image_set_name}/sift-features.pkl"):
        print(f"File [bak/{image_set_name}/sift-features.pkl] exists")
        keypoints, descriptors = load_sift_features(f"bak/{image_set_name}/sift-features.pkl")
    else:
        print("File [bak/{image_set_name}/sift-features.pkl] DO NOT exists")
        keypoints, descriptors = get_images_keypoints(gray_images, sift)
        dump_sift_features(f"bak/{image_set_name}/sift-features.pkl", keypoints, descriptors)
    print("Feature Extraction: SIFT DONE...")

    # 3. Image Matching
    if os.path.isfile(f"bak/{image_set_name}/image-matching-centroids.pkl"):
        print(f"File [bak/{image_set_name}/image-matching-centroids.pkl] exists")
        CLUSTER_COUNT, centroids = load_image_matching(f"bak/{image_set_name}/image-matching-centroids.pkl")
    else:
        print(f"File [bak/{image_set_name}/image-matching-centroids.pkl] DO NOT exists")
        centroids, variance, CLUSTER_COUNT = match_images(descriptors)
        dump_image_matching(f"bak/{image_set_name}/image-matching-centroids.pkl", CLUSTER_COUNT, centroids)
    matches_ids = get_matches_ids(descriptors, centroids, gray_images, images)
    print("Done Image Matching Step...")

    # 4. Feature Matching
    if os.path.isfile(f"bak/{image_set_name}/feature-matching-output.pkl"):
        print(f"File [bak/{image_set_name}/feature-matching-output.pkl] exists")
        feature_matches = load_feature_matching(f"bak/{image_set_name}/feature-matching-output.pkl")
    else:
        print("File [bak/{image_set_name}/feature-matching-output.pkl] Do NOT exists")
        logging.info('----> Processing {image_set_name}...')
        feature_matches = data_feature_matching(matches_ids, descriptors)
        dump_feature_matching(f"bak/{image_set_name}/feature-matching-output.pkl", feature_matches)
    
    # 5. Camera Calibration
    print("Camera Calibration starts ....")
    K_matrix = calibration.calibrate_camera(f"bak/{image_set_name}/calibration.pkl")
    
    # 6. Triangulation (3D reconstruction)
    print("Triangulation starts ....")
    points_cloud = generate_point_cloud(feature_matches, K_matrix)
    np.savetxt("points_cloud.txt", points_cloud)
    
    # 7. generate mesh
    print("Generate mesh ....")
    tri = Delaunay(points_cloud)
    mesh = trimesh.Trimesh(points_cloud, tri.simplices)
    mesh = mesh.simplify()

    # 8. output .obj, .stl and .ply files
    print("Generate mesh ....")
    mesh.export(f"output/{image_set_name}/snow_man.obj")
    mesh.export(f"output/{image_set_name}/snow_man.stl")
    mesh.export(f"output/{image_set_name}/snow_man.ply")

if __name__ == '__main__':
    davinci_run("snow-man")