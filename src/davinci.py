import logging
import time
from typing import Final
import cv2 as OpenCV
import numpy as np
from numpy.linalg import norm
from scipy.cluster.vq import kmeans, vq

import utils
import calibration


logging.basicConfig(
    filename='tune.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_images_keypoints(GRAY_IMAGES, SIFT):
    keypoints, descriptors = [], []
    for i in range(len(GRAY_IMAGES)):
        keyPoint, descriptor = SIFT.detectAndCompute(GRAY_IMAGES[i], None)
        keypoints.append(np.array(keyPoint))
        descriptors.append(np.array(descriptor))
    return keypoints, descriptors


def match_images(descriptors):
    all_descriptors = np.concatenate(descriptors)
    CLUSTER_COUNT: Final = 400
    ITER: Final = 2
    centroids, variance = kmeans(all_descriptors, CLUSTER_COUNT, ITER)
    return centroids, variance


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


def feature_matching(img_id_one, img_id_two, descriptors):
    matcher = OpenCV.BFMatcher()
    return matcher.match(descriptors[img_id_one], descriptors[img_id_two])


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


def triangulate_points(img_points, P):
    A = []
    for point in img_points:
        x, y = point[0], point[1]
        p1 = P[0][0] - P[0][2]*x
        p2 = P[0][1] - P[0][2]*y
        p3 = P[1][0] - P[1][2]*x
        p4 = P[1][1] - P[1][2]*y
        A.append([p1, p2, p3, p4])
    A = np.array(A)
    _, _, vt = np.linalg.svd(A)
    v = vt[-1]
    return v/v[-1]


def triangulate(i, j, matches, P_matrices):
    img_points_i, img_points_j = [], []
    for match in matches:
        img_points_i.append(keypoints[i][match.queryIdx].pt)
        img_points_j.append(keypoints[j][match.trainIdx].pt)
    img_points_i, img_points_j = np.array(img_points_i), np.array(img_points_j)
    world_points = []
    for p1, p2 in zip(img_points_i, img_points_j):
        X = triangulate_points([p1, p2], [P_matrices[i], P_matrices[j]])
        world_points.append(X)
    return np.array(world_points)


def triangulation(feature_matches, P_matrices):
    point_cloud = []
    for i, j, matches in feature_matches:
        world_points = triangulate(i, j, matches, P_matrices)
        point_cloud.append(world_points)
    return np.concatenate(point_cloud, axis=0)


def davinci_run(image_set_name):
    # 1. Load Images
    images = utils.read_images("images/{image_set_name}/")
    gray_images = utils.rgp_to_gray(images)
    # 2. Feature Extraction: SIFT
    sift = OpenCV.SIFT_create()
    keypoints, descriptors = get_images_keypoints(gray_images, sift)
    # 3. Image Matching
    centroids, variance = match_images(descriptors)
    visual_words = get_visual_words(descriptors, centroids)
    frequency_vectors = get_frequency_vectors(visual_words, centroids.shape[0])
    """ tf_idf: Term Frequency-Inverse Document Frequency """
    tf_idf = get_tf_idf(frequency_vectors, len(gray_images))
    matches_ids = [search_matches(i, 10, tf_idf) for i in range(len(images))]
    # 4. Feature Matching
    logging.info('----> Processing {image_set_name}...')
    feature_matches = data_feature_matching(matches_ids, descriptors)
    # 5. Camera Calibration
    calibration.calibrate(image_set_name, (9, 6), 'chessboard')
    # TODO: read calibration matrix
    # 6. Triangulation (3D reconstruction)
    cloud_points = []
    # for i, j, matches in feature_matches:
        
    # 7. generate 3D point cloud
    # 8. output .obj file

if __name__ == '__main__':
    davinci_run()