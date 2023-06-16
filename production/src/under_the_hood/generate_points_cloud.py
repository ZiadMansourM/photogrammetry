from collections import Counter

import cv2 as OpenCV
import numpy as np
import open3d as o3d

from data_structures import FeatureMatches, Image, Images
from utils import log_to_file, timeit

def analyize_points(
        points_cloud: np.ndarray, 
        core_points: np.ndarray, 
        outlier_points: np.ndarray
    ) -> None:
    log_to_file("logs/tune.log", "Analysis of X, Y, Z of Points cloud")
    log_to_file(
        "logs/tune.log",
        f"X<{len(points_cloud[:, 0]):,}>: {points_cloud[:, 0].min():,} to {points_cloud[:, 0].max():,}",
    )
    x_counter = Counter(points_cloud[:,0])
    log_to_file("logs/tune.log", f"We have {len(x_counter):,} unique X values")
    log_to_file(
        "logs/tune.log",
        f"Most Common X: {x_counter.most_common(1)}, Least Two Common X: {x_counter.most_common()[:-3:-1]}",
    )
    log_to_file(
        "logs/tune.log",
        "-----------------------------------------------------",
    )
    log_to_file(
        "logs/tune.log",
        f"Y<{len(points_cloud[:, 1]):,}>: {points_cloud[:, 1].min():,} to {points_cloud[:, 1].max():,}",
    )
    y_counter = Counter(points_cloud[:,1])
    log_to_file("logs/tune.log", f"We have {len(y_counter):,} unique Y values")
    log_to_file(
        "logs/tune.log",
        f"Most Common Y: {y_counter.most_common(1)}, Least Two Common Y: {y_counter.most_common()[:-3:-1]}",
    )
    log_to_file(
        "logs/tune.log",
        "-----------------------------------------------------",
    )
    log_to_file(
        "logs/tune.log",
        f"Z<{len(points_cloud[:, 2]):,}>: {points_cloud[:, 2].min():,} to {points_cloud[:, 2].max():,}",
    )
    z_counter = Counter(points_cloud[:,2])
    log_to_file("logs/tune.log", f"We have {len(z_counter):,} unique Z values")
    log_to_file(
        "logs/tune.log",
        f"Most Common Z: {z_counter.most_common(1)}, Least Two Common Y: {z_counter.most_common()[:-3:-1]}",
    )
    log_to_file(
        "logs/tune.log",
        "-----------------------------------------------------",
    )
    log_to_file("logs/tune.log", "Analysis of X, Y, Z of Core Points")
    log_to_file(
        "logs/tune.log",
        f"X<{len(core_points[:, 0]):,}>: {core_points[:, 0].min():,} to {core_points[:, 0].max():,}",
    )
    x_counter = Counter(core_points[:,0])
    log_to_file("logs/tune.log", f"We have {len(x_counter):,} unique X values")
    log_to_file(
        "logs/tune.log",
        f"Most Common X: {x_counter.most_common(1)}, Least Two Common X: {x_counter.most_common()[:-3:-1]}",
    )
    log_to_file(
        "logs/tune.log",
        "-----------------------------------------------------",
    )
    log_to_file(
        "logs/tune.log",
        f"Y<{len(core_points[:, 1]):,}>: {core_points[:, 1].min():,} to {core_points[:, 1].max():,}",
    )
    y_counter = Counter(core_points[:,1])
    log_to_file("logs/tune.log", f"We have {len(y_counter):,} unique Y values")
    log_to_file(
        "logs/tune.log",
        f"Most Common Y: {y_counter.most_common(1)}, Least Two Common Y: {y_counter.most_common()[:-3:-1]}",
    )
    log_to_file(
        "logs/tune.log",
        "-----------------------------------------------------",
    )
    log_to_file(
        "logs/tune.log",
        f"Z<{len(core_points[:, 2]):,}>: {core_points[:, 2].min():,} to {core_points[:, 2].max():,}",
    )
    z_counter = Counter(core_points[:,2])
    log_to_file("logs/tune.log", f"We have {len(z_counter):,} unique Z values")
    log_to_file(
        "logs/tune.log",
        f"Most Common Z: {z_counter.most_common(1)}, Least Two Common Y: {z_counter.most_common()[:-3:-1]}",
    )
    log_to_file(
        "logs/tune.log",
        "-----------------------------------------------------",
    )
    log_to_file("logs/tune.log", "Analysis of X, Y, Z of Outliers Points")
    log_to_file(
        "logs/tune.log",
        f"X<{len(outlier_points[:, 0]):,}>: {outlier_points[:, 0].min():,} to {outlier_points[:, 0].max():,}",
    )
    x_counter = Counter(outlier_points[:,0])
    log_to_file("logs/tune.log", f"We have {len(x_counter):,} unique X values")
    log_to_file(
        "logs/tune.log",
        f"Most Common X: {x_counter.most_common(1)}, Least Two Common X: {x_counter.most_common()[:-3:-1]}",
    )
    log_to_file(
        "logs/tune.log",
        "-----------------------------------------------------",
    )
    log_to_file(
        "logs/tune.log",
        f"Y<{len(outlier_points[:, 1]):,}>: {outlier_points[:, 1].min():,} to {outlier_points[:, 1].max():,}",
    )
    y_counter = Counter(outlier_points[:,1])
    log_to_file("logs/tune.log", f"We have {len(y_counter):,} unique Y values")
    log_to_file(
        "logs/tune.log",
        f"Most Common Y: {y_counter.most_common(1)}, Least Two Common Y: {y_counter.most_common()[:-3:-1]}",
    )
    log_to_file(
        "logs/tune.log",
        "-----------------------------------------------------",
    )
    log_to_file(
        "logs/tune.log",
        f"Z<{len(outlier_points[:, 2]):,}>: {outlier_points[:, 2].min():,} to {outlier_points[:, 2].max():,}",
    )
    z_counter = Counter(outlier_points[:,2])
    log_to_file("logs/tune.log", f"We have {len(z_counter):,} unique Z values")
    log_to_file(
        "logs/tune.log",
        f"Most Common Z: {z_counter.most_common(1)}, Least Two Common Y: {z_counter.most_common()[:-3:-1]}",
    )
    log_to_file(
        "logs/tune.log",
        "-----------------------------------------------------",
    )


@timeit
def compute_k_matrix(**kwargs) -> np.ndarray:
    k_matrix_dict: dict[str, np.ndarray] = {
        "hammer": np.array([
            [7600, 0, 2736],
            [0, 7600, 1824],
            [0, 0, 1.0],
        ]),
        "cottage": np.array([
            [4044.943820224719, 0, 3000],
            [0, 4044.943820224719, 2000],
            [0, 0, 1.0],
        ]),
        "fountain": np.array([
            [3708.232031805074, 0, 1536],
            [0, 3708.232031805074, 1024],
            [0, 0, 1.0],
        ])
    }
    image_set_name = kwargs['image_set_name']
    return k_matrix_dict[image_set_name]


to_tuple = lambda x: tuple(x.flatten())


def check_coherent_rotation(R: np.ndarray) -> bool:
    return np.abs(np.linalg.det(R) - 1.0) <= 1e-6


@timeit
def find_3D_2D_correspondences(
        image_two: Image,
        feature_matches: list[FeatureMatches], 
        global_dict: dict[np.ndarray, set[tuple[int]]],
        **kwargs
    ) -> dict[np.ndarray, np.ndarray]:
    local_dict: dict[np.ndarray, np.ndarray] = {}
    for feature_match in feature_matches: # 1, 2, 3, 4 -> [(1, 2), "(1, 3)", (1, 4), "(2, 3)", (2, 4), (3, 4)]
        if feature_match.image_two != image_two:
            continue
        for match in feature_match.matches:
            search_keypoint_one = feature_match.image_one.keypoints[match.queryIdx].pt
            search_img_id = feature_match.image_one.img_id
            search_tuple = (search_img_id, search_keypoint_one)
            for key, values in global_dict.items():
                if search_tuple in values and key not in local_dict:
                    local_dict[key] = image_two.keypoints[match.trainIdx].pt
    return local_dict


@timeit
def find_initial_camera_matrices(K: np.ndarray, keypoints_one: np.ndarray, keypoints_two: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    E, mask = OpenCV.findEssentialMat(keypoints_one, keypoints_two, K, method=OpenCV.RANSAC, prob=0.999, threshold=1.0)
    # TODO: use mask to filter out outliers
    _, R, t, _ = OpenCV.recoverPose(E, keypoints_one, keypoints_two, K)
    return (R, t) if check_coherent_rotation(R) else (None, None)


@timeit
def find_next_camera_matrices(
        images: Images,
        image_one: Image,
        image_two: Image, 
        K_matrix: np.ndarray, 
        global_dict: dict[np.ndarray, set[tuple[int]]],
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
    image_set_name = kwargs['image_set_name']
    if image_one is not None:
        log_to_file(
            "logs/tune.log",
            f"Using Images {image_one.img_id} and {image_two.img_id} in find_next_camera_matrices",
        )
    local_dict: dict[np.ndarray, np.ndarray] = find_3D_2D_correspondences(image_two, images.feature_matches, global_dict, image_set_name=image_set_name)
    objectPoints = np.array(list(local_dict.keys())).reshape(-1, 3)
    imagePoints = np.array(list(local_dict.values())).reshape(-1, 2)
    log_to_file(
        "logs/tune.log",
        f"Found {objectPoints.shape[0]} 3D Points and {imagePoints.shape[0]} Image Points 3D-2D correspondences",
    )
    _, rvec, tvec, _ = OpenCV.solvePnPRansac(objectPoints, imagePoints, K_matrix, None)
    R, _ = OpenCV.Rodrigues(rvec)
    return R, tvec


@timeit
def compute_points_3D(
        P1: np.ndarray, 
        P2: np.ndarray, 
        image_one: Image,
        image_two: Image,
        keypoints_one: np.ndarray,
        keypoints_two: np.ndarray,
        global_dict: dict[np.ndarray, set[tuple[int]]],
        **kwargs
    ) -> np.ndarray:
    image_set_name = kwargs['image_set_name']
    data_path: str = f"../../data/{image_set_name}"
    points_3D = np.empty((3, len(keypoints_one)))
    for point_counter, (keypoint_one, keypoint_two) in enumerate(zip(keypoints_one, keypoints_two)):
        point_4D = OpenCV.triangulatePoints(P1, P2, keypoint_one.T, keypoint_two.T)  # 4x1
        point_3D = (point_4D / point_4D[3])[:3]  # 3x1
        if to_tuple(point_3D) in global_dict:
            global_dict[to_tuple(point_3D)].add((image_one.img_id, to_tuple(keypoint_one)))
            global_dict[to_tuple(point_3D)].add((image_two.img_id, to_tuple(keypoint_two)))
        else:
            global_dict[to_tuple(point_3D)] = {
                (image_one.img_id, to_tuple(keypoint_one)),
                (image_two.img_id, to_tuple(keypoint_two))
            }
        points_3D[:, point_counter] = point_3D.flatten()
    log_to_file(
        "logs/tune.log",
        f"Computed {points_3D.shape[1]} 3D Points for Image pairs {image_one.img_id} and {image_two.img_id}",
    )
    return points_3D


@timeit
def find_cluster_feature_matches( 
        images: Images, 
        values: list[Image],
        **kwargs
    ) -> list[FeatureMatches]: # [1,2,3] ----> [1,2],[1,3]
    image_set_name = kwargs['image_set_name']
    data_path: str = f"../../data/{image_set_name}"
    cluster_reference_image = values[0]
    cluster_feature_matches: list[FeatureMatches] = []
    import itertools
    for image, matched_image in itertools.combinations(values, 2):
        if image.img_id != cluster_reference_image.img_id:
            log_to_file(
                "logs/tune.log",
                f"Breaking itertools loop for {image.img_id} and {matched_image.img_id} in find_cluster_feature_matches\n",
            )
            break
        else:
            appended_pair: FeatureMatches = next(
                fm for fm in images.feature_matches
                if fm.image_one.img_id == image.img_id and fm.image_two.img_id == matched_image.img_id
            )
            log_to_file("logs/tune.log", f"appended_pair: {appended_pair}")
            cluster_feature_matches.append(appended_pair)
    return cluster_feature_matches


@timeit
def generate_points_cloud(images: Images, K_matrix: np.ndarray, **kwargs) -> np.ndarray:
    image_set_name = kwargs['image_set_name']
    points_cloud: list[list[np.ndarray]] = []
    global_dict: dict[np.ndarray, set[tuple[int]]] = {}
    camera_matrices: list[np.ndarray] = [(np.eye(3), np.zeros((3, 1)))]
    for cluster, values in images.similar_images.items():
        log_to_file(
            "logs/tune.log",
            f"--------------------- Entering Cluster {cluster} ---------------------",
        )
        cluster_feature_matches:list[FeatureMatches] = find_cluster_feature_matches(images, values, image_set_name=image_set_name)
        log_to_file(
            "logs/tune.log",
            f"cluster_feature_matches: {cluster_feature_matches}\n",
        )
        if cluster == list(images.similar_images.keys())[0]: # First cluster
            P1 = K_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
            for feature_match in cluster_feature_matches:
                image_one = feature_match.image_one
                image_two = feature_match.image_two
                keypoints_one = np.array([image_one.keypoints[m.queryIdx].pt for m in feature_match.matches])
                keypoints_two = np.array([image_two.keypoints[m.trainIdx].pt for m in feature_match.matches])
                if feature_match == cluster_feature_matches[0]:  # First Feature Match Pair in the First Cluster, where we use recoverPose
                    log_to_file(
                        "logs/tune.log",
                        f"Using Images {image_one.img_id} and {image_two.img_id} in recoverPose",
                    )
                    R, t = find_initial_camera_matrices(K_matrix, keypoints_one, keypoints_two, image_set_name=image_set_name)
                    P2 = K_matrix @ np.hstack((R, t))
                    camera_matrices.append((R, t))
                else:
                    R, tvec = find_next_camera_matrices(images, image_one, image_two, K_matrix, global_dict, image_set_name=image_set_name)
                    P2 = K_matrix @ np.hstack((R, tvec))
                    camera_matrices.append((R, tvec))
                points_3D = compute_points_3D(P1, P2, image_one, image_two, keypoints_one, keypoints_two, global_dict, image_set_name=image_set_name)
                points_cloud.append(points_3D)
                log_to_file(
                    "logs/tune.log",
                    f"Global Dict 3D Points Size: {len(global_dict.keys())} \n",
                )
        else: # Next Clusters
            for feature_match in cluster_feature_matches:
                image_one = feature_match.image_one
                image_two = feature_match.image_two
                keypoints_one = np.array([image_one.keypoints[m.queryIdx].pt for m in feature_match.matches])
                keypoints_two = np.array([image_two.keypoints[m.trainIdx].pt for m in feature_match.matches])
                if feature_match == cluster_feature_matches[0]: # First Iteration of the next Cluster
                # Computing new P1 for the new cluster
                    log_to_file(
                        "logs/tune.log",
                        f"Entered First Iteration of the cluster {cluster}",
                    )
                    log_to_file(
                        "logs/tune.log",
                        f"Using Image {image_one.img_id} as Reference Image in cluster {cluster} to compute P1 for cluster {cluster}",
                    )
                    P1_R, P1_tvec = find_next_camera_matrices(images, None, image_one, K_matrix, global_dict, image_set_name=image_set_name)
                    P1 = K_matrix @ np.hstack((P1_R, P1_tvec))
                R, tvec = find_next_camera_matrices(images, image_one, image_two, K_matrix, global_dict, image_set_name=image_set_name)
                P2 = K_matrix @ np.hstack((R, tvec))
                camera_matrices.append((R, tvec))
                points_3D = compute_points_3D(P1, P2, image_one, image_two, keypoints_one, keypoints_two, global_dict, image_set_name=image_set_name)
                points_cloud.append(points_3D)
                log_to_file(
                    "logs/tune.log",
                    f"Global Dict 3D Points Size: {len(global_dict.keys())} \n",
                )
        log_to_file(
            "logs/tune.log",
            f"--------------------- End of cluster {cluster} ---------------------\n\n",
        )
    points_cloud = np.hstack(points_cloud).T
    log_to_file("logs/tune.log", "Done generating points cloud")
    return points_cloud, camera_matrices


@timeit
def create_camera_frustum(P: np.ndarray, scale: float) -> o3d.geometry.TriangleMesh:
    vertices = np.array([[0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0], [-0.5, 0.5, 0], [0, 0, -1]])
    vertices *= scale
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [1, 0, 3]])
    R, t = P
    vertices = vertices @ R.T + t[:3].T
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    vertex_colors = np.ones((len(vertices), 3)) * [1, 0, 0]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    # draw camera rod
    start_point = np.array([0, 0, 0])
    end_point = np.array([0, 0, 1])*scale
    start_point = start_point @ R.T + t[:3].T
    end_point = end_point @ R.T + t[:3].T
    rod = o3d.geometry.TriangleMesh.create_cylinder(radius=0.02*scale, height=np.linalg.norm(end_point-start_point), resolution=20, split=4)
    rod.vertices = o3d.utility.Vector3dVector(np.asarray(rod.vertices) + start_point)
    vertex_colors = np.ones((len(rod.vertices), 3)) * [0, 0, 0]
    rod.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh, rod