import enum
import gc
import os
import pickle
import sys
import time
from typing import Final, Optional

import cv2 as OpenCV
import numpy as np
import open3d as o3d

from under_the_hood import (
    analyize_points, compute_keypoints_descriptors, compute_k_matrix,
    create_camera_frustum, data_feature_matching, generate_points_cloud,
    image_matching, prepare_images,
)
from under_the_hood.data_structures import Images
from under_the_hood.utils import (
    dump_images_bak, log_to_file, load_images_bak, print_size,
)


class Mode(enum.Enum):
    OPTMIZED = "optimized"
    DEBUG = "debug"

def retrive_last_state(data_path: str) -> str:
    """ Reloading the last state """
    output_files_3D: list[str] = [
        "output/triangulate/points_cloud.stl",
        "output/triangulate/core_points.stl",
        "output/triangulate/camera_proj.stl",
        "output/triangulate/mesh.stl",
    ]
    output_files_triangulation: list[str] = [
        "bak/k-matrix.pkl",
        "bak/points-cloud.pkl",
        "bak/core-points.pkl",
        "bak/camera-proj.pkl",
        "bak/hdbscan-model.pkl",
    ]
    if all(
        os.path.isfile(output_file)
        for output_file in output_files_3D
    ):
        return "3D Reconstruction Step"
    elif all(
        os.path.isfile(output_file)
        for output_file in output_files_triangulation
    ):
        return "Triangulation Step"
    elif os.path.isfile("bak/feature-matching-output.pkl"):
        return "Feature Matching Step"
    elif os.path.isfile("bak/matched-images.pkl"):
        return "Images Matching Step"
    elif os.path.isfile("bak/sift-features.pkl"):
        return "SIFT Features Step"
    else:
        return "Images Loading Step"

def run(image_set_name: str) -> None:
    """ Intialized the pipeline """
    data_path: str = f"../data/{image_set_name}"
    mode: enum = Mode.OPTMIZED
    NUM_CLUSTERS: Final[int] = 1
    overwrite: Final[bool] = True
    images: Optional[Images] = None
    """ Partial functions """
    from functools import partial
    """ local logger to "logs/tune.log" and passes image_set_name=image_set_name """
    local_logger = partial(log_to_file, file_name=f"logs/tune.log", image_set_name=image_set_name)
    """ local loadder and dumper passes image_set_name=image_set_name """
    local_loader = partial(load_images_bak, image_set_name=image_set_name)
    local_dumper = partial(dump_images_bak, image_set_name=image_set_name)
    """ Start """
    local_logger("Welcome ScanMate...")
    local_logger(f"Running image_set_name {image_set_name} in {mode} mode...")
    """ Reloading the last state """
    last_state: str = retrive_last_state(data_path)
    local_logger(f"Last state for {image_set_name} is {last_state}.")
    """ #1. Step: Images Loading """
    if last_state == "Images Loading Step":
        if os.path.isfile(f"bak/images.pkl"):
            local_logger(f"File [bak/sift-images.pkl] exists")
            local_logger("Loading images from pickle file...")
            images: Images = local_loader(f"bak/images.pkl")
        else:
            local_logger(f"File [bak/images.pkl] does not exist")
            local_logger("Loading images from images directory...")
            images: Images = prepare_images(create_mask=True, image_set_name=image_set_name)
            local_logger("Saving images to pickle file...")
            local_dumper(f"bak/images.pkl", images)
        local_logger("Images loaded successfully")
        last_state = "SIFT Features Step"
    """ #2. Step: compute SIFT Features """
    if last_state == "SIFT Features Step":
        if os.path.isfile(f"bak/sift-features.pkl"):
            local_logger(f"File [bak/sift-features.pkl] exists")
            if images: 
                del images
            images: Images = local_loader(f"bak/sift-features.pkl")
        else:
            local_logger(f"File [bak/sift-features.pkl] DO NOT exists")
            local_logger("Extracting SIFT features...")
            sift = OpenCV.SIFT_create(contrastThreshold=0.01)
            local_logger(f"Ref count of images before: {sys.getrefcount(images)}")
            print_size(f"logs/tune.log", images, "images")
            compute_keypoints_descriptors(images, sift, image_set_name=image_set_name)
            print_size(f"logs/tune.log", images, "images")
            local_logger(f"Ref count of images after: {sys.getrefcount(images)}")
            local_dumper(f"bak/sift-features.pkl", images)
            if mode == Mode.OPTMIZED:
                if os.path.exists(f"bak/images.pkl"):
                    os.remove(f"bak/images.pkl")
                    local_logger(f"File bak/images.pkl removed successfully.")
                else:
                    local_logger(f"File bak/images.pkl does not exist.")
        local_logger("Feature Extraction: SIFT DONE...")
        last_state = "Images Matching Step"
    """ #3. Step: Image Matching """
    if last_state == "Images Matching Step":
        if os.path.isfile(f"bak/matched-images.pkl"):
            local_logger(f"File [bak/matched-images.pkl] exists")
            if images: 
                del images
            images: Images = local_loader(f"bak/matched-images.pkl")
        else:
            local_logger(f"File [bak/matched-images.pkl] DO NOT exists")
            local_logger("Matching images...")
            local_logger(f"Ref count of images before: {sys.getrefcount(images)}")
            print_size(f"logs/tune.log", images, "images")
            images.num_clusters = NUM_CLUSTERS
            image_matching(images, overwrite=overwrite, image_set_name=image_set_name)
            local_logger("image matching done")
            if not overwrite:
                images.save_similar_images()
                local_logger("saved image clusters")
                print_size(f"logs/tune.log", images, "images")
                local_logger(f"Ref count of images after: {sys.getrefcount(images)}")
                local_dumper(f"bak/matched-images.pkl", images)
            if mode == Mode.OPTMIZED:
                if os.path.exists(f"bak/sift-features.pkl"):
                    os.remove(f"bak/sift-features.pkl")
                    local_logger(f"File bak/sift-features.pkl removed successfully.")
                else:
                    local_logger(f"File bak/sift-features.pkl does not exist.")
        local_logger("Done Image Matching Step...")
        last_state = "Feature Matching Step"
    """ #4. Step: Feature Matching """
    if last_state == "Feature Matching Step":
        if os.path.isfile(f"bak/feature-matching-output.pkl"):
            local_logger(f"File [bak/feature-matching-output.pkl] exists")
            if images: 
                del images
            images: Images = local_loader(f"bak/feature-matching-output.pkl")
        else:
            local_logger(f"File [bak/feature-matching-output.pkl] Do NOT exists")
            local_logger("Matching features...")
            local_logger(f"Ref count of images before: {sys.getrefcount(images)}")
            print_size(f"logs/tune.log", images, "images")
            data_feature_matching(images, image_set_name=image_set_name)
            local_logger("done feature matching")
            images.save_feature_matches()
            local_logger("saved feature matching")
            print_size(f"logs/tune.log", images, "images")
            local_logger(f"Ref count of images after: {sys.getrefcount(images)}")
            local_dumper(f"bak/feature-matching-output.pkl", images)
            if mode == Mode.OPTMIZED:
                if os.path.exists(f"bak/matched-images.pkl"):
                    os.remove(f"bak/matched-images.pkl")
                    local_logger(f"File bak/matched-images.pkl removed successfully.")
                else:
                    local_logger(f"File bak/matched-images.pkl does not exist.")
        local_logger("Done Feature Matching Step...")
        last_state = "Triangulation Step"
    """ #5. Step: Triangulation """
    if last_state == "Triangulation Step":
        # 5.1 Camera Calibration
        local_logger("Camera Calibration starts ....")
        if not os.path.isfile(f"bak/k-matrix.pkl"):
            local_logger(f"File bak/k-matrix.pkl does not exist")
            K_matrix = compute_k_matrix(images.images[0].path, image_set_name=image_set_name)
            with open(f"bak/k-matrix.pkl", 'wb') as f:
                pickle.dump(K_matrix, f)
            local_logger(f"File bak/k-matrix.pkl saved successfully")
        else:
            local_logger(f"File bak/k-matrix.pkl exists")
            with open(f"bak/k-matrix.pkl", 'rb') as f:
                K_matrix = pickle.load(f)
        # 5.2 Triangulation
        local_logger("Triangulation starts ....")
        if os.path.isfile(f"bak/points-cloud.pkl"):
            local_logger(f"File bak/points-cloud.pkl exists")
            with open(f"bak/points-cloud.pkl", 'rb') as f:
                points_cloud: np.ndarray = pickle.load(f)
            with open(f"bak/camera-proj.pkl", 'rb') as f:
                camera_matrices: np.ndarray = pickle.load(f)
        else:
            local_logger(f"File bak/points-cloud.pkl does not exist")
            local_logger("Triangulating...")
            local_logger(f"Ref count of images before: {sys.getrefcount(images)}")
            print_size(f"logs/tune.log", images, "images")
            points_cloud, camera_matrices = generate_points_cloud(images, K_matrix, image_set_name=image_set_name)
            print_size(f"logs/tune.log", images, "images")
            local_logger(f"Ref count of images after: {sys.getrefcount(images)}")
            with open(f"bak/points-cloud.pkl", 'wb') as f:
                pickle.dump(points_cloud, f)
            with open(f"bak/camera-proj.pkl", 'wb') as f:
                pickle.dump(camera_matrices, f)
        # 5.3 Cleaning memory before clustering
        local_logger("Done Point Cloud Step...")
        local_logger(f"Ref count of images before: {sys.getrefcount(images)}")
        print_size(f"logs/tune.log", images, "images")
        print_size(f"logs/tune.log", points_cloud, "points_cloud")
        images = None
        local_logger(gc.collect())
        print_size(f"logs/tune.log", images, "images")
        local_logger(f"Reference count<images>: {sys.getrefcount(images)}")
        local_logger(gc.collect())
        # 5.4 Points Clustering
        local_logger("started clustring....")
        import hdbscan
        start_time = time.time()
        hdbscan_model = hdbscan.HDBSCAN().fit(points_cloud)
        end_time = time.time()
        local_logger(f"time taken: {end_time - start_time:,} seconds")
        with open(f"bak/hdbscan-model.pkl", 'wb') as f:
            pickle.dump(hdbscan_model, f)
        local_logger("File hdbscan-model.pkl saved successfully...")
        print_size(f"logs/tune.log", hdbscan_model, "hdbscan_model")
        labels = hdbscan_model.labels_
        local_logger("Labels Done...")
        core_indices = np.where(labels != -1)[0]
        local_logger("Core Indicies Done...")
        core_points = points_cloud[core_indices, :]
        local_logger("Core Points Done...")
        outlier_indices = np.where(labels == -1)[0]
        local_logger("Outlier Indicies Done...")
        outlier_points = points_cloud[outlier_indices, :]
        local_logger("Outlier Points Done...")
        local_logger(f"Number of clusters: {len(np.unique(labels))-1:,}")
        local_logger(f"Number of core points: {len(core_indices):,}")
        local_logger(f"Number of outlier points: {len(outlier_indices):,}")
        local_logger(f"Number of total points: {len(core_indices) + len(outlier_indices):,}")
        with open(f"bak/core_points.pkl", 'wb') as f:
            pickle.dump(core_points, f)
        local_logger("File core_points.pkl saved successfully...")
        print_size(f"logs/tune.log", core_points, "core_points")
        # 5.5 3D Reconstruction
        local_logger(f"points_cloud.shape: {points_cloud.shape:,}")
        local_logger(f"Number of cameras detected: {len(camera_matrices):,}")
        local_logger(f"core_points.shape: {core_points.shape:,}")
        points_cloud_stl = o3d.geometry.PointCloud()
        points_cloud_stl.points = o3d.utility.Vector3dVector(points_cloud)
        core_points_stl = o3d.geometry.PointCloud()
        core_points_stl.points = o3d.utility.Vector3dVector(core_points)
        camera_meshes = []
        camera_lines = []
        for camera_matrix in camera_matrices:
            camera_mesh, camera_line = create_camera_frustum(camera_matrix, scale=0.3)
            camera_meshes.append(camera_mesh)
            camera_lines.append(camera_line)
        combined_mesh = o3d.geometry.TriangleMesh()
        for mesh in camera_meshes + camera_lines:
            combined_mesh += mesh
        point_cloud_file = f"output/triangulate/points_cloud.ply"
        o3d.io.write_point_cloud(point_cloud_file, points_cloud_stl)
        point_cloud_file = f"output/triangulate/core_points.ply"
        o3d.io.write_point_cloud(point_cloud_file, core_points_stl)
        mesh_file = f"output/triangulate/camera_proj.ply"
        o3d.io.write_triangle_mesh(mesh_file, combined_mesh)
        # 5.6 Meshing
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(core_points[:, :3])
        pcd.estimate_normals()
        _, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        inlier_pcd = pcd.select_by_index(inlier_indices)
        distances = inlier_pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(inlier_pcd, o3d.utility.DoubleVector([radius, radius * 2]))
        dec_mesh = bpa_mesh.simplify_quadric_decimation(100_000)
        dec_mesh.remove_degenerate_triangles()
        dec_mesh.remove_duplicated_triangles()
        dec_mesh.remove_duplicated_vertices()
        dec_mesh.remove_non_manifold_edges()
        o3d.io.write_triangle_mesh(f"output/triangulate/mesh.stl", dec_mesh)
        # 5.7 Further Analysis
        analyize_points(points_cloud, core_points, outlier_points)