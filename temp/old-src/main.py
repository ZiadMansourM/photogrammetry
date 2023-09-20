import gc
import os
import pickle
import sys
import time
import uuid
from typing import Final, Optional

import cv2 as OpenCV
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.cluster.vq import kmeans, vq
from scipy.spatial import Delaunay


def log_to_file(file_name: str, message: str):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    with open(file_name, "a") as f:
        f.write(f"{log_message}\n")


def print_size(file_name: str, obj, obj_name="N/A"):
    from pympler import asizeof
    memory_usage = asizeof.asizeof(obj)
    # Convert memory usage to a more readable format
    if memory_usage < 1024:
        memory_usage_str = f"{memory_usage} bytes"
    elif memory_usage < 1024 ** 2:
        memory_usage_str = f"{memory_usage / 1024} KB"
    elif memory_usage < 1024 ** 3:
        memory_usage_str = f"{memory_usage / (1024 ** 2)} MB"
    else:
        memory_usage_str = f"{memory_usage / (1024 ** 3)} GB"
    # Print the memory usage and object name
    log_to_file(file_name, f"Memory usage of {obj_name}: {memory_usage_str}")

def timeit(func):
    def wrapper(*args, **kwargs):
        img_set_name = kwargs['img_set_name']
        log_to_file(f"data/{img_set_name}/logs/tune.log", f"Started {func.__name__}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        log_to_file(f"data/{img_set_name}/logs/tune.log", f"Done {func.__name__} took {end_time - start_time:,} seconds to execute.")
        return result
    return wrapper

class CalibrationError(Exception):
    def __init__(self, message):
        self.message = message

class IntrinsicParametersNotFoundError(Exception):
    def __init__(self, message):
        self.message = message

class Image:
    def __init__(self, img_id, rgb_image, gray_image, keypoints, descriptors, path):
        self.img_id: int = img_id
        self.unique_id: uuid = uuid.uuid4()
        self.rgb_image: Image = rgb_image
        self.gray_image: Image = gray_image
        self.keypoints: list[OpenCV.KeyPoint] = keypoints
        self.descriptors: np.ndarray = descriptors
        self.path: str = path
        self.similar_images: list[tuple[Image, float]] = []

    @property
    def length(self):
        return f"{len(self.keypoints)}" if len(self.keypoints) == len(self.descriptors) else f"{len(self.keypoints)}, {len(self.descriptors)}"
    
    def draw_sift_features(self):
        image_with_sift = OpenCV.drawKeypoints(self.rgb_image, self.keypoints, None, flags=OpenCV.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(image_with_sift)
        plt.title("Image with SIFT Features")
        plt.axis('off')
        plt.show()

    def display_rgb_image(self, title: Optional[str] = None):
        image = self.rgb_image
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.axis('off')
        plt.show()

    def display_gray_image(self, title: Optional[str] = None):
        image = self.gray_image
        plt.gray()
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.axes('off')
        plt.show()

    def display_similar_images(self):
        print(f"Ref image: Img({self.img_id}, {self.path})")
        self.display_rgb_image()
        print("-----------------------------------------------------")
        for sim_img, perc in self.similar_images:
            print(f"[{sim_img.img_id}, {sim_img.path}], image percentage {perc}, {sim_img.path}")
            sim_img.display_rgb_image()
    
    def __repr__(self):
        return f"Image({self.img_id})"
    
    def __str__(self):
        return self.__repr__()
    
    def __eq__(self, other):
        return self.unique_id == other.unique_id
    
    def __hash__(self):
        return hash(self.img_id)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['keypoints'] = [tuple(k.pt) + (k.size, k.angle, k.response, k.octave, k.class_id) for k in self.keypoints]
        return state
    
    def __setstate__(self, state):
        state['keypoints'] = [OpenCV.KeyPoint(x, y, size, angle, response, octave, class_id) for x, y, size, angle, response, octave, class_id in state['keypoints']]
        self.__dict__ = state

class FeatureMatches:
    def __init__(self, image_one: Image, image_two: Image, matches: list[OpenCV.DMatch]):
        self.image_one: Image = image_one
        self.image_two: Image = image_two
        self.matches: list[OpenCV.DMatch] = matches

    def draw_matches(self, output_filename: str) -> None:
        combined_image = OpenCV.hconcat([
            self.image_one.rgb_image,
            self.image_two.rgb_image
        ])

        for match in self.matches:
            x1, y1 = self.image_one.keypoints[match.queryIdx].pt
            x2, y2 = self.image_two.keypoints[match.trainIdx].pt
            # Draw a line connecting the matched keypoints
            OpenCV.line(
                combined_image, 
                (int(x1), int(y1)), 
                (int(x2) + self.image_one.rgb_image.shape[1], int(y2)), 
                (0, 255, 0), 
                1
            )

        OpenCV.imwrite(output_filename, combined_image)

    def __repr__(self):
        return f"FeatureMatches({self.image_one}, {self.image_two} ---> {len(self.matches)})"

    def __getstate__(self):
        state = self.__dict__.copy()
        state['matches'] = [
            {'queryIdx': m.queryIdx, 'trainIdx': m.trainIdx, 'distance': m.distance} for m in self.matches
        ]
        return state
    
    def __setstate__(self, state):
        state['matches'] = [
            OpenCV.DMatch(match['queryIdx'], match['trainIdx'], match['distance']) for match in state['matches']
        ]
        self.__dict__ = state
    
class Images:
    def __init__(self, images: list[Image], image_set_name: str):
        self.id = uuid.uuid4()
        self.images: list[Image] = images
        self.image_set_name: str = image_set_name
        self.feature_matches: list[FeatureMatches] = []

    def __len__(self):
        return len(self.images)


def dump_images_bak(images_file_path: str, images: Images) -> None:
    """ Dump images to a file """
    with open(images_file_path, "wb") as file:
        pickle.dump(images, file)

def load_images_bak(images_file_path: str) -> Images:
    """ Load images from a file """
    with open(images_file_path, "rb") as file:
        images = pickle.load(file)
    return images

"""Step One"""
@timeit
def prepare_images(**kwargs) -> Images:
    """ Read and load images """
    img_set_name = kwargs['img_set_name']
    folder_path = f"data/{img_set_name}/images"
    images: Images = Images([], folder_path.split("/")[-2])
    files: list[str] = filter(lambda file: ".jpg" in file, os.listdir(folder_path))
    for i, file in enumerate(files):
        image_path = f"{folder_path}/{file}"
        rgb_image = OpenCV.cvtColor(OpenCV.imread(image_path), OpenCV.COLOR_BGR2RGB)
        gray_image = OpenCV.cvtColor(rgb_image, OpenCV.COLOR_RGB2GRAY)
        images.images.append(Image(i, rgb_image, gray_image, [], [], image_path))
    return images


"""Step Two"""""
@timeit
def compute_keypoints_descriptors(images: list[Image], SIFT: OpenCV.SIFT, **kwargs) -> None:
    """Compute keypoints and descriptors for each image in the list of images using SIFT algorithm.
    Modifies each image in the list of images by adding its keypoints and descriptors as attributes.
    
    Args:
    - images: List of images to compute keypoints and descriptors for.
    - SIFT: OpenCV SIFT object used to detect and compute keypoints and descriptors.

    Returns:
    - None.
    """
    img_set_name = kwargs['img_set_name']
    for img in images.images:
        keypoints: list[OpenCV.KeyPoint]
        descriptors: np.ndarray
        keypoints, descriptors = SIFT.detectAndCompute(img.gray_image, None)
        img.keypoints = keypoints
        img.descriptors = descriptors
        log_to_file(f"data/{img_set_name}/logs/tune.log", f"Img({img.img_id}, {img.path}) has {len(img.keypoints)} keypoints and {len(img.descriptors)} descriptors.")

"""Step Three"""
@timeit
def get_matches(images: Images, **kwargs) -> None:
    """ Match images using k-means clustering.
    Args:
        images: Obj from Images class.
    """
    img_set_name = kwargs['img_set_name']
    all_descriptors = np.concatenate([image.descriptors for image in images.images])
    CLUSTER_COUNT: Final[int] = 400
    ITER: Final[int] = 2
    centroids, _ = kmeans(all_descriptors, CLUSTER_COUNT, ITER)
    matches_ids: list[list[tuple[int, float]]] =  get_matches_ids(
        [image.descriptors for image in images.images], 
        centroids, 
        images.images,
        **kwargs
    )
    for i, image in enumerate(images.images):
        inner_list: list[Image, float] = [
            (images.images[match[0]], match[1]) for match in matches_ids[i]
        ]
        image.similar_images = inner_list
        log_to_file(f"data/{img_set_name}/logs/tune.log", f"Img({image.img_id}, {image.path}) with similar images:")
        log_message = ' - '.join(f"({img.img_id}, {perc})" for img, perc in inner_list)
        log_to_file(f"data/{img_set_name}/logs/tune.log", log_message)

@timeit
def get_visual_words(descriptors: list[np.ndarray], centroids: np.ndarray, **kwargs) -> list[np.ndarray]:
    """ Get the visual words of a list of descriptors.
    Args:
        descriptors: A list of numpy arrays containing image descriptors.
        centroids: A numpy array containing cluster centroids.
    Returns:
        A list of numpy arrays representing the visual words of each image.
    """
    visual_words = []
    for descriptor in descriptors:
        words, _ = vq(descriptor, centroids)
        visual_words.append(words)
    return visual_words

@timeit
def get_frequency_vectors(visual_words: list[np.ndarray], CLUSTER_COUNT: int, **kwargs) -> np.ndarray:
    """ Get the frequency vectors for a list of visual words.
    Args:
        visual_words: A list of numpy arrays representing the visual words of each image.
        CLUSTER_COUNT: The number of clusters used to generate the visual words.
    Returns:
        A numpy array containing the frequency vectors for each image.
    """
    frequency_vectors = []
    for img_words in visual_words:
        histogram = np.zeros(CLUSTER_COUNT)
        for word in img_words:
            histogram[word] += 1
        frequency_vectors.append(histogram)
    return np.stack(frequency_vectors)

@timeit
def get_tf_idf(frequency_vectors, IMAGES_COUNT, **kwargs) -> np.ndarray:
    """ Get the Term Frequency-Inverse Document Frequency (TF-IDF) matrix for a list of frequency vectors.
    Args:
        frequency_vectors: A numpy array containing the frequency vectors for each image.
        IMAGES_COUNT: The total number of images in the dataset.
    Returns:
        A numpy array containing the TF-IDF matrix for the input frequency vectors.
    """
    df = np.sum(frequency_vectors > 0, axis = 0)
    idf = np.log(IMAGES_COUNT/df)
    return frequency_vectors * idf


def search_matches(i, top_clusters, tf_idf) -> list[tuple[int, float]]:
    """ Search for the top_clusters most similar images to the i-th image
    Args:
        i: the index of the image to search for similar images
        top_clusters: the number of similar images to return
        tf_idf: Term Frequency-Inverse Document Frequency
    Returns:
        A list of tuples, where each tuple contains the index of a similar image and the cosine similarity 
        between the i-th image and the similar image. The list is sorted by the cosine similarity in 
        descending order.
    """
    b = tf_idf
    a = tf_idf[i]
    b_subset = b[:tf_idf.shape[0]]
    cosine_similarity = np.dot(a, b_subset.T)/(norm(a) * norm(b_subset, axis=1))
    idx = np.argsort(-cosine_similarity)[:top_clusters]
    return list(zip(idx, cosine_similarity[idx]))

@timeit
def get_matches_ids(descriptors, centroids, images_list, **kwargs) -> list[list[tuple[int, float]]]:
    """Returns: a list of lists, where each list contains the top 10 most similar images to the i-th image."""
    visual_words = get_visual_words(descriptors, centroids, **kwargs)
    frequency_vectors = get_frequency_vectors(visual_words, centroids.shape[0], **kwargs)
    """ tf_idf: Term Frequency-Inverse Document Frequency """
    tf_idf = get_tf_idf(frequency_vectors, len(images_list), **kwargs)
    return [
        search_matches(i, 10, tf_idf)
        for i in range(len(images_list))
    ]

"""Step Four"""

@timeit
def feature_matching(
        img_one_descriptors: np.ndarray, 
        img_two_descriptors: np.ndarray,
        **kwargs
    ) -> list[OpenCV.DMatch]:
    """ Match features between two images using Brute Force Matcher
    Args:
        img_id_one: the index of the first image
        img_id_two: the index of the second image
        descriptors: a list of descriptors of the images
    Returns:
        A list of OpenCV.DMatch objects.
    """
    matcher = OpenCV.BFMatcher()
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
    """ Match features between images using Brute Force Matcher
    Args:
        matchesIDs: a list of lists of tuples, where each tuple contains the index of a similar image and the cosine similarity 
            between the i-th image and the similar image. The list is sorted by the cosine similarity in 
            descending order.
        descriptors: a list of descriptors of the images
    Returns:
        A list of lists, where each list contains 
        the index of the first image, the index of the second image, 
        and a list of OpenCV.DMatch objects.
    """
    num_images: int = len(images.images)
    checked = np.zeros((num_images, num_images), dtype=int)
    for image in images.images:
        log_to_file("logs/tune.log", f"Started Feature Match for Img({image.img_id}, {image.path}):")
        for matched_image, probability in image.similar_images:
            if ((checked[image.img_id][matched_image.img_id] == 0 or checked[matched_image.img_id][image.img_id] == 0) and image.img_id != matched_image.img_id and probability > 0.93):
                feature_matching_output = feature_matching(image.descriptors, matched_image.descriptors)
                ransac_output = apply_ransac(feature_matching_output, image.keypoints, matched_image.keypoints)
                images.feature_matches.append(FeatureMatches(image, matched_image, ransac_output))
                log_to_file("logs/tune.log", f"({image.img_id}, {matched_image.img_id}) with {len(feature_matching_output)}.")
                checked[image.img_id][matched_image.img_id], checked[matched_image.img_id][image.img_id] = 1, 1


"""Step Five"""
@timeit
def compute_k_matrix(img_path: str, **kwargs) -> np.ndarray:
    import exifread
    # Open the image file
    image = open(img_path, "rb")
    # Read the EXIF data
    exif = exifread.process_file(image)
    # Extract the intrinsic parameters
    focal_length = exif['EXIF FocalLength'].values[0]
    sensor_width = exif['EXIF ExifImageWidth'].values[0]
    sensor_height = exif['EXIF ExifImageLength'].values[0]
    principal_point_x = exif['EXIF ExifImageWidth'].values[0] / 2
    principal_point_y = exif['EXIF ExifImageLength'].values[0] / 2
    # distortion_coefficients = exif['EXIF MakerNote'].values[0]
    # Calculate the scaling factor for the K-matrix
    scaling_factor = 1.0
    return np.array(
        [
            [float(focal_length), 0, principal_point_x],
            [0, float(focal_length), principal_point_y],
            [0, 0, scaling_factor],
        ]
    )


"""Step Six"""
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
def generate_point_cloud(images: Images, K_matrix, **kwargs):
    """
    Generates a cloud of 3D points using triangulation from feature matches and camera calibration matrix.

    Parameters:
    feature_matches_list (list): List of feature matches between images.
    K_matrix (numpy.ndarray): 3x3 camera calibration matrix.

    Returns:
    numpy.ndarray: Nx3 matrix containing the cloud of 3D points.
    """
    point_cloud = []
    feature_matches_list = images.feature_matches
    for match in feature_matches_list:
        img_one = match.image_one
        img_two = match.image_two
        matches = match.matches
        pts1 = np.float32([img_one.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([img_two.keypoints[m.trainIdx].pt for m in matches])
        E, _ = OpenCV.findEssentialMat(pts1, pts2, K_matrix)
        R1, R2, t = OpenCV.decomposeEssentialMat(E)
        P1 = np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = np.hstack((R1, t))
        pts_3d = triangulatePoints(P1, P2, pts1, pts2)
        point_cloud.append(pts_3d)
    return np.concatenate(point_cloud, axis=0)


if __name__ == "__main__":
    import sys
    print(sys.executable)

    import enum

    class Mode(enum.Enum):
        OPTMIZED = "optimized"
        DEBUG = "debug"

    # image_set_name = "rubik-cube"
    # image_set_name = "snow-man"
    image_set_name = "test"

    log_to_file(f"data/{image_set_name}/logs/tune.log", "Welcome ScanMate...")
    mode: enum = Mode.DEBUG
    log_to_file(f"data/{image_set_name}/logs/tune.log", f"Running image_set_name {image_set_name} in {mode} mode...")
    images: Optional[Images] = None

    # 0. Reload the last state
    last_state: str
    if os.path.isfile(f"data/{image_set_name}/bak/feature-matching-output.pkl"):
        last_state = "Feature Matching Step"
    elif os.path.isfile(f"data/{image_set_name}/bak/images-matched.pkl"):
        last_state = "Images Matching Step"
    elif os.path.isfile(f"data/{image_set_name}/bak/sift-features.pkl"):
        last_state = "SIFT Features Step"
    else:
        last_state = "Images Loading Step"
    log_to_file(f"data/{image_set_name}/logs/tune.log", f"Last state for {image_set_name} is {last_state}")

    # 1. Load and prepare Images
    if last_state == "Images Loading Step":
        if os.path.isfile(f"data/{image_set_name}/bak/images.pkl"):
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File [data/{image_set_name}/bak/sift-images.pkl] exists")
            log_to_file(f"data/{image_set_name}/logs/tune.log", "Loading images from pickle file...")
            images: Images = load_images_bak(f"data/{image_set_name}/bak/images.pkl")
        else:
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File [data/{image_set_name}/bak/images.pkl] does not exist")
            log_to_file(f"data/{image_set_name}/logs/tune.log", "Loading images from images directory...")
            images: Images = prepare_images(f"images/{image_set_name}")
            log_to_file(f"data/{image_set_name}/logs/tune.log", "Saving images to pickle file...")
            dump_images_bak(f"data/{image_set_name}/bak/images.pkl", images)
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Images loaded successfully")
        last_state = "SIFT Features Step"

    # 2. Feature Extraction: SIFT
    if last_state == "SIFT Features Step":
        if os.path.isfile(f"data/{image_set_name}/bak/sift-features.pkl"):
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File [data/{image_set_name}/bak/sift-features.pkl] exists")
            if images: 
                del images
            images: Images = load_images_bak(f"data/{image_set_name}/bak/sift-features.pkl")
        else:
            log_to_file(f"data/{image_set_name}/logs/tune.log", "File [data/{image_set_name}/bak/sift-features.pkl] DO NOT exists")
            log_to_file(f"data/{image_set_name}/logs/tune.log", "Extracting SIFT features...")
            sift = OpenCV.SIFT_create()
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images before: {sys.getrefcount(images)}")
            print_size(images, "images")
            compute_keypoints_descriptors(images, sift)
            print_size(images, "images")
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images after: {sys.getrefcount(images)}")
            dump_images_bak(f"data/{image_set_name}/bak/sift-features.pkl", images)
            # remove bak/{image_set_name}/images.pkl
            if mode == Mode.OPTMIZED:
                if os.path.exists(f"data/{image_set_name}/bak/images.pkl"):
                    os.remove(f"data/{image_set_name}/bak/images.pkl")
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/images.pkl removed successfully.")
                else:
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/images.pkl does not exist.")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Feature Extraction: SIFT DONE...")
        last_state = "Images Matching Step"
    
    # 3. Image Matching
    if last_state == "Images Matching Step":
        if os.path.isfile(f"data/{image_set_name}/bak/images-matched.pkl"):
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File [data/{image_set_name}/bak/images-matched.pkl] exists")
            if images: 
                del images
            images: Images = load_images_bak(f"data/{image_set_name}/bak/images-matched.pkl")
        else:
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File [data/{image_set_name}/bak/images-matched.pkl] DO NOT exists")
            log_to_file(f"data/{image_set_name}/logs/tune.log", "Matching images...")
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images before: {sys.getrefcount(images)}")
            print_size(images, "images")
            get_matches(images)
            print_size(images, "images")
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images after: {sys.getrefcount(images)}")
            dump_images_bak(f"data/{image_set_name}/bak/images-matched.pkl", images)
            # remove bak/{image_set_name}/sift-features.pkl
            if mode == Mode.OPTMIZED:
                if os.path.exists(f"data/{image_set_name}/bak/sift-features.pkl"):
                    os.remove(f"data/{image_set_name}/bak/sift-features.pkl")
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/sift-features.pkl removed successfully.")
                else:
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/sift-features.pkl does not exist.")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Done Image Matching Step...")
    
    # 4. Feature Matching
    if os.path.isfile(f"data/{image_set_name}/bak/feature-matching-output.pkl"):
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"File [data/{image_set_name}/bak/feature-matching-output.pkl] exists")
        if images: 
            del images
        images: Images = load_images_bak(f"data/{image_set_name}/bak/feature-matching-output.pkl")
    else:
        log_to_file(f"data/{image_set_name}/logs/tune.log", "File [data/{image_set_name}/bak/feature-matching-output.pkl] Do NOT exists")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Matching features...")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images before: {sys.getrefcount(images)}")
        print_size(images, "images")
        data_feature_matching(images)
        print_size(images, "images")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images after: {sys.getrefcount(images)}")
        dump_images_bak(f"data/{image_set_name}/bak/feature-matching-output.pkl", images)
        # remove bak/{image_set_name}/images-matched.pkl
        if mode == Mode.OPTMIZED:
            if os.path.exists(f"data/{image_set_name}/bak/images-matched.pkl"):
                os.remove(f"data/{image_set_name}/bak/images-matched.pkl")
                log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/images-matched.pkl removed successfully.")
            else:
                log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/images-matched.pkl does not exist.")
    log_to_file(f"data/{image_set_name}/logs/tune.log", "Done Feature Matching Step...")

    # 5. Camera Calibration
    log_to_file(f"data/{image_set_name}/logs/tune.log", "Camera Calibration starts ....")
    if not os.path.isfile(f"data/{image_set_name}/bak/K_matrix.pickle"):
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/K_matrix.pickle does not exist")
        K_matrix = compute_k_matrix(images.images[0].path)
        with open(f"data/{image_set_name}/bak/K_matrix.pickle", 'wb') as f:
            pickle.dump(K_matrix, f)
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/K_matrix.pickle saved successfully")
    else:
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/K_matrix.pickle exists")
        with open(f"data/{image_set_name}/bak/K_matrix.pickle", 'rb') as f:
            K_matrix = pickle.load(f)
    
    # 6. Triangulation
    log_to_file(f"data/{image_set_name}/logs/tune.log", "Triangulation starts ....")
    if os.path.isfile(f"data/{image_set_name}/bak/point-cloud.pkl"):
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/point-cloud.pkl exists")
        with open(f"data/{image_set_name}/bak/point-cloud.pkl", 'rb') as f:
            points_cloud: np.ndarray = pickle.load(f)
    else:
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/point-cloud.pkl does not exist")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Triangulating...")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images before: {sys.getrefcount(images)}")
        print_size(images, "images")
        points_cloud: np.ndarray = generate_point_cloud(images, K_matrix)
        print_size(images, "images")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images after: {sys.getrefcount(images)}")
        # Pickle the point cloud
        with open(f"data/{image_set_name}/bak/point-cloud.pkl", 'wb') as f:
            pickle.dump(points_cloud, f)
    log_to_file(f"data/{image_set_name}/logs/tune.log", "Done Point Cloud Step...")

    log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images before: {sys.getrefcount(images)}")
    print_size(images, "images")
    print_size(points_cloud, "points_cloud")
    images = None
    log_to_file(f"data/{image_set_name}/logs/tune.log", gc.collect())
    print_size(images, "images")
    log_to_file(f"data/{image_set_name}/logs/tune.log", f"Reference count<images>: {sys.getrefcount(images)}")
    log_to_file(f"data/{image_set_name}/logs/tune.log", gc.collect())

    # 7. 3D reconstruction
    log_to_file(f"data/{image_set_name}/logs/tune.log", "3D reconstruction starts ....")
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cloud[:,:3])

    # Save it as a .PLY file
    o3d.io.write_point_cloud(f"data/{image_set_name}/output/point_cloud_before_clustring.ply", pcd)