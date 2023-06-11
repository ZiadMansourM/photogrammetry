import os
from typing import Optional, Final
import cv2 as OpenCV
from matplotlib import pyplot as plt
import numpy as np
import uuid
import pickle
import time
import sys
import os
import numpy as np
import gc
import open3d as o3d
from collections import Counter


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
        image_set_name = kwargs['image_set_name']
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Started {func.__name__}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Done {func.__name__} took {end_time - start_time:,} seconds to execute.")
        return result
    return wrapper


""" Data Structures """
class Image:
    def __init__(self, img_id, rgb_image, gray_image, mask, keypoints, descriptors, path):
        self.img_id: int = int(img_id)
        self.unique_id: uuid = uuid.uuid4()
        self.rgb_image: Image = rgb_image
        self.gray_image: Image = gray_image
        self.mask: Image = mask
        self.keypoints: list[OpenCV.KeyPoint] = keypoints
        self.descriptors: np.ndarray = descriptors
        self.path: str = path

    @property
    def length(self):
        return f"{len(self.keypoints)}" if len(self.keypoints) == len(self.descriptors) else f"{len(self.keypoints)}, {len(self.descriptors)}"
    
    def draw_sift_features(self):
        image_with_sift = OpenCV.drawKeypoints(self.rgb_image, self.keypoints, None, flags=OpenCV.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(image_with_sift)
        plt.title("Image with SIFT Features")
        plt.axis('off')
        plt.show()
        
    
    def save_sift_features(self):
        output_filename = f"data/{self.image_set_name}/output/sift/{self.img_id}_sift_features.jpg"
        image_with_sift = OpenCV.drawKeypoints(self.rgb_image, self.keypoints, None, flags=OpenCV.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        OpenCV.imwrite(output_filename, OpenCV.cvtColor(image_with_sift, OpenCV.COLOR_RGB2BGR))

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
        
    def display_mask_image(self, title: Optional[str] = None):
        image = self.mask
        plt.gray()
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        # plt.axes('off')
        plt.show()
        
    def display_dialated_image(self, title: Optional[str] = None):
        print(self.mask.shape)
        print(self.rgb_image.shape)
        image = OpenCV.bitwise_and(self.rgb_image, self.rgb_image, mask=self.mask)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        # plt.axis('off')
        plt.show()
        
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
        
    def animate_matches(self, output_filename: str) -> None:
        import subprocess
        for match in self.matches:
            combined_image = OpenCV.hconcat([
                self.image_one.rgb_image,
                self.image_two.rgb_image
            ])
            x1, y1 = self.image_one.keypoints[match.queryIdx].pt
            x2, y2 = self.image_two.keypoints[match.trainIdx].pt
            # Write match.queryIdx at the top left corner
            OpenCV.putText(
                combined_image,
                f"{match.queryIdx}",
                (50, 150),  # position: 10 pixels from left, 20 pixels from top
                OpenCV.FONT_HERSHEY_SIMPLEX,  # font
                5,  # font scale
                (0, 255, 0),  # font color (green)
                5,  # thickness
                OpenCV.LINE_AA  # line type
            )
            # Write match.trainIdx at the top right corner
            image_two_width = self.image_one.rgb_image.shape[1]
            OpenCV.putText(
                combined_image,
                f"{match.trainIdx}",
                (image_two_width + 50, 150),  # position: 10 pixels from right, 20 pixels from top
                OpenCV.FONT_HERSHEY_SIMPLEX,  # font
                5,  # font scale
                (0, 255, 0),  # font color (green)
                5,  # thickness
                OpenCV.LINE_AA  # line type
            )
            # Draw a line connecting the matched keypoints
            OpenCV.line(
                combined_image, 
                (int(x1), int(y1)), 
                (int(x2) + self.image_one.rgb_image.shape[1], int(y2)), 
                (0, 255, 0), 
                1
            )
            OpenCV.imwrite(
                f"{output_filename}/{match.queryIdx}_{match.trainIdx}.jpg",
                combined_image,
            )
        framerate = 120
        # Get a list of image files in the directory
        image_files = [f for f in os.listdir(output_filename) if f.endswith(".jpg")]
        image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        # Create a temporary file with a list of input images
        with open("input_files.txt", "w") as f:
            for image_file in image_files:
                f.write(f"file '{os.path.join(output_filename, image_file)}'\n")
        # Run FFmpeg command to create a video
        command = f'ffmpeg -y -f concat -safe 0 -i "input_files.txt" -framerate {framerate} -c:v libx264 -pix_fmt yuv420p "{output_filename}/output.mp4"'
        subprocess.run(command, shell=True, check=True)
        # Remove temporary file
        os.remove("input_files.txt")

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
        self.similar_images: dict[list[Image]] = {}
        self.num_clusters: int = 50

    def save_feature_matches(self):
        for match in self.feature_matches:
            match.draw_matches(f"data/{self.image_set_name}/output/feature-match/{match.image_one.img_id}_{match.image_two.img_id}.jpg")

    def __len__(self):
        return len(self.images)
    
    def display_similar_images(self, key):
        print(f"cluster {key}")
        print("-----------------------------------------------------")
        for value in self.similar_images[key]:
            print(value)
            rgb_image = OpenCV.cvtColor(OpenCV.imread(value.path), OpenCV.COLOR_BGR2RGB)
            plt.imshow(rgb_image)
            plt.title(value.path)
            plt.axis('off')
            plt.show()

    def save_similar_images(self):
        for cluster in self.similar_images.keys():
            if not os.path.exists(f"data/{self.image_set_name}/output/image-match/{cluster}"):
                os.makedirs(f"data/{self.image_set_name}/output/image-match/{cluster}")
            for value in self.similar_images[cluster]:
                OpenCV.imwrite(f"data/{self.image_set_name}/output/image-match/{cluster}/{value.img_id}.jpg", value.rgb_image)

    def __getitem__(self, key: int) -> Image:
        for image in self.images:
            if image.img_id == key:
                return image
        raise KeyError(f'Image with img_id {key} not found.')


def dump_images_bak(images_file_path: str, images: Images) -> None:
    """ Dump images to a file """
    with open(images_file_path, "wb") as file:
        pickle.dump(images, file)


def load_images_bak(images_file_path: str) -> Images:
    """ Load images from a file """
    with open(images_file_path, "rb") as file:
        images = pickle.load(file)
    return images


@timeit
def prepare_images(create_mask = True, **kwargs) -> Images:
    image_set_name = kwargs['image_set_name']
    folder_path = f"data/{image_set_name}"
    images: Images = Images([], folder_path.split("/")[-1])
    files: list[str] = list(
        filter(
            lambda file: ".jpg" in file, os.listdir(f"{folder_path}/images")
        )
    )
    if create_mask:
        from rembg import remove
        for file in files:
            image_path = f"{folder_path}/images/{file}"
            rgb_image = OpenCV.cvtColor(OpenCV.imread(image_path), OpenCV.COLOR_BGR2RGB)
            gray_image = OpenCV.cvtColor(rgb_image, OpenCV.COLOR_RGB2GRAY)
            mask = remove(rgb_image)
            mask = OpenCV.cvtColor(mask, OpenCV.COLOR_RGB2GRAY)
            mask[mask > 0] = 255
            OpenCV.imwrite(f"{folder_path}/masks/{file}", mask)
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = OpenCV.dilate(mask, kernel, iterations=20)
            images.images.append(Image(file.split(".")[0], rgb_image, gray_image, dilated_mask, [], [], image_path))
    else:
        for file in files:
            image_path = f"{folder_path}/images/{file}"
            mask_path = f"{folder_path}/masks/{file}"
            rgb_image = OpenCV.cvtColor(OpenCV.imread(image_path), OpenCV.COLOR_BGR2RGB)
            gray_image = OpenCV.cvtColor(rgb_image, OpenCV.COLOR_RGB2GRAY)
            mask = OpenCV.imread(mask_path, OpenCV.IMREAD_GRAYSCALE)
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = OpenCV.dilate(mask, kernel, iterations=20)
            images.images.append(Image(file.split(".")[0], rgb_image, gray_image, dilated_mask, [], [], image_path))
    return images


@timeit
def compute_keypoints_descriptors(images: list[Image], SIFT: OpenCV.SIFT, **kwargs) -> None:
    image_set_name = kwargs['image_set_name']
    for img in images.images:
        keypoints: list[OpenCV.KeyPoint]
        descriptors: np.ndarray
        dialated_image = OpenCV.bitwise_and(img.gray_image, img.gray_image, mask=img.mask)
        keypoints, descriptors = SIFT.detectAndCompute(dialated_image, None)
        img.keypoints = keypoints
        img.descriptors = descriptors
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Img({img.img_id}, {img.path}) has {len(img.keypoints)} keypoints and {len(img.descriptors)} descriptors.")


@timeit
def overwriting_similar_images(images: Images, **kwargs) -> dict[str, list[Image]]:
    image_set_name = kwargs['image_set_name']
    similar_images_dict: dict[str, dict[str,list[Image]]] = {}
    if image_set_name == "hammer":
        similar_images_dict[image_set_name] = {
                "0": [images[1], images[2], images[3], images[4], images[5]],
                "1": [images[5], images[6], images[7]],
                "2": [images[7], images[8], images[9], images[10], images[11], images[12]],
                "3": [images[12], images[13], images[14], images[15]],
                "4": [images[15], images[16], images[17], images[18]],
                "5": [images[18], images[19], images[20]],
                "6": [images[20], images[21], images[22], images[23]],
                "7": [images[23], images[24], images[25]],
                "8": [images[25], images[26], images[27], images[28]],
                "9": [images[28], images[29], images[30], images[31]],
                "10": [images[31], images[32], images[33]],
                "11": [images[33], images[34], images[35]],
                "12": [images[35], images[36], images[37], images[38]],
        }
    elif image_set_name == "cottage":
        similar_images_dict[image_set_name] = {
                "0": [images[1], images[2], images[3]],
                "1": [images[3], images[4], images[5]],
                "2": [images[5], images[6], images[7]],
                "3": [images[7], images[8], images[9]],
                "4": [images[9], images[10], images[11]],
                "5": [images[11], images[12], images[13]],
                "6": [images[13], images[14], images[15]],
                "7": [images[15], images[16], images[17]],
                "8": [images[17], images[18], images[19]],
                "9": [images[19], images[20], images[21]],
                "10": [images[21], images[22], images[23]],
                "11": [images[23], images[24], images[25]],
                "12": [images[25], images[26], images[27]],
                "13": [images[27], images[28], images[29]],
                "14": [images[29], images[30]]
        }
    elif image_set_name == "fountain":
        similar_images_dict[image_set_name] = {
                "0": [images[1], images[2], images[3], images[4], images[5]],
                "1": [images[5], images[6], images[7], images[8]],
                "2": [images[8], images[9], images[10], images[11]]
        },
    return similar_images_dict[image_set_name]


@timeit
def image_matching(images_obj: Images, overwrite:bool =False, **kwargs) -> None:
    import os
    import numpy as np
    from sklearn.cluster import KMeans
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
    def load_image(image_path, target_size=(224, 224)):
        img = keras_image.load_img(image_path, target_size=target_size)
        img = keras_image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    image_set_name = kwargs['image_set_name']
    image_dir = f'data/{image_set_name}/images'
    image_files = os.listdir(image_dir)
    images = [load_image(os.path.join(image_dir, f)) for f in image_files]
    images = np.vstack(images)

    import ssl
    import certifi

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = ssl._create_unverified_context
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = model.predict(images)

    kmeans = KMeans(n_clusters=images_obj.num_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)

    for i, cluster in enumerate(clusters):
        if cluster not in images_obj.similar_images:
            images_obj.similar_images[cluster] = []
        images_obj.similar_images[cluster].append(images_obj[int(image_files[i].split(".")[0])])

    if overwrite:
        images_obj.similar_images = overwriting_similar_images()
    else:
        images_obj.similar_images = {key: value for key, value in images_obj.similar_images.items() if len(value) > 1}


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
    for key, values in images.similar_images.items():
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Started Feature Match for cluster number {key}:")
        for image, matched_image in itertools.combinations(values, 2):
            feature_matching_output = feature_matching(image.descriptors, matched_image.descriptors, **kwargs)
            ransac_output = apply_ransac(feature_matching_output, image.keypoints, matched_image.keypoints, threshold=150, **kwargs)
            images.feature_matches.append(FeatureMatches(image, matched_image, ransac_output))
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"({image.img_id}, {matched_image.img_id}) with {len(ransac_output)} / {len(feature_matching_output)}.")


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
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Using Images {image_one.img_id} and {image_two.img_id} in find_next_camera_matrices")
    local_dict: dict[np.ndarray, np.ndarray] = find_3D_2D_correspondences(image_two, images.feature_matches, global_dict, image_set_name=image_set_name)
    objectPoints = np.array(list(local_dict.keys())).reshape(-1, 3)
    imagePoints = np.array(list(local_dict.values())).reshape(-1, 2)
    log_to_file(f"data/{image_set_name}/logs/tune.log", f"Found {objectPoints.shape[0]} 3D Points and {imagePoints.shape[0]} Image Points 3D-2D correspondences")
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
    log_to_file(f"data/{image_set_name}/logs/tune.log", f"Computed {points_3D.shape[1]} 3D Points for Image pairs {image_one.img_id} and {image_two.img_id}")
    return points_3D


@timeit
def find_cluster_feature_matches( 
        images: Images, 
        values: list[Image],
        **kwargs
    ) -> list[FeatureMatches]: # [1,2,3] ----> [1,2],[1,3]
    image_set_name = kwargs['image_set_name']
    cluster_reference_image = values[0]
    cluster_feature_matches: list[FeatureMatches] = []
    import itertools
    for image, matched_image in itertools.combinations(values, 2):
        if image.img_id != cluster_reference_image.img_id:
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Breaking itertools loop for {image.img_id} and {matched_image.img_id} in find_cluster_feature_matches\n")
            break
        else:
            appended_pair: FeatureMatches = next(
                fm for fm in images.feature_matches
                if fm.image_one.img_id == image.img_id and fm.image_two.img_id == matched_image.img_id
            )
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"appended_pair: {appended_pair}")
            cluster_feature_matches.append(appended_pair)
    return cluster_feature_matches


@timeit
def generate_points_cloud(images: Images, K_matrix: np.ndarray, **kwargs) -> np.ndarray:
    # sourcery skip: low-code-quality
    points_cloud: list[list[np.ndarray]] = []
    global_dict: dict[np.ndarray, set[tuple[int]]] = {}
    camera_matrices: list[np.ndarray] = [(np.eye(3), np.zeros((3, 1)))]
    image_set_name = kwargs['image_set_name']
    for cluster, values in images.similar_images.items():
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"--------------------- Entering Cluster {cluster} ---------------------")
        cluster_feature_matches:list[FeatureMatches] = find_cluster_feature_matches(images, values, image_set_name=image_set_name)
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"cluster_feature_matches: {cluster_feature_matches}\n")
        if cluster == list(images.similar_images.keys())[0]: # First cluster
            P1 = K_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
            for feature_match in cluster_feature_matches:
                image_one = feature_match.image_one
                image_two = feature_match.image_two
                keypoints_one = np.array([image_one.keypoints[m.queryIdx].pt for m in feature_match.matches])
                keypoints_two = np.array([image_two.keypoints[m.trainIdx].pt for m in feature_match.matches])
                if feature_match == cluster_feature_matches[0]:  # First Feature Match Pair in the First Cluster, where we use recoverPose
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"Using Images {image_one.img_id} and {image_two.img_id} in recoverPose")
                    R, t = find_initial_camera_matrices(K_matrix, keypoints_one, keypoints_two, image_set_name=image_set_name)
                    P2 = K_matrix @ np.hstack((R, t))
                    camera_matrices.append((R, t))
                else:
                    R, tvec = find_next_camera_matrices(images, image_one, image_two, K_matrix, global_dict, image_set_name=image_set_name)
                    P2 = K_matrix @ np.hstack((R, tvec))
                    camera_matrices.append((R, tvec))
                points_3D = compute_points_3D(P1, P2, image_one, image_two, keypoints_one, keypoints_two, global_dict, image_set_name=image_set_name)
                points_cloud.append(points_3D)
                log_to_file(f"data/{image_set_name}/logs/tune.log", f"Global Dict 3D Points Size: {len(global_dict.keys())} \n")
        else: # Next Clusters
            for feature_match in cluster_feature_matches:
                image_one = feature_match.image_one
                image_two = feature_match.image_two
                keypoints_one = np.array([image_one.keypoints[m.queryIdx].pt for m in feature_match.matches])
                keypoints_two = np.array([image_two.keypoints[m.trainIdx].pt for m in feature_match.matches])
                if feature_match == cluster_feature_matches[0]: # First Iteration of the next Cluster
                # Computing new P1 for the new cluster
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"Entered First Iteration of the cluster {cluster}")
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"Using Image {image_one.img_id} as Reference Image in cluster {cluster} to compute P1 for cluster {cluster}")
                    P1_R, P1_tvec = find_next_camera_matrices(images, None, image_one, K_matrix, global_dict, image_set_name=image_set_name)
                    P1 = K_matrix @ np.hstack((P1_R, P1_tvec))
                R, tvec = find_next_camera_matrices(images, image_one, image_two, K_matrix, global_dict, image_set_name=image_set_name)
                P2 = K_matrix @ np.hstack((R, tvec))
                camera_matrices.append((R, tvec))
                points_3D = compute_points_3D(P1, P2, image_one, image_two, keypoints_one, keypoints_two, global_dict, image_set_name=image_set_name)
                points_cloud.append(points_3D)
                log_to_file(f"data/{image_set_name}/logs/tune.log", f"Global Dict 3D Points Size: {len(global_dict.keys())} \n")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"--------------------- End of cluster {cluster} ---------------------\n\n")

    points_cloud = np.hstack(points_cloud).T
    log_to_file(f"data/{image_set_name}/logs/tune.log", "Done generating points cloud")
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


def run(image_set_name: str) -> None:  # sourcery skip: low-code-quality
    import enum
    class Mode(enum.Enum):
        OPTMIZED = "optimized"
        DEBUG = "debug"
    output_files_3D: list[str] = [
        f"data/{image_set_name}/output/triangulate/points_cloud.stl",
        f"data/{image_set_name}/output/triangulate/core_points.stl",
        f"data/{image_set_name}/output/triangulate/camera_proj.stl",
        f"data/{image_set_name}/output/triangulate/mesh.stl"
    ]
    output_files_triangulation: list[str] = [
        f"data/{image_set_name}/bak/k-matrix.pkl",
        f"data/{image_set_name}/bak/points-cloud.pkl",
        f"data/{image_set_name}/bak/core-points.pkl",
        f"data/{image_set_name}/bak/camera-proj.pkl",
        f"data/{image_set_name}/bak/hdbscan-model.pkl",
    ]
    log_to_file(f"data/{image_set_name}/logs/tune.log", "Welcome ScanMate...")
    mode: enum = Mode.OPTMIZED
    NUM_CLUSTERS: Final[int] = 1
    log_to_file(f"data/{image_set_name}/logs/tune.log", f"Running image_set_name {image_set_name} in {mode} mode...")
    images: Optional[Images] = None
    """ Reloading the last state """
    last_state: str
    if all(
        os.path.isfile(output_file)
        for output_file in output_files_3D
    ):
        last_state = "3D Reconstruction Step"
    elif all(
        os.path.isfile(output_file)
        for output_file in output_files_triangulation
    ):
        last_state = "Triangulation Step"
    elif os.path.isfile(f"data/{image_set_name}/bak/feature-matching-output.pkl"):
        last_state = "Feature Matching Step"
    elif os.path.isfile(f"data/{image_set_name}/bak/matched-images.pkl"):
        last_state = "Images Matching Step"
    elif os.path.isfile(f"data/{image_set_name}/bak/sift-features.pkl"):
        last_state = "SIFT Features Step"
    else:
        last_state = "Images Loading Step"
    log_to_file(f"data/{image_set_name}/logs/tune.log", f"Last state for {image_set_name} is {last_state}.")
    if last_state == "Images Loading Step":
        if os.path.isfile(f"data/{image_set_name}/bak/images.pkl"):
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File [data/{image_set_name}/bak/sift-images.pkl] exists")
            log_to_file(f"data/{image_set_name}/logs/tune.log", "Loading images from pickle file...")
            images: Images = load_images_bak(f"data/{image_set_name}/bak/images.pkl")
        else:
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File [data/{image_set_name}/bak/images.pkl] does not exist")
            log_to_file(f"data/{image_set_name}/logs/tune.log", "Loading images from images directory...")
            images: Images = prepare_images(create_mask=True, image_set_name=image_set_name)
            log_to_file(f"data/{image_set_name}/logs/tune.log", "Saving images to pickle file...")
            dump_images_bak(f"data/{image_set_name}/bak/images.pkl", images)
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Images loaded successfully")
        last_state = "SIFT Features Step"
    if last_state == "SIFT Features Step":
        if os.path.isfile(f"data/{image_set_name}/bak/sift-features.pkl"):
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File [data/{image_set_name}/bak/sift-features.pkl] exists")
            if images: 
                del images
            images: Images = load_images_bak(f"data/{image_set_name}/bak/sift-features.pkl")
        else:
            log_to_file(f"data/{image_set_name}/logs/tune.log", "File [data/{image_set_name}/bak/sift-features.pkl] DO NOT exists")
            log_to_file(f"data/{image_set_name}/logs/tune.log", "Extracting SIFT features...")
            sift = OpenCV.SIFT_create(contrastThreshold=0.01)
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images before: {sys.getrefcount(images)}")
            print_size(f"data/{image_set_name}/logs/tune.log", images, "images")
            compute_keypoints_descriptors(images, sift, image_set_name=image_set_name)
            print_size(f"data/{image_set_name}/logs/tune.log", images, "images")
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images after: {sys.getrefcount(images)}")
            dump_images_bak(f"data/{image_set_name}/bak/sift-features.pkl", images)
            if mode == Mode.OPTMIZED:
                if os.path.exists(f"data/{image_set_name}/bak/images.pkl"):
                    os.remove(f"data/{image_set_name}/bak/images.pkl")
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/images.pkl removed successfully.")
                else:
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/images.pkl does not exist.")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Feature Extraction: SIFT DONE...")
        last_state = "Images Matching Step"
    if last_state == "Images Matching Step":
        if os.path.isfile(f"data/{image_set_name}/bak/matched-images.pkl"):
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File [data/{image_set_name}/bak/matched-images.pkl] exists")
            if images: 
                del images
            images: Images = load_images_bak(f"data/{image_set_name}/bak/matched-images.pkl")
        else:
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File [data/{image_set_name}/bak/matched-images.pkl] DO NOT exists")
            log_to_file(f"data/{image_set_name}/logs/tune.log", "Matching images...")
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images before: {sys.getrefcount(images)}")
            print_size(f"data/{image_set_name}/logs/tune.log", images, "images")
            images.num_clusters = NUM_CLUSTERS
            overwrite = True
            image_matching(images, overwrite=overwrite, image_set_name=image_set_name)
            log_to_file(f"data/{image_set_name}/logs/tune.log", "image matching done")
            images.save_similar_images()
            log_to_file(f"data/{image_set_name}/logs/tune.log", "saved image clusters")
            print_size(f"data/{image_set_name}/logs/tune.log", images, "images")
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images after: {sys.getrefcount(images)}")
            dump_images_bak(f"data/{image_set_name}/bak/matched-images.pkl", images)
            if mode == Mode.OPTMIZED:
                if os.path.exists(f"data/{image_set_name}/bak/sift-features.pkl"):
                    os.remove(f"data/{image_set_name}/bak/sift-features.pkl")
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/sift-features.pkl removed successfully.")
                else:
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/sift-features.pkl does not exist.")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Done Image Matching Step...")
        last_state = "Feature Matching Step"
    if last_state == "Feature Matching Step":
        if os.path.isfile(f"data/{image_set_name}/bak/feature-matching-output.pkl"):
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File [data/{image_set_name}/bak/feature-matching-output.pkl] exists")
            if images: 
                del images
            images: Images = load_images_bak(f"data/{image_set_name}/bak/feature-matching-output.pkl")
        else:
            log_to_file(f"data/{image_set_name}/logs/tune.log", "File [data/{image_set_name}/bak/feature-matching-output.pkl] Do NOT exists")
            log_to_file(f"data/{image_set_name}/logs/tune.log", "Matching features...")
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images before: {sys.getrefcount(images)}")
            print_size(f"data/{image_set_name}/logs/tune.log", images, "images")
            data_feature_matching(images, image_set_name=image_set_name)
            log_to_file(f"data/{image_set_name}/logs/tune.log", "done feature matching")
            images.save_feature_matches()
            log_to_file(f"data/{image_set_name}/logs/tune.log", "saved feature matching")
            print_size(f"data/{image_set_name}/logs/tune.log", images, "images")
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images after: {sys.getrefcount(images)}")
            dump_images_bak(f"data/{image_set_name}/bak/feature-matching-output.pkl", images)
            if mode == Mode.OPTMIZED:
                if os.path.exists(f"data/{image_set_name}/bak/matched-images.pkl"):
                    os.remove(f"data/{image_set_name}/bak/matched-images.pkl")
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/matched-images.pkl removed successfully.")
                else:
                    log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/matched-images.pkl does not exist.")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Done Feature Matching Step...")
        last_state = "Triangulation Step"
    if last_state == "Triangulation Step":
        # 5. Camera Calibration
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Camera Calibration starts ....")
        if not os.path.isfile(f"data/{image_set_name}/bak/k-matrix.pkl"):
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/k-matrix.pkl does not exist")
            K_matrix = compute_k_matrix(images.images[0].path, image_set_name=image_set_name)
            with open(f"data/{image_set_name}/bak/k-matrix.pkl", 'wb') as f:
                pickle.dump(K_matrix, f)
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/k-matrix.pkl saved successfully")
        else:
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/k-matrix.pkl exists")
            with open(f"data/{image_set_name}/bak/k-matrix.pkl", 'rb') as f:
                K_matrix = pickle.load(f)
        # 6. Triangulation
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Triangulation starts ....")
        if os.path.isfile(f"data/{image_set_name}/bak/points-cloud.pkl"):
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/points-cloud.pkl exists")
            with open(f"data/{image_set_name}/bak/points-cloud.pkl", 'rb') as f:
                points_cloud: np.ndarray = pickle.load(f)
            with open(f"data/{image_set_name}/bak/camera-proj.pkl", 'rb') as f:
                camera_matrices: np.ndarray = pickle.load(f)
        else:
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"File data/{image_set_name}/bak/points-cloud.pkl does not exist")
            log_to_file(f"data/{image_set_name}/logs/tune.log", "Triangulating...")
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images before: {sys.getrefcount(images)}")
            print_size(f"data/{image_set_name}/logs/tune.log", images, "images")
            points_cloud, camera_matrices = generate_points_cloud(images, K_matrix, image_set_name=image_set_name)
            print_size(f"data/{image_set_name}/logs/tune.log", images, "images")
            log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images after: {sys.getrefcount(images)}")
            with open(f"data/{image_set_name}/bak/points-cloud.pkl", 'wb') as f:
                pickle.dump(points_cloud, f)
            with open(f"data/{image_set_name}/bak/camera-proj.pkl", 'wb') as f:
                pickle.dump(camera_matrices, f)
        # Cleaning memory before clustering
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Done Point Cloud Step...")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Ref count of images before: {sys.getrefcount(images)}")
        print_size(f"data/{image_set_name}/logs/tune.log", images, "images")
        print_size(f"data/{image_set_name}/logs/tune.log", points_cloud, "points_cloud")
        images = None
        log_to_file(f"data/{image_set_name}/logs/tune.log", gc.collect())
        print_size(f"data/{image_set_name}/logs/tune.log", images, "images")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Reference count<images>: {sys.getrefcount(images)}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", gc.collect())
        # 7. Points Clustering
        log_to_file(f"data/{image_set_name}/logs/tune.log", "started clustring....")
        import hdbscan
        start_time = time.time()
        hdbscan_model = hdbscan.HDBSCAN().fit(points_cloud)
        end_time = time.time()
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"time taken: {end_time - start_time:,} seconds")
        with open(f"data/{image_set_name}/bak/hdbscan-model.pkl", 'wb') as f:
            pickle.dump(hdbscan_model, f)
        log_to_file(f"data/{image_set_name}/logs/tune.log", "File hdbscan-model.pkl saved successfully...")
        print_size(f"data/{image_set_name}/logs/tune.log", hdbscan_model, "hdbscan_model")
        # Get the cluster labels for each point
        labels = hdbscan_model.labels_
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Labels Done...")
        # Get the indices of the core points (i.e., points that are part of a dense region)
        core_indices = np.where(labels != -1)[0]
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Core Indicies Done...")
        # Get the coordinates of the core points
        core_points = points_cloud[core_indices, :]
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Core Points Done...")
        # Get the indices of the outlier points (i.e., points that are not part of any dense region)
        outlier_indices = np.where(labels == -1)[0]
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Outlier Indicies Done...")
        # Get the coordinates of the outlier points
        outlier_points = points_cloud[outlier_indices, :]
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Outlier Points Done...")
        # Log the number of clusters and the number of outlier points
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Number of clusters: {len(np.unique(labels))-1:,}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Number of core points: {len(core_indices):,}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Number of outlier points: {len(outlier_indices):,}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Number of total points: {len(core_indices) + len(outlier_indices):,}")
        with open(f"data/{image_set_name}/bak/core_points.pkl", 'wb') as f:
            pickle.dump(core_points, f)
        log_to_file(f"data/{image_set_name}/logs/tune.log", "File core_points.pkl saved successfully...")
        print_size(f"data/{image_set_name}/logs/tune.log", core_points, "core_points")
        # 8. 3D Reconstruction
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"points_cloud.shape: {points_cloud.shape:,}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Number of cameras detected: {len(camera_matrices):,}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"core_points.shape: {core_points.shape:,}")
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
        point_cloud_file = f"data/{image_set_name}/output/triangulate/points_cloud.ply"
        o3d.io.write_point_cloud(point_cloud_file, points_cloud_stl)
        point_cloud_file = f"data/{image_set_name}/output/triangulate/core_points.ply"
        o3d.io.write_point_cloud(point_cloud_file, core_points_stl)
        mesh_file = f"data/{image_set_name}/output/triangulate/camera_proj.ply"
        o3d.io.write_triangle_mesh(mesh_file, combined_mesh)
        # 9. Meshing
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
        o3d.io.write_triangle_mesh(f"data/{image_set_name}/output/triangulate/mesh.stl", dec_mesh)
        # 10. Further Analysis
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Analysis of X, Y, Z of Points cloud")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"X<{len(points_cloud[:,0]):,}>: {points_cloud[:,0].min():,} to {points_cloud[:,0].max():,}")
        x_counter = Counter(points_cloud[:,0])
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"We have {len(x_counter):,} unique X values")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Most Common X: {x_counter.most_common(1)}, Least Two Common X: {x_counter.most_common()[:-3:-1]}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "-----------------------------------------------------")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Y<{len(points_cloud[:,1]):,}>: {points_cloud[:,1].min():,} to {points_cloud[:,1].max():,}")
        y_counter = Counter(points_cloud[:,1])
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"We have {len(y_counter):,} unique Y values")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Most Common Y: {y_counter.most_common(1)}, Least Two Common Y: {y_counter.most_common()[:-3:-1]}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "-----------------------------------------------------")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Z<{len(points_cloud[:,2]):,}>: {points_cloud[:,2].min():,} to {points_cloud[:,2].max():,}")
        z_counter = Counter(points_cloud[:,2])
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"We have {len(z_counter):,} unique Z values")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Most Common Z: {z_counter.most_common(1)}, Least Two Common Y: {z_counter.most_common()[:-3:-1]}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "-----------------------------------------------------")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Analysis of X, Y, Z of Core Points")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"X<{len(core_points[:,0]):,}>: {core_points[:,0].min():,} to {core_points[:,0].max():,}")
        x_counter = Counter(core_points[:,0])
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"We have {len(x_counter):,} unique X values")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Most Common X: {x_counter.most_common(1)}, Least Two Common X: {x_counter.most_common()[:-3:-1]}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "-----------------------------------------------------")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Y<{len(core_points[:,1]):,}>: {core_points[:,1].min():,} to {core_points[:,1].max():,}")
        y_counter = Counter(core_points[:,1])
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"We have {len(y_counter):,} unique Y values")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Most Common Y: {y_counter.most_common(1)}, Least Two Common Y: {y_counter.most_common()[:-3:-1]}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "-----------------------------------------------------")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Z<{len(core_points[:,2]):,}>: {core_points[:,2].min():,} to {core_points[:,2].max():,}")
        z_counter = Counter(core_points[:,2])
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"We have {len(z_counter):,} unique Z values")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Most Common Z: {z_counter.most_common(1)}, Least Two Common Y: {z_counter.most_common()[:-3:-1]}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "-----------------------------------------------------")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "Analysis of X, Y, Z of Outliers Points")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"X<{len(outlier_points[:,0]):,}>: {outlier_points[:,0].min():,} to {outlier_points[:,0].max():,}")
        x_counter = Counter(outlier_points[:,0])
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"We have {len(x_counter):,} unique X values")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Most Common X: {x_counter.most_common(1)}, Least Two Common X: {x_counter.most_common()[:-3:-1]}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "-----------------------------------------------------")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Y<{len(outlier_points[:,1]):,}>: {outlier_points[:,1].min():,} to {outlier_points[:,1].max():,}")
        y_counter = Counter(outlier_points[:,1])
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"We have {len(y_counter):,} unique Y values")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Most Common Y: {y_counter.most_common(1)}, Least Two Common Y: {y_counter.most_common()[:-3:-1]}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "-----------------------------------------------------")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Z<{len(outlier_points[:,2]):,}>: {outlier_points[:,2].min():,} to {outlier_points[:,2].max():,}")
        z_counter = Counter(outlier_points[:,2])
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"We have {len(z_counter):,} unique Z values")
        log_to_file(f"data/{image_set_name}/logs/tune.log", f"Most Common Z: {z_counter.most_common(1)}, Least Two Common Y: {z_counter.most_common()[:-3:-1]}")
        log_to_file(f"data/{image_set_name}/logs/tune.log", "-----------------------------------------------------")
