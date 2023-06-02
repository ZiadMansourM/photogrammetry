```python
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

class FeatureMatches:
    def __init__(self, image_one: Image, image_two: Image, matches: list[OpenCV.DMatch]):
        self.image_one: Image = image_one
        self.image_two: Image = image_two
        self.matches: list[OpenCV.DMatch] = matches

class Images:
    def __init__(self, images: list[Image], image_set_name: str):
        self.id = uuid.uuid4()
        self.images: list[Image] = images
        self.image_set_name: str = image_set_name
        self.feature_matches: list[FeatureMatches] = []
        self.similar_images: dict[list[Image]] = {}
        self.num_clusters: int = 50

def check_coherent_rotation(R: np.ndarray) -> bool:
    epsilon = 1e-6
    return np.abs(np.linalg.det(R) - 1.0) <= epsilon

def find_camera_matrices(K: np.ndarray, keypoints_one: np.ndarray, keypoints_two: np.ndarray, matches: List) -> Tuple[np.ndarray, np.ndarray]:
    E, mask = OpenCV.findEssentialMat(keypoints_one, keypoints_two, K, method=OpenCV.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = OpenCV.recoverPose(E, keypoints_one, keypoints_two, K)
    if not check_coherent_rotation(R):
        return None, None
    return np.hstack((R, t))

def generate_point_cloud_general(images: Images, K_matrix: np.ndarray, **kwargs) -> np.ndarray:
    point_cloud = []
    for feature_match in images.feature_matches:
        image_one = feature_match.image_one
        image_two = feature_match.image_two
        keypoints_one = np.array([image_one.keypoints[m.queryIdx].pt for m in feature_match.matches])
        keypoints_two = np.array([image_two.keypoints[m.trainIdx].pt for m in feature_match.matches])
        P2 = find_camera_matrices(K_matrix, keypoints_one, keypoints_two, feature_match.matches)
        if P2 is None:
            continue
        P1 = K_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K_matrix @ P2
        points_4D = OpenCV.triangulatePoints(P1, P2, keypoints_one.T, keypoints_two.T)
        points_3D = (points_4D / points_4D[3])[:3]
        point_cloud.append(points_3D)
    return np.hstack(point_cloud).T
```

The above code is good at Estimating the camera motion from a pair of images and Reconstructing the scene, but now we want to add support for Reconstruction from many views.
to do this we will use incremental 3d reconstruction with camera resection or camera pose estimation, known as "Perspective N-Point PnP".
the steps we thought of:
1) get 3d to 2d correspondence 
2) use solvePnPRansac
3) Now that we have a new P2 matrix, we can simply populate our point cloud with more 3D points.

modfiy the above code to support Reconstruction from many views.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

best views selection.

we get more image candidates for next best views selection. We iterate like that, adding cameras and triangulating new 2D features into 3D points and removing 3D points that became invalidated, until we can’t localize new views.