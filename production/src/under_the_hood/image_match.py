import os
import ssl

import certifi
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from data_structures import Image, Images
from utils import timeit

@timeit
def overwriting_similar_images(images: Images, **kwargs) -> dict[str, list[Image]]:
    image_set_name = kwargs['image_set_name']
    if image_set_name == 'cottage':
        similar_images = {
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
    elif image_set_name == 'fountain':
        similar_images = {
        "0": [images[1], images[2], images[3], images[4], images[5]],
        "1": [images[5], images[6], images[7], images[8]],
        "2": [images[8], images[9], images[10], images[11]]
        }
    elif image_set_name == 'hammer':
        similar_images = {
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
            "12": [images[35], images[36], images[37], images[38]]
        }
    return similar_images


@timeit
def image_matching(images_obj: Images, overwrite:bool =False, **kwargs) -> None:
    def load_image(image_path, target_size=(224, 224)):
        img = keras_image.load_img(image_path, target_size=target_size)
        img = keras_image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    image_set_name = kwargs['image_set_name']
    image_dir = f'../data/{image_set_name}/images'
    image_files = os.listdir(image_dir)
    images = [load_image(os.path.join(image_dir, f)) for f in image_files]
    images = np.vstack(images)

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