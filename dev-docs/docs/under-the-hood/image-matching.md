---
sidebar_position: 4
id: Image Matching
description: Match similar images.
slug: /under-the-hood/image-matching
---

## 📝 Image Matching/Stitching
- [X] Match similar images together into a cluster.
- [X] Images in a cluster has to see the very same points.
```py
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
```
