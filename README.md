# 📝 Pipeline
```Console
*** We have the following 7 steps in our pipeline:
$ prepare_images
$ compute_keypoints_descriptors
$ image_matching
$ data_feature_matching
$ compute_k_matrix
$ generate_point_cloud
$ 3D reconstruction
```

## 🏛️ Datasets
- [X] snow-man.
- [X] hammer.
- [ ] cottage

> ? Something simplier and on our own terms

## ❌ Problems
#### Poor Feature Matches
#### Annoying Magic numbers
- number of clusters.
- Ransic threshold.

## 🧐 What we noticed that caused poor feature match:
- Blury images.
- Change in focus.
- Change in distance.


## ❓ Questions
#### Shall we use maske and where ?
#### Scalling ?
#### How to improve feature match? deep feature?
 
