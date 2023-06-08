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
#### Annoying Magic numbers
- number of clusters while image matching.
- Ransic threshold.
#### Triangulation Step looks to UnFold Object Points
#### Masking Regular Objects for datasets with UnMasked Images

## 🧐 What we noticed that caused poor feature match:
- Blury images.
- Change in focus.
- Change in distance.

## ❓ Questions
#### Triangulation ?

## 📊 Figures and Plots
#### Snow-Man Points Cloud
![SnowMan_Points_Cloud](https://github.com/ZiadMansourM/photogrammetry/assets/81488138/d9f5cb71-aa91-44b9-9d41-06a1545beb65)

#### Hammer Feature Matches
Good Feature Match |  Bad Feature Match
:--:|:--:
![1_with_2](https://github.com/ZiadMansourM/photogrammetry/assets/81488138/56caa6e4-a6f2-44c8-b072-76c787f28d43) | ![1_with_9](https://github.com/ZiadMansourM/photogrammetry/assets/81488138/6657e433-6fc6-45d8-b254-735801e81b81)

#### Hammer Feature Match Animated Video
https://github.com/ZiadMansourM/photogrammetry/assets/81488138/16dc84d2-a7fd-49f6-9d56-b54a185151aa

#### Triangulation With Hand-Written Labels (Yesterday)
![Hammer_Points_Cloud](https://github.com/ZiadMansourM/photogrammetry/assets/81488138/b51e77ec-d99e-4772-9126-5ac204afa2d5)

#### Traingulation Without Hand-Written Labels (Yesterday)
![Hammer_Points_Cloud](https://github.com/ZiadMansourM/photogrammetry/assets/81488138/d49e9630-ba6d-466a-bbf0-92ced86d6867)


License
-------

This project is licensed under the terms of the GNU General Public License version 3.0 (GPLv3). See the LICENSE file for details.
