# 📝 Pipeline
```Console
*** We have the following 7 steps in our pipeline:
$ prepare_images
- Load Dataset Images
- Compute Mask
$ compute_sift_keypoints_descriptors
$ image_matching
$ data_feature_matching
- Apply crossCheck BF Matcher
- Apply Ransac on BF Matcher Output
- Loop without repeatition using Itertools
$ compute_k_matrix
$ generate_point_cloud
- Recover Pose of referance camera
- Recover rest camera poses using solvePNPRansac
- Apply Triangulation
$ 3D reconstruction
- Use PointsCloud to generate 3D Object (.stl) file
```

## 🏛️ Datasets
- [ ] snow-man.
- [ ] hammer.
- [ ] cottage.
- [ ] fountain

License
-------

This project is licensed under the terms of the GNU General Public License version 3.0 (GPLv3). See the LICENSE file for details.
