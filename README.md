<div align="center">

<img src="https://github.com/ZiadMansourM/photogrammetry/assets/64917739/b8d80fd8-e261-4ed5-9afa-f0cdf6d79022" alt="ScanMate" width="400" height="400">

![Code Convention](https://github.com/ZiadMansourM/photogrammetry/actions/workflows/type-check.yml/badge.svg)
[![Build and Deploy Scanmate Docs](https://github.com/ZiadMansourM/photogrammetry/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/ZiadMansourM/photogrammetry/actions/workflows/docs.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![python 3.10.4](https://img.shields.io/badge/python-v3.10.4-<COLOR>.svg)](https://shields.io/)

</div>

## 📝 Pipeline - [docs](https://docs.scanmate.sreboy.com/) - [videos](https://www.youtube.com/playlist?list=PLtRAgw3FCYeBXUeBIDOmbzzEEryIvtJo3)
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
- Loop without repetition using Itertools
$ compute_k_matrix
$ generate_point_cloud
- Recover Pose of reference camera
- Recover rest camera poses using solvePNPRansac
- Apply Triangulation
$ 3D reconstruction
- Use PointsCloud to generate a 3D Object (.stl) file
```

## 🏛️ Datasets
- [ ] snow-man.
- [X] hammer.
- [X] cottage.
- [X] fountain.

## 🧐 Production Structure
```console
(venv) ziadh@Ziads-MacBook-Air production % tree 
.
├── conf
│   ├── certs
│   ├── html
│   ├── kong-config
│   │   └── kong.yaml
│   ├── logs
│   └── nginx.conf
├── data
├── docker-compose.yml
└── src
    ├── Dockerfile
    ├── main.py
    ├── scanmate.py
    └── under_the_hood
        ├── __init__.py
        ├── compute_sift_features.py
        ├── data_feature_match.py
        ├── data_structures
        │   ├── __init__.py
        │   ├── feature_matches.py
        │   ├── image.py
        │   └── images.py
        ├── generate_points_cloud.py
        ├── image_match.py
        ├── prepare_images.py
        └── utils
            ├── __init__.py
            └── utils.py

10 directories, 18 files
```

⚖️ License
-------

This project is licensed under the terms of the GNU General Public License version 3.0 (GPLv3). See the LICENSE file for details.
