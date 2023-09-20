---
sidebar_position: 1
id: Introduction
description: 🚁 Helicopter view of Our Graduation Project.
slug: /
---

export const CenterContainer = ({ children }) => (
  <div style={{ textAlign: 'center' }}>
    {children}
  </div>
);

export const Table = ({ headers, rows }) => (
  <div style={{ display: 'inline-block' }}>
    <table>
      <tbody>
        <tr>
          <td colSpan={3} style={{ textAlign: 'center' }}>Developed by four talented young men as their graduation project.</td>
        </tr>
        <tr>
          {headers.map(header => <th key={header}>{header}</th>)}
        </tr>
        {rows.map((row, rowIndex) => (
          <tr key={rowIndex}>
            {Array.isArray(row) ? row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>) : null}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

<CenterContainer>
  <img src="/img/ScanMate.png" alt="ScanMate" width="400" />
  <Table
    headers={['Name', 'Github', 'Twitter']}
    rows={[
      ['Ziad Mansour', <a href="https://github.com/ZiadMansourM">ZiadMansourM</a>, <a href="https://twitter.com/Ziad_M_404">@ziad_m_404</a>],
      ['Mohamed Wael', '-', '-'],
      ['Maged Elosail', '-', '-'],
      ['Yousif adel', '-', '-'],
    ]}
  />
  <p>ScanMate: Implementation of Close Range Photogrammetry using Classical Image Processing Techniques</p>
</CenterContainer>

## 📝 Pipeline - [docs][docs] - [videos][videos]
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

| Status | Dataset Link | Description |
|--------|--------------|-------------|
| ❌  | [Gingerbread Man](https://www.capturingreality.com/free-datasets) | 3D model of a gingerbread cookie. Created in RealityCapture from 158 images. |
| ✅  | [Hammer](https://www.capturingreality.com/free-datasets) | Hammer dataset with size of 750 MB. |
| ✅  | [Small Cottage](https://www.capturingreality.com/free-datasets) | Objects Scanned from all sides using Masks. |
| ✅  | [Fountain](https://sketchfab.com/3d-models/fountain-dataset-bdcf73513f404370a80cd3d8d0871fa8) | 3D reconstruction images from the popular Strecha dataset. |


## 🧐 Production Folder Structure
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

## ⚖️ License

This project is licensed under the terms of the GNU General Public License version 3.0 (GPLv3). See the LICENSE file for details.


[docs]: https://docs.scanmate.sreboy.com/
[videos]: https://www.youtube.com/playlist?list=PLtRAgw3FCYeBXUeBIDOmbzzEEryIvtJo3