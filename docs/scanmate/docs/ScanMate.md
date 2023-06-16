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
  <p>Developed by four talented young men as their graduation project.</p>
</CenterContainer>

## 📝 Pipeline
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

## ⚖️ License

This project is licensed under the terms of the GNU General Public License version 3.0 (GPLv3). See the LICENSE file for details.
