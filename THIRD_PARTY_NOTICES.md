# Third-Party Notices

PixelRacers bundles or depends on the following third-party software and model weights.
This file follows the common **Third-Party Notices** pattern used alongside project licenses.
If you redistribute this codebase, retain this file and comply with each component’s license.

---

## 1. Git submodule (source included)

| Component | Location | License | Upstream |
|-----------|----------|---------|----------|
| **Hierarchical Localization (hloc)** | `third_party/Hierarchical-Localization/` | [Apache License 2.0](third_party/Hierarchical-Localization/LICENSE) | [cvg/Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization) |

Installed editable via `requirements.txt` (`-e third_party/Hierarchical-Localization`).

---

## 2. Python packages (installed via pip)

| Package | Role in PixelRacers | Typical license |
|---------|---------------------|-----------------|
| **PyTorch** / **torchvision** | Deep learning backend (SuperPoint, NetVLAD, MPS/CUDA) | BSD-style ([PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE)) |
| **LightGlue** (via hloc requirements) | **SuperPoint** local feature extractor | See [cvg/LightGlue](https://github.com/cvg/LightGlue) (Apache 2.0) |
| **kornia** | Image / geometry utilities used by hloc | Apache 2.0 |
| **pycolmap** | COLMAP bindings (SfM / mapping notebooks) | BSD ([colmap/pycolmap](https://github.com/colmap/pycolmap)) |
| **OpenCV** (`opencv-python` / `opencv-python-headless`) | Image I/O, undistortion, preprocessing | Apache 2.0 |
| **NumPy**, **SciPy**, **pandas**, **matplotlib**, **plotly**, **h5py**, **tqdm**, **Pillow**, **packaging**, **gdown** | Numerics, plots, downloads | Per-package PyPI licenses |
| **mcap**, **mcap-ros2-support** | ROS 2 bag / `.mcap` playback | See [foxglove/mcap](https://github.com/foxglove/mcap) |
| **Jupyter**, **ipykernel**, **ipywidgets** | Notebooks | BSD / project-specific |
| **PyYAML** | EKF noise config (`submission-demo`) | MIT |

Full pinned/declared versions: [`requirements.txt`](requirements.txt) and [`third_party/Hierarchical-Localization/requirements.txt`](third_party/Hierarchical-Localization/requirements.txt).

---

## 3. Model weights (downloaded at runtime, not in git)

These are **not** committed in this repository (except optional Git LFS map bundles under `submission-demo/sfm-model/`). First run may download or build caches on your machine.

| Model / weights | Used by | Source / notes |
|-----------------|---------|----------------|
| **VGG16-NetVLAD-Pitts30K** | Global retrieval (`NetVLAD` in hloc) | Downloaded by hloc on first use; trained on **Pittsburgh / Pitts30K**-style place recognition data (see hloc NetVLAD docs). |
| **SuperPoint** | Local features (`lightglue.SuperPoint`) | Weights pulled via LightGlue / torch hub on first use; original method: DeTone et al., SuperPoint. |
| **Prebuilt map bundles** (`map_bundle.npz`, `netvlad_db.npz`, `points3D.ply`) | `submission-demo` localization | Shipped in-repo via **Git LFS** (built with `sp_nv_nn` pipeline); descriptors must match query backend. |
| **HF-Net ONNX** (optional) | Future / research notebooks | **Not** included. See [`3d-map-test/models/README_HFNET.md`](3d-map-test/models/README_HFNET.md) and [`submission-demo/sfm-model/models/README_HFNET.md`](submission-demo/sfm-model/models/README_HFNET.md). Paper: Sarlin et al., HF-Net (CVPR 2019). |

---

## 4. Data and challenge context

| Item | Notes |
|------|--------|
| **Constructor Labs** camera localization bags (`.mcap`) | Provided separately for the challenge; not part of this repo. |
| **Track / map mcap** (`track-model/map.mcap`) | Racing-line geometry for plots and (s,n) frame; bundled under submission-demo (Git LFS). |

---

## 5. Disclaimer

Component names and trademarks belong to their respective owners. This project is research / submission tooling; **verify license terms** before commercial use or redistribution of weights and datasets.

For questions about hloc or LightGlue, refer to upstream repositories and citation requirements in their READMEs.
