# Third-party notices

| Component | Included path | Upstream / license |
|---|---|---|
| PointWorld-derived model/data utilities | `pointworld/`, `dataset_components/`, root utilities | Original Apache-2.0 notices retained |
| Point Transformer V3 | `ptv3/` | [Pointcept/PTv3](https://github.com/Pointcept/PointTransformerV3), MIT; see `ptv3/LICENSE` |
| Sonata integration lineage | `ptv3/` | [Sonata](https://github.com/facebookresearch/sonata), Apache-2.0 |
| OmniGibson transforms | `transform_utils.py` | [OmniGibson](https://github.com/StanfordVL/OmniGibson), MIT; in-file notices retained |
| deoxys_control routines | `transform_utils.py` | [deoxys_control](https://github.com/UT-Austin-RPL/deoxys_control), Apache-2.0 |

External, not vendored: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (Apache-2.0), Wan model files, DINOv3 weights and MANO assets. The video backend is pinned in `checkpoints.json`. The hand and transport modules retain their research-inspiration citations in their source docstrings. See `docs/LICENSING.md` for dataset and model-weight terms.
