# Pole-Learning: Hadronic State Inference
[![PRD](https://img.shields.io/badge/Phys._Rev._D-10.1103/1jn6--l5vq-blue.svg)](https://doi.org/10.1103/1jn6-l5vq)


This repository provides the implementation for **Learning Pole Structures of Hadronic States using Predictive Uncertainty Estimation.** It utilizes an ensemble of Gradient Boosting classifier chains to identify the S-matrix pole configurations ($[bt]$, $[bb]$, and $[tb]$ Riemann sheets) from near-threshold line shapes.

## Usage & Installation

The project is structured as a Python package with supporting scripts:

**Core Library:** `./src/utils_ml` contains feature extraction and model utilities.
 
**Training:** scripts in `./scripts` handle model training and ensemble creation.
 
**Visualization:** notebooks in `./notebooks` reproduce figures from the paper, including uncertainty calibration plots.



**Quick Start:**

1. Install the package: `pip install .`
2. Run training scripts: `python scripts/train_ensemble.py`
3. Analyze results via Jupyter in the `./notebooks` directory.

## Citation

If you use this code or the associated data in your research, feel free to cite our work:

```bibtex
@article{frohnert-uncertainty-2026,
  title = {Learning pole structures of hadronic states using predictive uncertainty estimation},
  author = {Frohnert, Felix and Sombillo, Denny Lane B. and Nieuwenburg, Evert van and Emonts, Patrick},
  journal = {Phys. Rev. D},
  year = {2026},
  month = {Mar},
  publisher = {American Physical Society},
  doi = {10.1103/1jn6-l5vq},
  url = {[https://link.aps.org/doi/10.1103/1jn6-l5vq](https://link.aps.org/doi/10.1103/1jn6-l5vq)}
}
```

---

## Dataset Hierarchy (GitHub Releases)

Data artifacts provided in the [Releases](https://github.com/FelixFrohnertDB/pole-learning/releases/tag/v1.0) section are organized by source (e.g., `Data-Training` for synthetic samples or `Data-Delta1232`). Each release follows this standard hierarchy:

```text
<Dataset-Name>/
├── Extracted Data/         # ML-ready features (NumPy format)
│   ├── no_normalization/   # Features not scaled before extraction
│   └── normalization/      # Features scaled before extraction
├── Inference Data/         # Pickled data for model testing/validation
└── Raw Data/               # Original S-matrix distributions

```

### Data Components

| Component | Format | Description |
| --- | --- | --- |
| **Extracted Features** | `.npy` | 128 distinct features and their names (statistical, temporal, frequency-domain) selected via importance ranking.
| **Energy** | `.pkl` | Evaluations of synthetic/experimental line shapes at discrete energy values.
| **Intensity** | `.pkl` | The distribution $\frac{dN}{d\sqrt{s}}$ capturing resonance peaks and threshold behavior.


---

## Special Handling: Delta Datasets

For datasets with the **Delta** prefix (e.g., `Data-Delta1232`), intensity is stored as separate **real** and **imaginary** components to preserve phase information during the uniformization procedure.

### Intensity Calculation

To obtain the observable intensity (vector magnitude) for these specific datasets, you must compute the squared sum of the components:

$$Intensity = \text{real}^2 + \text{imag}^2$$

**Python Implementation:**

```python
import pickle
import numpy as np

with open('Data-Delta1232/Raw Data/P_intensity.pkl', 'rb') as f:
    data = pickle.load(f)

# Compute intensity from complex components
intensity_vector = np.square(data['real']) + np.square(data['imag'])

```

---

