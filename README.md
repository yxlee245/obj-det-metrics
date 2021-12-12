# About this Repo

This repo contains simple code to compute object detection metrics. The code in this repo can be packaged and published to PyPI via Poetry.

# Getting Started

## For package users

Install the package with one of the following commands:

```bash
# Poetry users
poetry add obj-det-metrics

# Pip users
pip install obj-det-metrics
```

Refer to [compute_ap_map.py](./bin/compute_ap_map.py) for an example of using the package.

## For contributors

1. Install Poetry (refer to the [documentation](https://python-poetry.org/docs/) for installation steps)
2. Fork this repo and do git clone.
3. Run `poetry install` to install dependency packages.

# Milestones

Date | Changes
--- | ---
12 Dec 2021 | Adapted and refactored codes to compute APs and mAP

# Acknowledgement
- The codes for computing APs and mAP were adapted from [`mAP` repo by Cartucho](https://github.com/Cartucho/mAP) and [`mapcalc` repo by LeMuecke](https://github.com/LeMuecke/mapcalc).
