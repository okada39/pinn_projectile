# pinn_projectile

This module implements the Physics Informed Neural Network (PINN) model for a projectile motion.
For simplicity, the spacial variables are reduced to two dimensions `x(t)` and `z(t)`, and initial positions are fixed to `x(0) = z(0) = 0`. The PINN model predicts `x(t), z(t)` for `t, v0_x, v0_z`, where `v0_x, v0_z` are initial velocities at `t=0`.

## Description

The PINN is a deep learning approach to solve partial differential equations. Well-known finite difference, volume and element methods are formulated on discrete meshes to approximate derivatives. Meanwhile, the automatic differentiation using neural networks provides differential operations directly. The PINN is the automatic differentiation based solver and has an advantage of being meshless.

The effectiveness of PINNs is validated in the following works.

* [M. Raissi, et al., Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations, arXiv: 1711.10561 (2017).](https://arxiv.org/abs/1711.10561)
* [M. Raissi, et al., Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations, arXiv: 1711.10566 (2017).](https://arxiv.org/abs/1711.10566)

L-BFGS-B in scipy

Scripts and data is given as follows.

* *example : example video data.*
    * *golf*
        * `golf.mp4` : example video (free license in [pixbay](https://pixabay.com/)).
    * *yoga*
        * `yoga_1.mp4` : example video (free license in [pixbay](https://pixabay.com/)).
        * `yoga_2.mp4` : example video (free license in [pixbay](https://pixabay.com/)).
    * `list.csv` : example input video list with labels.
* *lib : libraries to implement YouTube-8M based feature extraction.*
    * `feature_extractor.py` : extracting rgb and audio features from a video.
    * `path_listing.py` : listing filepaths with data label as a dictionary.
    * `video_crawler.py` : crawl videos extracting features as .yml files.
* `main.py` : main routine to extract rgb and audio features from videos.

## Requirement

You need Python 3.6 and the following packages.

| package    | version (recommended) |
| -          | -       |
| argparse   | -       |
| numpy      | 1.16.5  |
| pandas     | 1.0.3   |
| pathlib    | -       |
| pyyaml     | 3.12    |
| tensorflow | 1.14.0  |

## Usage

example result (image)
