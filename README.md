# pinn_projectile

This module implements the Physics Informed Neural Network (PINN) model for a projectile motion. The differential equation is given by `(d^2 r)/(dt^2) = -gz`, where `r = (x, z)` is reduced to two dimensions for simplicity, and $g$ is the gravity acceleration. The initial positions are fixed to `x(0) = z(0) = 0`. The PINN model predicts `x(t), z(t)` for `t, v0_x, v0_z`, where `v0_x, v0_z` are initial velocities at `t=0`.

## Description

The PINN is a deep learning approach to solve partial differential equations. Well-known finite difference, volume and element methods are formulated on discrete meshes to approximate derivatives. Meanwhile, the automatic differentiation using neural networks provides differential operations directly. The PINN is the automatic differentiation based solver and has an advantage of being meshless.

The effectiveness of PINNs is validated in the following works.

* [M. Raissi, et al., Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations, arXiv: 1711.10561 (2017).](https://arxiv.org/abs/1711.10561)
* [M. Raissi, et al., Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations, arXiv: 1711.10566 (2017).](https://arxiv.org/abs/1711.10566)

In addition, an effective convergent optimizer is required to solve the differential equations accurately using PINNs. The stochastic gradient dicent is generally used in deep learnigs, but it only depends on the primary gradient (Jacobian). In contrast, the quasi-Newton based approach such as the limited-memory Broyden-Fletcher-Goldfarb-Shanno method for bound constraints (L-BFGS-B) incorporates the quadratic gradient (Hessian), and gives a more accurate convergence.

We implement a PINN model with the L-BFGS-B optimization for a projectile motion as the simple example.  
Scripts is given as follows.

* *lib : libraries to implement the PINN model for a projectile motion.*
    * `layer.py` : computing 1st and 2nd derivatives as a custom layer.
    * `network.py` : building a keras network model.
    * `optimizer.py` : implementing the L-BFGS-B optimization.
    * `pinn.py` : training the projectile motion in the network model.
    * `tf_silent.py` : suppressing tensorflow warnings
* `main.py` : main routine to run and test the PINN solver.

## Requirement

You need Python 3.6 and the following packages.

| package    | version (recommended) |
| -          | -      |
| matplotlib | 3.2.1  |
| numpy      | 1.18.1 |
| scipy      | 1.3.1  |
| tensorflow | 2.1.0  |

GPU acceleration is recommended in the following environments.

| package        | version (recommended) |
| -              | -     |
| cuda           | 10.1  |
| cudnn          | 7.6.5 |
| tensorflow-gpu | 2.1.0 |

## Usage

An example of solving the projectile motion by the PINN is demonstraned in `main.py`. The PINN is trained by the following procedure.

1. Building the keras network model
    ```Python
    from lib.network import Network
    network = Network.build().
    network.summary()
    ```
    The following table is the default layers in the network model.
    ```
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         [(None, 3)]               0
    _________________________________________________________________
    dense (Dense)                (None, 32)                128
    _________________________________________________________________
    dense_1 (Dense)              (None, 32)                1056
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 66
    =================================================================
    Total params: 1,250
    Trainable params: 1,250
    Non-trainable params: 0
    _________________________________________________________________
    ```
2. Building the PINN model.
    ```Python
    from lib.pinn import PINN
    pinn = PINN(network, g).build()
    ```
3. Optimizing the PINN model for training samples.
    ```Python
    from lib.optimizer import L_BFGS_B
    samples = np.random.rand(num_train_samples, 3)
    lbfgs = L_BFGS_B(model=pinn, samples=samples)
    lbfgs.fit()
    ```
    The progress is printed as follows. The optimization is terminated for loss ~ 3e-6. 
    ```
    Optimizer: L-BFGS-B (maxiter=3000)
    2619/3000 [=========================>....] - ETA: 19s - loss: 3.5697e-06
    ```

An example result of `main.py` is shown below.

![result_img](result_img.png)
