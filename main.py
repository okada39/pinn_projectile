import lib.tf_silent
import numpy as np
import matplotlib.pyplot as plt
from lib.pinn import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B

def theoretical_motion(input, g):
    """
    Compute the theoretical projectile motion.

    Args:
        input: ndarray with shape (num_samples, 3) for t, v0_x, v0_z
        g: gravity acceleration

    Returns:
        theoretical motion of x, z.
    """
    t, v0_x, v0_z = np.split(input, 3, axis=-1)
    x = v0_x * t
    z = v0_z * t - 0.5 * g * t * t
    return x, z

if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for a projectile motion.
    """

    # number of training samples
    num_train_samples = 10000
    # number of test samples
    num_test_samples = 100
    # gravity acceleration
    g = 1.0

    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network, g).build()

    # train the model using L-BFGS-B algorithm
    samples = np.random.rand(num_train_samples, 3)
    lbfgs = L_BFGS_B(model=pinn, samples=samples)
    lbfgs.fit()

    # Test
    t = np.linspace(0, 1, num_test_samples).reshape((num_test_samples, 1))
    v0 = 0.5 * np.ones((num_test_samples, 2))
    x = np.concatenate([t, v0], axis=-1)
    r_pred = network.predict(x, batch_size=num_test_samples)
    # plot theory vs prediction
    plt.plot(*theoretical_motion(x, g), label='theory', color='crimson')
    plt.scatter(r_pred[..., 0], r_pred[..., 1], label='pinn', s=5, color='royalblue')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.legend()
    plt.show()
