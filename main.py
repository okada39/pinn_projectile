import tf_silent
import numpy as np
import matplotlib.pyplot as plt
from pinn import PINN
from dataset import Dataset
from network import Network

def train(pinn, num_data, epochs, batch_size):
    """
    Train the pinn model.

    Args:
        num_data: number of training data.
        epochs: number of training epochs.
        batch_size: batch size.
    """

    dataset = Dataset(num_data=num_data, batch_size=batch_size)
    pinn.fit_generator(dataset.generator(),
        epochs=epochs, steps_per_epoch=dataset.steps_per_epoch())

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

    # number of training data
    num_data = 10000
    # number of training epochs
    epochs = 100
    # batch size
    batch_size = 64
    # gravity acceleration
    g = 1.0

    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN trainer
    pinn = PINN(network, g).build()
    pinn.compile(optimizer='adam')
    # train the pinn model
    train(pinn, num_data=num_data, epochs=epochs, batch_size=batch_size)

    # Test
    num_test = 100
    t = np.linspace(0, 1, num_test).reshape((num_test, 1))
    v0 = 0.5 * np.ones((num_test, 2))
    x = np.concatenate([t, v0], axis=-1)
    r_pred = network.predict(x, batch_size=num_test)
    # plot theory vs prediction
    plt.plot(*theoretical_motion(x, g), label='theory')
    plt.plot(r_pred[..., 0], r_pred[..., 1], label='pinn')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.legend()
    plt.show()
