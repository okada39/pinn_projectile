import tensorflow as tf
from .layer import GradientLayer

class PINN:
    """
    Train a projectile motion in the keras network model
    using the physics informed neural network (PINN).

    Attributes:
        input: input layer with shape (t, v0_x, v0_z).
        grads: gradient layer.
        g: gravity acceleration.
    """

    def __init__(self, network, g=1):
        """
        Args:
            input: input layer with shape (t, v0_x, v0_z).
            grads: gradient layer.
            g: gravity acceleration.
        """

        self.input = network.input
        self.grads = GradientLayer(network)
        self.g = g

    def input_t0(self):
        """
        Force t=0 for the input.

        Args:
            x: input layer with shape (t, v0_x, v0_z).

        Returns:
            t=0 forced lambda layer with shape (0, v0_x, v0_z).
        """
        return tf.keras.layers.Lambda(lambda x:
            x * tf.concat([tf.zeros(1), tf.ones(x.shape[-1] - 1)], axis=0))

    def mse(self):
        """
        Compute a mean-squared-error (MSE) for (x1 - x2).

        Args:
            x: input list [x1, x2].

        Returns:
            MSE computed lambda layer
        """
        return tf.keras.layers.Lambda(lambda x:
            tf.reduce_mean(tf.square(tf.subtract(*x))))

    def build(self):
        """
        Build a training model using the physics informed neural network (PINN).

        Returns:
            PINN training model for the projectile motion
        """

        # compute d2r(t)/dt2
        _, _, d2r_dt2 = self.grads(self.input)
        # compute r(t0) and dr(t0)/dt
        r_t0, dr_dt_t0, _ = self.grads(self.input_t0()(self.input))

        # build the PINN model
        model = tf.keras.models.Model(inputs=self.input, outputs=[d2r_dt2, r_t0, dr_dt_t0])
        # add a loss related to differential equation
        model.add_loss(self.mse()([d2r_dt2, [0, -self.g]]))
        # add a loss related to initial position
        model.add_loss(self.mse()([r_t0, [0, 0]]))
        # add a loss related to initial velocity
        model.add_loss(self.mse()([dr_dt_t0, self.input[..., 1:]]))

        return model
