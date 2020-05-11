import tensorflow as tf

class Network:
    """
    Build a keras network model for the physics informed neural network (PINN).
    """

    @classmethod
    def build(cls, num_inputs=3, layers=[32, 16, 16, 32], activation='softplus', num_outputs=2):
        """
        Build a keras network model with input shape (t, v0_x, v0_z) and output shape (x, z).

        Args:
            num_inputs: number of input variables. Default is 3 for (t, v0_x, v0_z).
            layers: number of hidden layers.
            activation: activation function in hidden layers.
            num_outpus: number of output variables. Default is 2 for (x, z).

        Returns:
            keras network model
        """

        # input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        # hidden layers
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation)(x)
        # output layer
        outputs = tf.keras.layers.Dense(num_outputs)(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
