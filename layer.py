import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives.

    Attributes:
        model: keras network model.
    """
    
    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, x):
        """
        Computing 1st and 2nd derivatives.

        Args:
            x: input variable.

        Returns:
            model output, 1st and 2nd derivatives.
        """

        with tf.GradientTape() as g:
            g.watch(x)
            with tf.GradientTape() as gg:
                gg.watch(x)
                r = self.model(x)
            dr_dt = gg.batch_jacobian(r, x)[..., 0]
        d2r_dt2 = g.batch_jacobian(dr_dt, x)[..., 0]
        return r, dr_dt, d2r_dt2
