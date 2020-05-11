import numpy as np

class Dataset:
    """
    Generate sample data for a projectile motion.

    Attributes:
        num_data: number of all data.
        batch_size: batch size.
        dims: number of data dimension. Default is 3 for (t, v0_x, v0_z).
    """

    def __init__(self, num_data, batch_size, dims=3):
        """
        Args:
            num_data: number of all data.
            batch_size: batch size.
            dims: number of data dimension. Default is 3 for (t, v0_x, v0_z).
        """
        self.num_data = num_data
        self.batch_size = batch_size
        self.dims = dims

    def sample(self, num_samples):
        """
        Sample data.

        Args:
            num_samples: number of samples.

        Returns:
            sample data with shape (num_samples, dims), None
        """
        return np.random.rand(num_samples, self.dims), None

    def generator(self):
        """
        Generate batch data.

        Yields:
            batch data with shape (batch_size, dims), None
        """
        while True:
            yield self.sample(self.batch_size)

    def steps_per_epoch(self):
        """
        Get the number of steps per epoch.

        Returns:
            number of steps per epoch.
        """

        return self.num_data // self.batch_size
