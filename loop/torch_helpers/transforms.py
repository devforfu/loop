"""
Custom transformations to process input data.
"""

class ExpandChannels:
    """Converts gray scale 1-channel image tensor into 3-channels tensor."""

    def __init__(self, num_of_channels=3):
        self.nc = num_of_channels

    def __call__(self, x):
        return x.expand((self.nc,) + x.shape[1:])
