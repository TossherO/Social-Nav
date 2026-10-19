"""Display colors shared by BEV and 3D; never changes the source point cloud."""
import numpy as np


BACKGROUND = '#1c2732'
# A bright, cool height ramp leaves yellow/red/green for review decisions.
HEIGHT_COLORS = ('#68a9ed', '#69d3e5', '#b4e6ef', '#bca9ee', '#e7cdf5')
HEIGHT_RGB = np.array([[int(color[i:i + 2], 16) / 255. for i in (1, 3, 5)]
                       for color in HEIGHT_COLORS])


def height_colors(z, low, high):
    """Use a fixed source range, so equal heights keep their color across frames."""
    values = np.clip((np.asarray(z) - low) / max(high - low, 1e-6), 0., 1.)
    stops = np.linspace(0., 1., len(HEIGHT_RGB))
    return np.stack([np.interp(values, stops, channel) for channel in HEIGHT_RGB.T], axis=-1)
