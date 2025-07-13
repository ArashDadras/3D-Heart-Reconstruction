from typing import List, Tuple
import numpy as np


class Compose(object):
    """Composes several transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.RandomBackground(),
    >>>     transforms.CenterCrop(127, 127, 3),
    >>>  ])
    """

    def __init__(self, transforms: List[object]):
        self.transforms = transforms

    def __call__(self, rendering_images):
        for t in self.transforms:
            rendering_images = t(rendering_images)

        return rendering_images


class PadVolume(object):
    """
    Pad a 3D volume to a target size with zeros.
    Args:
        target_size (Tuple[int, int, int]): (depth, height, width)
    Returns:
        np.ndarray: Padded input or ground truth
    """

    def __init__(
        self,
        target_size: Tuple[int, int, int],
    ):
        self.target_size = target_size

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        current_shape = volume.shape
        padded_volume = np.zeros(self.target_size, dtype=volume.dtype)

        # Calculate padding for each dimension
        pad_d = max(0, self.target_size[0] - current_shape[0])
        pad_h = max(0, self.target_size[1] - current_shape[1])
        pad_w = max(0, self.target_size[2] - current_shape[2])

        # Calculate start indices for placing the original volume
        start_d = pad_d // 2
        start_h = pad_h // 2
        start_w = pad_w // 2

        # Place the original volume in the center of the padded volume
        end_d = start_d + min(current_shape[0], self.target_size[0])
        end_h = start_h + min(current_shape[1], self.target_size[1])
        end_w = start_w + min(current_shape[2], self.target_size[2])

        padded_volume[start_d:end_d, start_h:end_h, start_w:end_w] = volume[
            : end_d - start_d, : end_h - start_h, : end_w - start_w
        ]

        return padded_volume


class Normalize(object):
    """
    Normalize a 3D volume to [0, 1] range.
    Args:
    Returns:
        np.ndarray: Normalized input or ground truth
    """

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        return (volume - volume.min()) / (volume.max() - volume.min())


class AddChannel(object):
    """
    Add a channel dimension to numpy array.
    Args:
        number_of_channels (int): Number of channels to add
        axis (int): Axis to add the channel dimension to
    Returns:
        np.ndarray: Volume with added channel dimension
    """

    def __init__(self, number_of_channels: int, axis: int = 0):
        self.number_of_channels = number_of_channels
        self.axis = axis

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        expanded_volume = np.expand_dims(volume, axis=0)
        expanded_volume = np.repeat(
            expanded_volume, self.number_of_channels, axis=self.axis
        )
        return expanded_volume
