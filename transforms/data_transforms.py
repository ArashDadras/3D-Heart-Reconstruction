from typing import List, Tuple, Optional
import numpy as np
from scipy.ndimage import zoom


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

    def __call__(self, volume, *args, **kwargs):
        for t in self.transforms:
            volume = t(volume, *args, **kwargs)

        return volume


class ResizeAndPad(object):
    """
    First resize volume with aspect-aware mode to maintain proportions,
    then pad with zeros to reach exact target size.

    This is the best solution for neural networks that need exact dimensions
    while maintaining anatomical accuracy.

    Args:
        target_size (Tuple[int, int, int]): (depth, height, width)
        spacing (Optional[Tuple[float, float, float]]): Original voxel spacing
        resize_order (int): Interpolation order for resizing
    """

    def __init__(
        self,
        target_size: Tuple[int, int, int],
        resize_order: int = 1,
    ):
        self.target_size = target_size
        self.resize_order = resize_order

    def __call__(self, volume: np.ndarray, header: dict) -> np.ndarray:
        spacing = self._extract_spacing_from_header(header)

        # Step 1: Resize with aspect-aware mode
        if spacing is not None:
            resizer = ResizeVolumeAdvanced(
                self.target_size,
                mode="aspect_aware",
                spacing=spacing,
                order=self.resize_order,
            )
        else:
            resizer = ResizeVolumeAdvanced(
                self.target_size,
                mode="isotropic",
                order=self.resize_order,
            )

        resized_volume = resizer(volume)

        # Step 2: Pad to exact target size
        padder = PadVolume(self.target_size)
        final_volume = padder(resized_volume)

        return final_volume

    def _extract_spacing_from_header(
        self, header: dict
    ) -> Optional[Tuple[float, float, float]]:
        """Extract voxel spacing from NRRD header."""
        if "space directions" in header:
            space_dirs = header["space directions"]
            # Extract diagonal values (voxel spacing)
            spacing = [abs(space_dirs[i][i]) for i in range(len(space_dirs))]
            return tuple(spacing)
        return None


class ResizeVolumeAdvanced(object):
    """
    Advanced resize class with different strategies for handling 3D volumes.

    Args:
        target_size (Tuple[int, int, int]): (depth, height, width)
        mode (str): Resize mode - 'anisotropic', 'isotropic', or 'aspect_aware'
        spacing (Optional[Tuple[float, float, float]]): Original voxel spacing
        order (int): Interpolation order (0=nearest, 1=linear, 3=cubic)
    """

    def __init__(
        self,
        target_size: Tuple[int, int, int],
        mode: str = "anisotropic",
        spacing: Optional[Tuple[float, float, float]] = None,
        order: int = 1,
    ):
        self.target_size = target_size
        self.mode = mode
        self.spacing = spacing
        self.order = order

    def __call__(self, volume: np.ndarray, *args, **kwargs) -> np.ndarray:
        current_shape = volume.shape

        if self.mode == "anisotropic":
            # Stretch/compress each dimension independently
            zoom_factors = [
                self.target_size[i] / current_shape[i]
                for i in range(len(current_shape))
            ]

        elif self.mode == "isotropic":
            # Use same zoom factor for all dimensions (maintains aspect ratio)
            zoom_factor = min(
                self.target_size[i] / current_shape[i]
                for i in range(len(current_shape))
            )
            zoom_factors = [zoom_factor] * len(current_shape)

        elif self.mode == "aspect_aware":
            # Consider original voxel spacing to maintain proper proportions
            if self.spacing is None:
                raise ValueError(
                    "spacing must be provided for aspect_aware mode"
                )

            # Calculate physical size of original volume
            physical_size = [
                current_shape[i] * self.spacing[i]
                for i in range(len(current_shape))
            ]

            # Calculate target spacing to fit within target_size
            target_spacing = [
                physical_size[i] / self.target_size[i]
                for i in range(len(current_shape))
            ]

            # Use the largest spacing to maintain aspect ratio
            uniform_spacing = max(target_spacing)

            # Calculate actual target size that maintains aspect ratio
            actual_target_size = [
                int(physical_size[i] / uniform_spacing)
                for i in range(len(current_shape))
            ]

            zoom_factors = [
                actual_target_size[i] / current_shape[i]
                for i in range(len(current_shape))
            ]

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Apply zoom
        resized_volume = zoom(volume, zoom_factors, order=self.order)

        # Clamp values to [0,1] to fix numerical precision errors
        resized_volume = np.clip(resized_volume, 0.0, 1.0)

        return resized_volume


class ResizeVolume(object):
    """
    Resize a 3D volume to a target size using interpolation.
    This handles both upscaling and downscaling properly.

    Args:
        target_size (Tuple[int, int, int]): (depth, height, width)
        order (int): Interpolation order (0=nearest, 1=linear, 3=cubic)
    Returns:
        np.ndarray: Resized volume
    """

    def __init__(
        self,
        target_size: Tuple[int, int, int],
        order: int = 1,  # Linear interpolation by default
    ):
        self.target_size = target_size
        self.order = order

    def __call__(self, volume: np.ndarray, *args, **kwargs) -> np.ndarray:
        current_shape = volume.shape

        # Calculate zoom factors for each dimension
        zoom_factors = [
            self.target_size[i] / current_shape[i]
            for i in range(len(current_shape))
        ]

        # Use scipy's zoom function for proper interpolation
        resized_volume = zoom(volume, zoom_factors, order=self.order)

        return resized_volume


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

    def __call__(self, volume: np.ndarray, *args, **kwargs) -> np.ndarray:
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

    def __call__(self, volume: np.ndarray, *args, **kwargs) -> np.ndarray:
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

    def __call__(self, volume: np.ndarray, *args, **kwargs) -> np.ndarray:
        expanded_volume = np.expand_dims(volume, axis=0)
        expanded_volume = np.repeat(
            expanded_volume, self.number_of_channels, axis=self.axis
        )
        return expanded_volume
