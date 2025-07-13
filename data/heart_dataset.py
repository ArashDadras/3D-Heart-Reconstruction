import nrrd
from pathlib import Path
from typing import Callable, Optional, Tuple
from torch.utils.data import Dataset
import torch
import numpy as np
from transforms.data_transforms import AddChannel, PadVolume


class HeartDataset(Dataset):
    """
    A PyTorch Dataset for loading heart reconstruction data from NRRD files.

    This dataset loads 4-channel heart data (ch2, ch3, ch4) and corresponding
    ground truth segmentation masks. It automatically validates data
    completeness and removes incomplete entries.

    Attributes:
        root_dir (str): Root directory containing the dataset
        image_paths (dict): Dictionary mapping image IDs to their file paths
        image_paths_keys (list): List of valid image IDs for indexing
        device (str): Device to load tensors on ('cuda' or 'cpu')
        target_size (tuple): Target size for ground truth volumes
                            (depth, height, width)
    """

    # Constants for path keys
    GT_PATH = "gt_path"
    CH2_PATH = "ch2_path"
    CH3_PATH = "ch3_path"
    CH4_PATH = "ch4_path"

    def __init__(
        self,
        root_dir: str,
        ground_truth_size: Tuple[int, int, int] = (128, 128, 128),
        transform: Optional[Callable] = None,
        device: str = "cpu",
    ):
        """
        Initialize the HeartDataset.

        Args:
            root_dir (str): Path to the root directory containing the dataset.
                           Should contain subdirectories with NRRD files.
            transform (Optional[Callable]): Optional transform to apply to
                the data
            target_size (Optional[Tuple[int, int, int]]): Target size for
                ground truth volumes. If None, ground truth will not be
                resized/padded.
        """
        self.root_dir = root_dir
        self.ground_truth_size = ground_truth_size
        self.image_paths = {}
        self.list_image_paths = list(Path(self.root_dir).glob("*/*.nrrd"))

        for image in self.list_image_paths:
            image_name = image.name
            image_id = f"{image_name.split('_')[0]}_{image_name.split('_')[1]}"

            if image_id not in self.image_paths:
                self.image_paths[image_id] = {}

            if "_3D_crop_gt.nrrd" in image_name:
                self.image_paths[image_id][self.GT_PATH] = image.as_posix()
            elif "_ch2.nrrd" in image_name:
                self.image_paths[image_id][self.CH2_PATH] = image.as_posix()
            elif "_ch3.nrrd" in image_name:
                self.image_paths[image_id][self.CH3_PATH] = image.as_posix()
            elif "_ch4.nrrd" in image_name:
                self.image_paths[image_id][self.CH4_PATH] = image.as_posix()

        self.check_data_completeness()

        # Keep only the first 5 items for testing
        if len(self.image_paths) > 5:
            keys_to_remove = list(self.image_paths.keys())[5:]
            for key in keys_to_remove:
                del self.image_paths[key]
            print(
                f"Keeping only first 5 samples for testing. Removed {len(keys_to_remove)} additional samples."
            )

        self.image_paths_keys = list(self.image_paths.keys())
        self.device = device
        self.transform = transform

    def check_data_completeness(self) -> dict[str, str]:
        """
        Check if each image ID has all 4 required keys with valid values.
        Removes incomplete items from self.image_paths.

        Returns:
            dict: Dictionary with image_id as key and status as value
                  - 'complete': All 4 keys present with values
                  - 'incomplete': Missing one or more keys
                  - 'empty_values': Has keys but some values are empty/None
        """
        required_keys = [
            self.GT_PATH,
            self.CH2_PATH,
            self.CH3_PATH,
            self.CH4_PATH,
        ]
        status = {}
        incomplete_ids = []

        for image_id, paths in self.image_paths.items():
            missing_keys = [key for key in required_keys if key not in paths]

            if missing_keys:
                status[image_id] = "incomplete"
                print(
                    f"WARNING: Image {image_id} missing keys: {missing_keys}"
                )
                incomplete_ids.append(image_id)
            else:
                empty_values = [
                    key
                    for key in required_keys
                    if not paths[key] or paths[key].strip() == ""
                ]

                if empty_values:
                    status[image_id] = "empty_values"
                    print(
                        f"WARNING: Image {image_id} has empty values for "
                        f"keys: {empty_values}"
                    )
                    incomplete_ids.append(image_id)
                else:
                    status[image_id] = "complete"

        for image_id in incomplete_ids:
            del self.image_paths[image_id]
            print(f"Removing image {image_id} from dataset")

        return status

    def __len__(self) -> int:
        """
        Return the number of complete image entries in the dataset.

        Returns:
            int: Number of valid image IDs with complete data.
        """
        return len(self.image_paths)

    def load_images(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return the 4-channel data and ground truth for a given index.

        Args:
            index (int): Index of the image to load.

        Returns:
            tuple: A 2-tuple containing (input_images, ground_truth_volume)
                   where each tensor is loaded from the corresponding NRRD
                   file and moved to the appropriate device.

        Raises:
            IndexError: If the index is out of bounds.
            FileNotFoundError: If any of the required NRRD files are missing.
        """
        image_id = self.image_paths_keys[index]
        ch2, _ = nrrd.read(self.image_paths[image_id][self.CH2_PATH])
        ch3, _ = nrrd.read(self.image_paths[image_id][self.CH3_PATH])
        ch4, _ = nrrd.read(self.image_paths[image_id][self.CH4_PATH])
        gt, _ = nrrd.read(self.image_paths[image_id][self.GT_PATH])

        # For time series images (ch2, ch3, ch4), take the middle frame
        ch2 = ch2[ch2.shape[0] // 2]
        ch3 = ch3[ch3.shape[0] // 2]
        ch4 = ch4[ch4.shape[0] // 2]
        ch2 = AddChannel(3)(ch2)
        ch3 = AddChannel(3)(ch3)
        ch4 = AddChannel(3)(ch4)

        ground_truth = PadVolume(self.ground_truth_size)(gt)

        if self.transform:
            ch2 = self.transform(ch2)
            ch3 = self.transform(ch3)
            ch4 = self.transform(ch4)

        # Stack the channels (3 views)
        stacked_images = np.stack([ch2, ch3, ch4], axis=0)

        return (
            torch.tensor(stacked_images, dtype=torch.float32).to(self.device),
            torch.tensor(ground_truth, dtype=torch.float32).to(self.device),
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset at the specified index.

        This method is called by PyTorch DataLoader to retrieve samples.
        It delegates to the load_images method to load the actual data.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A 2-tuple containing (input_images, ground_truth_volume)
                   for the specified index.

        Raises:
            IndexError: If the index is out of bounds.
        """
        return self.load_images(index)
