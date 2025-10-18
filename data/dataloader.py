from torch.utils.data import random_split, DataLoader
from data.heart_dataset import HeartDataset
import torch
from transforms.data_transforms import (
    Compose,
    ResizeAndPad,
    AddChannel,
    Normalize,
)
from config import Config


def create_dataset(config: Config):
    """
    Create HeartDataset from dataset folder.
    Be careful when getting items since it also
    sets transforms for input and ground truth.

    Args:
        config: Configuration object with dataset parameters

    Returns:
        HeartDataset: HeartDataset object
    """

    input_transform = Compose(
        [
            AddChannel(3),
            Normalize(),
        ]
    )

    gt_transform = Compose(
        [
            ResizeAndPad(
                target_size=config.dataset.ground_truth_size,
                resize_order=1,
            )
        ]
    )

    full_dataset = HeartDataset(
        root_dir=config.dataset.root_dir,
        ground_truth_size=config.dataset.ground_truth_size,
        device=config.misc.device,
        input_transform=input_transform,
        gt_transform=gt_transform,
    )

    return full_dataset


def create_train_val_datasets(config: Config):
    """
    Create train and validation datasets with proper splitting.
    Be careful when getting items since it also
    sets transforms for input and ground truth.

    Args:
        config: Configuration object with dataset parameters

    Returns:
        tuple: (train_dataset, val_dataset)
    """

    full_dataset = create_dataset(config)

    total_size = len(full_dataset)
    val_size = int(total_size * config.dataset.val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.misc.random_seed),
    )

    return train_dataset, val_dataset


def create_dataloaders(config) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        config: Configuration object

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    train_dataset, val_dataset = create_train_val_datasets(config)

    # Use num_workers=0 when using CUDA to avoid multiprocessing issues
    num_workers = 0 if config.misc.device == "cuda" else 4

    # Disable pin_memory when using CUDA since tensors are already on GPU
    # pin_memory = False if config.misc.device == "cuda" else True
    pin_memory = False

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=config.dataset.shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, val_dataloader
