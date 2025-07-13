from torch.utils.data import random_split, DataLoader
from data.heart_dataset import HeartDataset
import torch


def create_train_val_datasets(config):
    """
    Create train and validation datasets with proper splitting.

    Args:
        config: Configuration object with dataset parameters

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Create full dataset
    full_dataset = HeartDataset(
        root_dir=config.dataset.root_dir,
        ground_truth_size=config.dataset.ground_truth_size,
        device=config.misc.device,
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * config.dataset.val_split)
    train_size = total_size - val_size

    # Split the dataset
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

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=config.dataset.shuffle,
        num_workers=4,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,  # No shuffling for validation
        num_workers=4,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader
