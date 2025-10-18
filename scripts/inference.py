#!/usr/bin/env python3
"""
3D Heart Reconstruction Inference Script

This script loads trained models from checkpoints and runs inference on test
data. It saves the generated volumes and provides detailed reports.
"""

import os
import torch
import nrrd
import logging

from config import Config
from data.heart_dataset import HeartDataset
from models.encoder import Encoder
from models.decoder import Decoder
from models.merger import Merger
from models.refiner import Refiner
from transforms.data_transforms import (
    Compose,
    ResizeAndPad,
    AddChannel,
    Normalize,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str, config: Config
) -> tuple[Encoder, Decoder, Merger, Refiner]:
    """
    Load trained models from checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        config (Config): Configuration object

    Returns:
        tuple[Encoder, Decoder, Merger, Refiner]: Loaded models
    """
    logging.info(f"Loading models from checkpoint: {checkpoint_path}")

    encoder = Encoder()
    decoder = Decoder()
    merger = Merger()
    refiner = Refiner()

    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=config.misc.device,
            weights_only=False,  # Fix for PyTorch 2.6+ compatibility
        )

        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        merger.load_state_dict(checkpoint["merger_state_dict"])
        refiner.load_state_dict(checkpoint["refiner_state_dict"])

        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        logging.warning(f"Checkpoint not found at {checkpoint_path}")
        logging.warning("Using randomly initialized models")

    encoder = encoder.to(config.misc.device)
    decoder = decoder.to(config.misc.device)
    merger = merger.to(config.misc.device)
    refiner = refiner.to(config.misc.device)

    encoder.eval()
    decoder.eval()
    merger.eval()
    refiner.eval()

    logging.info("Models loaded and set to evaluation mode")
    return encoder, decoder, merger, refiner


def create_test_dataset(test_data_path: str, config: Config) -> DataLoader:
    """
    Create test dataset for inference.

    Args:
        test_data_path (str): Path to test data directory
        config (Config): Configuration object

    Returns:
        DataLoader: Data loader for test dataset
    """
    logging.info(f"Creating test dataset from: {test_data_path}")

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

    test_dataset = HeartDataset(
        root_dir=test_data_path,
        ground_truth_size=config.dataset.ground_truth_size,
        device=config.misc.device,
        input_transform=input_transform,
        gt_transform=gt_transform,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    logging.info(f"Created test dataset with {len(test_dataset)} samples")
    return test_dataloader


def create_and_run_inference_on_test_data(
    checkpoint_path: str,
    test_data_path: str,
    output_dir: str,
    config: Config,
):
    """
    Create test dataset and run inference on it.

    Args:
        checkpoint_path: Path to model checkpoint
        test_data_path: Path to test data directory
        output_dir: Directory to save inference results
        config: Configuration object
    """

    test_dataloader = create_test_dataset(test_data_path, config)

    run_inference_on_data_loader(
        checkpoint_path=checkpoint_path,
        data_loader=test_dataloader,
        output_dir=output_dir,
        config=config,
    )


def run_inference_on_data_loader(
    checkpoint_path: str,
    data_loader: DataLoader,
    output_dir: str,
    config: Config,
):
    """
    Run inference on a given data loader.

    Args:
        checkpoint_path: Path to model checkpoint
        data_loader: Data loader
        output_dir: Directory to save inference results
        config: Configuration object
    """
    os.makedirs(output_dir, exist_ok=True)
    volumes_dir = os.path.join(output_dir, "generated_volumes")
    os.makedirs(volumes_dir, exist_ok=True)

    encoder, decoder, merger, refiner = load_model_from_checkpoint(
        checkpoint_path, config
    )

    if len(data_loader) == 0:
        logger.error("Data loader is empty!")
        return

    logger.info(f"Running inference on {len(data_loader)} samples...")
    logger.info(f"Saving results to: {output_dir}")
    logger.info(f"Using checkpoint: {checkpoint_path}")

    inference_results = []

    with torch.no_grad():
        for batch_idx, (input_images, ground_truth) in enumerate(data_loader):
            logger.info(
                f"Processing sample {batch_idx + 1}/{len(data_loader)}"
            )

            try:
                image_features = encoder(input_images)
                raw_features, generated_volume = decoder(image_features)
                generated_volume = merger(raw_features, generated_volume)
                refined_volume = refiner(generated_volume)

                refined_np = refined_volume.cpu().squeeze().numpy()
                ground_truth_np = ground_truth.cpu().squeeze().numpy()

                volume_filename = f"inference_result_{batch_idx:04d}.nrrd"
                volume_path = os.path.join(volumes_dir, volume_filename)

                header = {
                    "dimension": 3,
                    "type": "float32",
                    "sizes": refined_np.shape,
                    "endian": "little",
                    "encoding": "raw",
                }

                nrrd.write(volume_path, refined_np, header)

                if ground_truth_np.size > 0:
                    gt_filename = f"ground_truth_{batch_idx:04d}.nrrd"
                    gt_path = os.path.join(volumes_dir, gt_filename)
                    nrrd.write(gt_path, ground_truth_np, header)

                inference_results.append(
                    {
                        "sample_id": batch_idx,
                        "generated_volume_path": volume_path,
                        "ground_truth_path": (
                            gt_path if ground_truth_np.size > 0 else None
                        ),
                        "generated_shape": refined_np.shape,
                        "ground_truth_shape": ground_truth_np.shape,
                    }
                )

                logger.info(f"Saved: {volume_filename}")
                logger.info(f"Generated shape: {refined_np.shape}")
                logger.info(f"Ground truth shape: {ground_truth_np.shape}")

            except Exception as e:
                logger.error(f"Error processing sample {batch_idx}: {str(e)}")
                continue

    summary_path = os.path.join(output_dir, "inference_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Inference Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total samples processed: {len(inference_results)}\n")
        f.write(f"Checkpoint used: {checkpoint_path}\n")
        f.write(f"Data loader used: {data_loader}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        f.write("Generated Files:\n")
        f.write("-" * 20 + "\n")
        for result in inference_results:
            f.write(f"Sample {result['sample_id']:04d}:\n")
            f.write(f"  Generated: {result['generated_volume_path']}\n")
            f.write(f"  Ground truth: {result['ground_truth_path']}\n")
            f.write(f"  Shape: {result['generated_shape']}\n")
            f.write("\n")

    logger.info("Inference completed successfully")
    logger.info(f"Processed {len(inference_results)} samples")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary saved to: {summary_path}")
