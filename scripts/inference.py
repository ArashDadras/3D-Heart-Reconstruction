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


def load_model_from_checkpoint(checkpoint_path: str, config: Config):
    """
    Load trained models from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        config: Configuration object

    Returns:
        tuple: (encoder, decoder, merger, refiner) loaded models
    """
    logging.info(f"Loading models from checkpoint: {checkpoint_path}")

    # Initialize models
    encoder = Encoder()
    decoder = Decoder()
    merger = Merger()
    refiner = Refiner()

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=config.misc.device,
            weights_only=False,  # Fix for PyTorch 2.6+ compatibility
        )

        # Load model states
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        merger.load_state_dict(checkpoint["merger_state_dict"])
        refiner.load_state_dict(checkpoint["refiner_state_dict"])

        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        logging.warning(f"Checkpoint not found at {checkpoint_path}")
        logging.warning("Using randomly initialized models")

    # Move models to device
    encoder = encoder.to(config.misc.device)
    decoder = decoder.to(config.misc.device)
    merger = merger.to(config.misc.device)
    refiner = refiner.to(config.misc.device)

    # Set to evaluation mode
    encoder.eval()
    decoder.eval()
    merger.eval()
    refiner.eval()

    logging.info("Models loaded and set to evaluation mode")
    return encoder, decoder, merger, refiner


def create_test_dataset(test_data_path: str, config: Config):
    """
    Create a test dataset for inference.

    Args:
        test_data_path: Path to test data directory
        config: Configuration object

    Returns:
        DataLoader: Test data loader
    """
    logging.info(f"Creating test dataset from: {test_data_path}")

    # Input transforms for 3-channel data
    input_transform = Compose(
        [
            AddChannel(3),
            Normalize(),
        ]
    )

    # Ground truth transforms (same as training)
    gt_transform = Compose(
        [
            ResizeAndPad(
                target_size=config.dataset.ground_truth_size,
                resize_order=1,
            )
        ]
    )

    # Create test dataset
    test_dataset = HeartDataset(
        root_dir=test_data_path,
        ground_truth_size=config.dataset.ground_truth_size,
        device=config.misc.device,
        input_transform=input_transform,
        gt_transform=gt_transform,
    )

    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one sample at a time
        shuffle=False,
        num_workers=0,  # No multiprocessing for inference
        pin_memory=False,
    )

    logging.info(f"Created test dataset with {len(test_dataset)} samples")
    return test_dataloader


def run_inference_on_test_data(
    checkpoint_path: str, test_data_path: str, output_dir: str, config: Config
):
    """
    Run inference on test data and save results.

    Args:
        checkpoint_path: Path to model checkpoint
        test_data_path: Path to test data directory
        output_dir: Directory to save inference results
        config: Configuration object
    """
    print("=" * 60)
    print("RUNNING INFERENCE ON TEST DATA")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    volumes_dir = os.path.join(output_dir, "generated_volumes")
    os.makedirs(volumes_dir, exist_ok=True)

    # Load models
    encoder, decoder, merger, refiner = load_model_from_checkpoint(
        checkpoint_path, config
    )

    # Create test dataset
    test_dataloader = create_test_dataset(test_data_path, config)

    if len(test_dataloader) == 0:
        print("No test data found!")
        return

    print(f"Running inference on {len(test_dataloader)} samples...")
    print(f"Saving results to: {output_dir}")
    print()

    # Run inference
    inference_results = []

    with torch.no_grad():
        for batch_idx, (input_images, ground_truth) in enumerate(
            test_dataloader
        ):
            print(f"Processing sample {batch_idx + 1}/{len(test_dataloader)}")

            try:
                # Forward pass through models
                image_features = encoder(input_images)
                raw_features, generated_volume = decoder(image_features)
                generated_volume = merger(raw_features, generated_volume)
                refined_volume = refiner(generated_volume)

                # Convert to numpy for saving
                refined_np = refined_volume.cpu().squeeze().numpy()
                ground_truth_np = ground_truth.cpu().squeeze().numpy()

                # Save generated volume
                volume_filename = f"inference_result_{batch_idx:04d}.nrrd"
                volume_path = os.path.join(volumes_dir, volume_filename)

                # Create NRRD header
                header = {
                    "dimension": 3,
                    "type": "float32",
                    "sizes": refined_np.shape,
                    "endian": "little",
                    "encoding": "raw",
                }

                nrrd.write(volume_path, refined_np, header)

                # Save ground truth for comparison
                if ground_truth_np.size > 0:
                    gt_filename = f"ground_truth_{batch_idx:04d}.nrrd"
                    gt_path = os.path.join(volumes_dir, gt_filename)
                    nrrd.write(gt_path, ground_truth_np, header)

                # Store results
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

                print(f"  ✅ Saved: {volume_filename}")
                print(f"     Generated shape: {refined_np.shape}")
                print(f"     Ground truth shape: {ground_truth_np.shape}")

            except Exception as e:
                print(f"  ❌ Error processing sample {batch_idx}: {str(e)}")
                continue

    # Save summary
    summary_path = os.path.join(output_dir, "inference_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Inference Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total samples processed: {len(inference_results)}\n")
        f.write(f"Checkpoint used: {checkpoint_path}\n")
        f.write(f"Test data path: {test_data_path}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        f.write("Generated Files:\n")
        f.write("-" * 20 + "\n")
        for result in inference_results:
            f.write(f"Sample {result['sample_id']:04d}:\n")
            f.write(f"  Generated: {result['generated_volume_path']}\n")
            f.write(f"  Ground truth: {result['ground_truth_path']}\n")
            f.write(f"  Shape: {result['generated_shape']}\n")
            f.write("\n")

    print()
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"✅ Processed {len(inference_results)} samples")
    print(f"💾 Results saved to: {output_dir}")
    print(f"📊 Summary saved to: {summary_path}")


def main():
    """Main function for standalone inference execution."""
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml("config.yaml")
    config.misc.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the latest checkpoint
    checkpoint_path = "results/checkpoints/checkpoint_epoch_46.pth"
    test_data_path = "dataset/test"
    output_dir = "results/inference_output"

    print(f"Using device: {config.misc.device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test data: {test_data_path}")
    print(f"Output directory: {output_dir}")

    run_inference_on_test_data(
        checkpoint_path=checkpoint_path,
        test_data_path=test_data_path,
        output_dir=output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
