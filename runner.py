import logging
import torch

from config import Config
from data.dataloader import create_train_val_datasets
from scripts.inference import run_inference_on_test_data
from scripts.train import train_network


def check_dataset_integrity(config):
    """
    Check all items in train and validation datasets for corrupted shapes.
    This helps identify volumes with zero dimensions that cause division by
    zero.
    """
    print("=" * 60)
    print("CHECKING DATASET INTEGRITY")
    print("=" * 60)

    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(config)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print()

    corrupted_items = []
    valid_items = []

    # Check training dataset
    print("Checking training dataset...")
    for i in range(len(train_dataset)):
        try:
            input_images, ground_truth = train_dataset[i]

            # Get the actual image ID from the underlying dataset
            actual_index = train_dataset.indices[i]
            image_id = train_dataset.dataset.image_paths_keys[actual_index]

            # Check shapes
            input_shape = input_images.shape
            gt_shape = ground_truth.shape

            # Check for zero dimensions
            has_zero_dims = False
            if any(dim == 0 for dim in input_shape):
                corrupted_items.append(
                    {
                        "dataset": "train",
                        "index": i,
                        "image_id": image_id,
                        "type": "input_images",
                        "shape": input_shape,
                        "issue": "zero_dimension",
                    }
                )
                print(
                    f"  CORRUPTED: Train[{i}] ({image_id}) input_images "
                    f"shape: {input_shape}"
                )
                has_zero_dims = True

            if any(dim == 0 for dim in gt_shape):
                corrupted_items.append(
                    {
                        "dataset": "train",
                        "index": i,
                        "image_id": image_id,
                        "type": "ground_truth",
                        "shape": gt_shape,
                        "issue": "zero_dimension",
                    }
                )
                print(
                    f"  CORRUPTED: Train[{i}] ({image_id}) ground_truth "
                    f"shape: {gt_shape}"
                )
                has_zero_dims = True

            # If no zero dimensions, mark as valid
            if not has_zero_dims:
                valid_items.append(
                    {
                        "dataset": "train",
                        "index": i,
                        "image_id": image_id,
                        "input_shape": input_shape,
                        "gt_shape": gt_shape,
                    }
                )

            # Print shapes for first few items
            if i < 3:
                print(
                    f"  Train[{i}] ({image_id}) - Input: {input_shape}, "
                    f"GT: {gt_shape}"
                )

        except Exception as e:
            # Try to get image ID even if loading fails
            try:
                actual_index = train_dataset.indices[i]
                image_id = train_dataset.dataset.image_paths_keys[actual_index]
            except (IndexError, KeyError, AttributeError):
                image_id = "unknown"

            corrupted_items.append(
                {
                    "dataset": "train",
                    "index": i,
                    "image_id": image_id,
                    "type": "loading_error",
                    "shape": None,
                    "issue": str(e),
                }
            )
            print(f"  ERROR: Train[{i}] ({image_id}) failed to load: {str(e)}")
            # Continue to next item instead of stopping
            continue

    print()

    # Check validation dataset
    print("Checking validation dataset...")
    for i in range(len(val_dataset)):
        try:
            input_images, ground_truth = val_dataset[i]

            # Get the actual image ID from the underlying dataset
            actual_index = val_dataset.indices[i]
            image_id = val_dataset.dataset.image_paths_keys[actual_index]

            # Check shapes
            input_shape = input_images.shape
            gt_shape = ground_truth.shape

            # Check for zero dimensions
            has_zero_dims = False
            if any(dim == 0 for dim in input_shape):
                corrupted_items.append(
                    {
                        "dataset": "val",
                        "index": i,
                        "image_id": image_id,
                        "type": "input_images",
                        "shape": input_shape,
                        "issue": "zero_dimension",
                    }
                )
                print(
                    f"  CORRUPTED: Val[{i}] ({image_id}) input_images "
                    f"shape: {input_shape}"
                )
                has_zero_dims = True

            if any(dim == 0 for dim in gt_shape):
                corrupted_items.append(
                    {
                        "dataset": "val",
                        "index": i,
                        "image_id": image_id,
                        "type": "ground_truth",
                        "shape": gt_shape,
                        "issue": "zero_dimension",
                    }
                )
                print(
                    f"  CORRUPTED: Val[{i}] ({image_id}) ground_truth "
                    f"shape: {gt_shape}"
                )
                has_zero_dims = True

            # If no zero dimensions, mark as valid
            if not has_zero_dims:
                valid_items.append(
                    {
                        "dataset": "val",
                        "index": i,
                        "image_id": image_id,
                        "input_shape": input_shape,
                        "gt_shape": gt_shape,
                    }
                )

            # Print shapes for first few items
            if i < 3:
                print(
                    f"  Val[{i}] ({image_id}) - Input: {input_shape}, "
                    f"GT: {gt_shape}"
                )

        except Exception as e:
            # Try to get image ID even if loading fails
            try:
                actual_index = val_dataset.indices[i]
                image_id = val_dataset.dataset.image_paths_keys[actual_index]
            except (IndexError, KeyError, AttributeError):
                image_id = "unknown"

            corrupted_items.append(
                {
                    "dataset": "val",
                    "index": i,
                    "image_id": image_id,
                    "type": "loading_error",
                    "shape": None,
                    "issue": str(e),
                }
            )
            print(f"  ERROR: Val[{i}] ({image_id}) failed to load: {str(e)}")
            # Continue to next item instead of stopping
            continue

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_items = len(train_dataset) + len(val_dataset)
    print(f"Total items checked: {total_items}")
    print(f"Valid items: {len(valid_items)}")
    print(f"Corrupted items: {len(corrupted_items)}")
    print()

    if corrupted_items:
        print("🔍 CORRUPTED ITEMS FOUND:")
        for item in corrupted_items:
            print(
                f"  - {item['dataset']}[{item['index']}] "
                f"({item['image_id']}) {item['type']}: {item['issue']}"
            )
            if item["shape"]:
                print(f"    Shape: {item['shape']}")

        print()
        print("📁 CORRUPTED IMAGE IDs TO CLEAN:")
        corrupted_ids = set()
        for item in corrupted_items:
            if item["image_id"] != "unknown":
                corrupted_ids.add(item["image_id"])

        for image_id in sorted(corrupted_ids):
            print(f"  - {image_id}")

        print()
        print(
            f"⚠️  Found {len(corrupted_items)} corrupted items out of "
            f"{total_items} total"
        )
        print(
            f"✅ {len(valid_items)} items are valid and can be used "
            f"for training"
        )

        return {
            "valid": len(valid_items),
            "corrupted": len(corrupted_items),
            "total": total_items,
            "corrupted_ids": sorted(corrupted_ids),
        }
    else:
        print("✅ All dataset items have valid shapes!")
        return {
            "valid": len(valid_items),
            "corrupted": 0,
            "total": total_items,
            "corrupted_ids": [],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml("config.yaml")
    config.misc.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(config.misc.device)

    # Choose what to run
    print("\nChoose an option:")
    print("1. Check dataset integrity")
    print("2. Run inference on test data")
    print("3. Train model")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        # Check dataset integrity before training
        result = check_dataset_integrity(config)
        if result["corrupted"] > 0:
            print(
                "Dataset integrity check failed. Please fix corrupted data "
                "before training."
            )
            print(f"Total items checked: {result['total']}")
            print(f"Valid items: {result['valid']}")
            print(f"Corrupted items: {result['corrupted']}")
            print(f"Corrupted image IDs to clean: {result['corrupted_ids']}")
        else:
            print("Dataset integrity check passed. Starting training...")
            # train_network(config)  # Uncomment to enable training

    elif choice == "2":
        # Run inference on test data using the existing checkpoints
        checkpoint_path = "results/checkpoints/checkpoint_epoch_46.pth"
        test_data_path = "dataset/test"
        output_dir = "results/inference_output"

        print(f"Using checkpoint: {checkpoint_path}")
        print(f"Test data: {test_data_path}")
        print(f"Output directory: {output_dir}")

        run_inference_on_test_data(
            checkpoint_path=checkpoint_path,
            test_data_path=test_data_path,
            output_dir=output_dir,
            config=config,
        )

    elif choice == "3":
        # Train model (commented out for now)
        print(
            "Training is currently disabled. Uncomment train_network(config)"
            " to enable."
        )
        train_network(config)

    else:
        print("Invalid choice. Please run again and choose 1, 2, or 3.")
