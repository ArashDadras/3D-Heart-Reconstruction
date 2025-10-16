import torch
import logging

from utils.logger import setup_logging
from config import Config
from data.dataloader import create_dataset
from scripts.inference import run_inference_on_test_data
from scripts.train import train_network

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config = Config.from_yaml("config.yaml")
    config.misc.device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logging(config.results.logs)

    print("\nChoose an option:")
    print("1. Check dataset integrity")
    print("2. Run inference on test data")
    print("3. Train model")

    choice = input("Enter your choice (1/2/3): ").strip()

    logger.info("************************************")

    if choice == "1":
        logger.info("Starting dataset integrity check")
        full_heart_dataset = create_dataset(config)
        result = full_heart_dataset.do_health_checks()
        if result["corrupted"] > 0:
            logger.warning(
                "Dataset integrity check failed. Please fix corrupted data "
                "before training."
            )
            logger.warning(f"Total items checked: {result['total']}")
            logger.warning(f"Valid items: {result['valid']}")
            logger.warning(f"Corrupted items: {result['corrupted']}")
            logger.warning(
                f"Corrupted image IDs to clean: {result['corrupted_ids']}"
            )
        else:
            logger.info("Dataset integrity check passed")

    elif choice == "2":
        checkpoint_path = "results/checkpoints/checkpoint_epoch_46.pth"
        test_data_path = "dataset/test"
        output_dir = "results/inference_output"

        logger.info(f"Starting inference with checkpoint: {checkpoint_path}")
        logger.info(f"Device: {config.misc.device}")
        logger.info(f"Test data path: {test_data_path}")
        logger.info(f"Output directory: {output_dir}")

        print(f"Using checkpoint: {checkpoint_path}")
        print(f"Test data: {test_data_path}")
        print(f"Output directory: {output_dir}")

        try:
            run_inference_on_test_data(
                checkpoint_path=checkpoint_path,
                test_data_path=test_data_path,
                output_dir=output_dir,
                config=config,
            )
            logger.info("Inference completed successfully")
        except Exception as e:
            logger.error(f"Inference failed with error: {str(e)}")
            raise

    elif choice == "3":
        # Train model
        logger.info("Starting model training")
        logger.info(f"Device: {config.misc.device}")
        print(
            "Training is currently disabled. Uncomment train_network(config)"
            " to enable."
        )
        try:
            train_network(config)
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise

    else:
        logger.warning(f"Invalid choice selected: {choice}")
        print("Invalid choice. Please run again and choose 1, 2, or 3.")

    logger.info("************************************")
