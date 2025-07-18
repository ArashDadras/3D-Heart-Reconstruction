import logging
import torch
from config import Config
from scripts.train import train_network


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml("config.yaml")

    config.misc.device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_network(config)
