from dataclasses import dataclass
from typing import Tuple
import yaml


@dataclass
class TrainingConfig:
    num_epochs: int
    policy: str
    betas: Tuple[float, float]
    encoder_learning_rate: float
    decoder_learning_rate: float
    merger_learning_rate: float
    refiner_learning_rate: float
    encoder_lr_milestones: Tuple[int]
    decoder_lr_milestones: Tuple[int]
    merger_lr_milestones: Tuple[int]
    refiner_lr_milestones: Tuple[int]
    momentum: float
    gamma: float


@dataclass
class DatasetConfig:
    root_dir: str
    ground_truth_size: Tuple[int, int, int]
    batch_size: int
    shuffle: bool
    val_split: float


@dataclass
class MiscConfig:
    random_seed: int
    device: str


@dataclass
class ResultsConfig:
    out_path: str
    logs: str
    checkpoints: str


@dataclass
class TestingConfig:
    voxel_thresh: Tuple[float]


@dataclass
class Config:
    training: TrainingConfig
    dataset: DatasetConfig
    misc: MiscConfig
    results: ResultsConfig
    testing: TestingConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Convert list to tuple
        data["dataset"]["ground_truth_size"] = tuple(
            data["dataset"]["ground_truth_size"]
        )
        data["training"]["betas"] = tuple(data["training"]["betas"])
        data["training"]["encoder_lr_milestones"] = tuple(
            data["training"]["encoder_lr_milestones"]
        )
        data["training"]["decoder_lr_milestones"] = tuple(
            data["training"]["decoder_lr_milestones"]
        )
        data["training"]["merger_lr_milestones"] = tuple(
            data["training"]["merger_lr_milestones"]
        )
        data["training"]["refiner_lr_milestones"] = tuple(
            data["training"]["refiner_lr_milestones"]
        )

        return cls(
            training=TrainingConfig(**data["training"]),
            testing=TestingConfig(**data["testing"]),
            dataset=DatasetConfig(**data["dataset"]),
            misc=MiscConfig(**data["misc"]),
            results=ResultsConfig(**data["results"]),
        )
