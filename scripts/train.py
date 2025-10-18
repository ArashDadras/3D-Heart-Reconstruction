import torch
import logging
import os
import numpy as np
from config import Config
from models.encoder import Encoder
from models.decoder import Decoder
from models.merger import Merger
from models.refiner import Refiner
from data.dataloader import create_dataloaders
from utils.average_meter import AverageMeter
from scripts.inference import run_inference_on_data_loader
from pathlib import Path

logger = logging.getLogger(__name__)


def init_weights(m):
    if (
        isinstance(m, torch.nn.Conv2d)
        or isinstance(m, torch.nn.Conv3d)
        or isinstance(m, torch.nn.ConvTranspose2d)
        or isinstance(m, torch.nn.ConvTranspose3d)
    ):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(
        m, torch.nn.BatchNorm3d
    ):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoint(
    config: Config,
    encoder: Encoder,
    decoder: Decoder,
    merger: Merger,
    refiner: Refiner,
    epoch: int,
    loss: float,
    iou: float = None,
    checkpoint_type: str = "regular",
) -> str:
    """
    Save model checkpoint.
    Args:
        config: Config object
        encoder: Encoder model
        decoder: Decoder model
        merger: Merger model
        refiner: Refiner model
        epoch: Current epoch
        loss: Current loss
        iou: Current IoU
        checkpoint_type: Type of checkpoint ("regular", "best_loss",
                                           "best_iou", "best_iou_loss")
    Returns:
        checkpoint_path: Path to saved checkpoint
    """
    os.makedirs(config.results.checkpoints, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "merger_state_dict": merger.state_dict(),
        "refiner_state_dict": refiner.state_dict(),
        "loss": loss,
        "iou": iou,
    }

    if checkpoint_type == "best_loss":
        checkpoint_path = os.path.join(
            config.results.checkpoints, f"best_loss_epoch_{epoch}.pth"
        )
    elif checkpoint_type == "best_iou":
        checkpoint_path = os.path.join(
            config.results.checkpoints, f"best_iou_epoch_{epoch}.pth"
        )
    elif checkpoint_type == "best_iou_loss":
        checkpoint_path = os.path.join(
            config.results.checkpoints, f"best_iou_loss_epoch_{epoch}.pth"
        )
    else:
        checkpoint_path = os.path.join(
            config.results.checkpoints, f"checkpoint_epoch_{epoch}.pth"
        )

    torch.save(checkpoint, checkpoint_path)

    if iou is not None:
        logging.info(
            f"Checkpoint saved to {checkpoint_path}\
                 (Loss: {loss:.4f}, IoU: {iou:.4f})"
        )
    else:
        logging.info(
            f"Checkpoint saved to {checkpoint_path}\
                 (Loss: {loss:.4f}, IoU: N/A)"
        )

    return checkpoint_path


def train_network(config: Config):
    checkpoints_paths = []
    train_set_dataloader, val_set_dataloader = create_dataloaders(config)

    logging.info("train dataset size: %s", len(train_set_dataloader.dataset))
    logging.info("val dataset size: %s", len(val_set_dataloader.dataset))

    encoder = Encoder().to(config.misc.device)
    decoder = Decoder().to(config.misc.device)
    merger = Merger().to(config.misc.device)
    refiner = Refiner().to(config.misc.device)

    if config.training.policy == "adam":
        encoder_solver = torch.optim.Adam(
            filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=config.training.encoder_learning_rate,
            betas=config.training.betas,
        )
        decoder_solver = torch.optim.Adam(
            decoder.parameters(),
            lr=config.training.decoder_learning_rate,
            betas=config.training.betas,
        )
        refiner_solver = torch.optim.Adam(
            refiner.parameters(),
            lr=config.training.refiner_learning_rate,
            betas=config.training.betas,
        )
        merger_solver = torch.optim.Adam(
            merger.parameters(),
            lr=config.training.merger_learning_rate,
            betas=config.training.betas,
        )
    elif config.training.policy == "sgd":
        encoder_solver = torch.optim.SGD(
            filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=config.training.encoder_learning_rate,
            momentum=config.training.momentum,
        )
        decoder_solver = torch.optim.SGD(
            decoder.parameters(),
            lr=config.training.decoder_learning_rate,
            momentum=config.training.momentum,
        )
        refiner_solver = torch.optim.SGD(
            refiner.parameters(),
            lr=config.training.refiner_learning_rate,
            momentum=config.training.momentum,
        )
        merger_solver = torch.optim.SGD(
            merger.parameters(),
            lr=config.training.merger_learning_rate,
            momentum=config.training.momentum,
        )
    else:
        raise Exception(
            "[FATAL] Unknown optimizer %s." % (config.training.policy)
        )

    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        encoder_solver,
        milestones=config.training.encoder_lr_milestones,
        gamma=config.training.gamma,
    )
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        decoder_solver,
        milestones=config.training.decoder_lr_milestones,
        gamma=config.training.gamma,
    )
    refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        refiner_solver,
        milestones=config.training.refiner_lr_milestones,
        gamma=config.training.gamma,
    )
    merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        merger_solver,
        milestones=config.training.merger_lr_milestones,
        gamma=config.training.gamma,
    )

    bce_loss = torch.nn.BCELoss()

    best_loss = float("inf")
    best_iou = 0.0
    best_loss_epoch = 0
    best_iou_epoch = 0

    for epoch_idx in range(config.training.num_epochs):
        encoder_losses = AverageMeter()
        refiner_losses = AverageMeter()

        encoder.train()
        decoder.train()
        merger.train()
        refiner.train()

        for input_images, ground_truth_volumes in train_set_dataloader:
            image_features = encoder(input_images)
            raw_features, generated_volumes = decoder(image_features)
            generated_volumes = merger(raw_features, generated_volumes)

            encoder_loss = (
                bce_loss(generated_volumes, ground_truth_volumes) * 10
            )

            generated_volumes = refiner(generated_volumes)
            refiner_loss = (
                bce_loss(generated_volumes, ground_truth_volumes) * 10
            )

            encoder.zero_grad()
            decoder.zero_grad()
            refiner.zero_grad()
            merger.zero_grad()

            encoder_loss.backward(retain_graph=True)
            refiner_loss.backward()

            encoder_solver.step()
            decoder_solver.step()
            refiner_solver.step()
            merger_solver.step()

            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        refiner_lr_scheduler.step()
        merger_lr_scheduler.step()

        avg_encoder_loss = encoder_losses.avg
        avg_refiner_loss = refiner_losses.avg
        total_loss = avg_encoder_loss + avg_refiner_loss

        val_loss, max_iou = validate_network(
            config, val_set_dataloader, encoder, decoder, merger, refiner
        )

        logging.info(
            f"[Epoch {epoch_idx + 1}/{config.training.num_epochs}] "
            f"Train -> EncoderLoss: {avg_encoder_loss:.4f},\
                 RefinerLoss: {avg_refiner_loss:.4f},\
                 Total: {total_loss:.4f}"
        )
        logging.info(
            f"[Epoch {epoch_idx + 1}/{config.training.num_epochs}] "
            f"Validation -> Loss: {val_loss:.4f},\
                 IoU: {max_iou:.4f}"
        )

        is_best_loss = total_loss < best_loss
        is_best_iou = max_iou > best_iou

        if is_best_loss:
            best_loss = total_loss
            best_loss_epoch = epoch_idx + 1

        if is_best_iou:
            best_iou = max_iou
            best_iou_epoch = epoch_idx + 1

        checkpoint_type = ""
        if is_best_loss and is_best_iou:
            checkpoint_type = "best_iou_loss"
        elif is_best_loss:
            checkpoint_type = "best_loss"
        elif is_best_iou:
            checkpoint_type = "best_iou"

        checkpoint_path = save_checkpoint(
            config,
            encoder,
            decoder,
            merger,
            refiner,
            epoch_idx + 1,
            total_loss,
            max_iou,
            checkpoint_type=checkpoint_type,
        )
        checkpoints_paths.append(checkpoint_path)

    # ---- final summary ----
    logging.info("===== Training Summary =====")
    logging.info(f"Best Loss: {best_loss:.4f} (Epoch {best_loss_epoch})")
    logging.info(f"Best IoU : {best_iou:.4f} (Epoch {best_iou_epoch})")

    # Save the inference on validation set and train set
    # with the best checkpoint
    print(checkpoints_paths)

    best_iou_loss_checkpoint_path = find_best_checkpoint(
        "best_iou_loss", checkpoints_paths
    )
    best_loss_checkpoint_path = find_best_checkpoint(
        "best_loss", checkpoints_paths
    )
    best_iou_checkpoint_path = find_best_checkpoint(
        "best_iou", checkpoints_paths
    )

    if best_iou_loss_checkpoint_path is not None:
        logging.info(
            f"Running inference on validation set with \
                best iou loss checkpoint: {best_iou_loss_checkpoint_path}"
        )
        run_inference_on_data_loader(
            best_iou_loss_checkpoint_path,
            val_set_dataloader,
            "results/inference_output",
            config,
        )

    if (
        best_loss_checkpoint_path is not None
        and best_iou_loss_checkpoint_path != best_loss_checkpoint_path
    ):
        logging.info(
            f"Running inference on validation set with \
                best loss checkpoint: {best_loss_checkpoint_path}"
        )
        run_inference_on_data_loader(
            best_loss_checkpoint_path,
            val_set_dataloader,
            "results/inference_output",
            config,
        )

    if (
        best_iou_checkpoint_path is not None
        and best_iou_loss_checkpoint_path != best_iou_checkpoint_path
    ):
        logging.info(
            f"Running inference on validation set with \
                best iou checkpoint: {best_iou_checkpoint_path}"
        )
        run_inference_on_data_loader(
            best_iou_checkpoint_path,
            val_set_dataloader,
            "results/inference_output",
            config,
        )


def validate_network(
    config: Config,
    val_set_dataloader: torch.utils.data.DataLoader,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    merger: torch.nn.Module,
    refiner: torch.nn.Module,
):
    """
    Validate the network and return mean IoU and mean loss.
    Args:
        config: Config object
        val_set_dataloader: Validation set dataloader
        encoder: Encoder model
        decoder: Decoder model
        merger: Merger model
        refiner: Refiner model

    Returns:
        mean_loss: Mean loss
        max_iou: Maximum IoU
    """
    bce_loss = torch.nn.BCELoss()

    encoder_losses = AverageMeter()
    refiner_losses = AverageMeter()
    all_ious = []

    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    for input_images, ground_truth_volume in val_set_dataloader:
        with torch.no_grad():
            image_features = encoder(input_images)
            raw_features, generated_volume = decoder(image_features)
            generated_volume = merger(raw_features, generated_volume)

            encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10
            generated_volume = refiner(generated_volume)
            refiner_loss = bce_loss(generated_volume, ground_truth_volume) * 10

            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            sample_iou = []
            for th in config.testing.voxel_thresh:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(
                    _volume.mul(ground_truth_volume)
                ).float()
                union = torch.sum(
                    torch.ge(_volume.add(ground_truth_volume), 1)
                ).float()
                sample_iou.append((intersection / union).item())

            all_ious.append(sample_iou)

    mean_iou = np.mean(all_ious, axis=0)
    max_iou = np.max(mean_iou)
    avg_loss = encoder_losses.avg + refiner_losses.avg

    return avg_loss, max_iou


def find_best_checkpoint(
    type: str, checkpoints_paths: list[str]
) -> str | None:
    """
    Find the best checkpoint based on the type.
    Args:
        type: Type of checkpoint ("best_loss", "best_iou", "best_iou_loss")
        checkpoints_paths: List of checkpoint paths
    Returns:
        best_checkpoint_path: Path to best checkpoint
    """

    if type not in ["best_loss", "best_iou", "best_iou_loss"]:
        raise ValueError(f"Invalid checkpoint type: {type}")

    selected_checkpoints = []
    for checkpoint in checkpoints_paths:
        filename = Path(checkpoint).name
        if filename.startswith(type):
            selected_checkpoints.append(str(Path(checkpoint)))

    if not selected_checkpoints:
        return None

    sorted_checkpoints = sorted(
        selected_checkpoints, key=lambda p: Path(p).name
    )
    return sorted_checkpoints[-1]
