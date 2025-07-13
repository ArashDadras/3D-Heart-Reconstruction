import torch
import logging
import os
from config import Config
from models.encoder import Encoder
from models.decoder import Decoder
from models.merger import Merger
from models.refiner import Refiner
from data.dataloader import create_dataloaders
from utils.average_meter import AverageMeter
from time import time
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


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


def save_checkpoint(config, encoder, decoder, merger, refiner, epoch, loss, iou=None):
    """Save model checkpoint."""
    # Create checkpoint directory if it doesn't exist
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

    checkpoint_path = os.path.join(
        config.results.checkpoints, f"checkpoint_epoch_{epoch}.pth"
    )

    torch.save(checkpoint, checkpoint_path)
    
    # Fix the logging line - use proper conditional formatting
    if iou is not None:
        logging.info(f"Checkpoint saved to {checkpoint_path} (Loss: {loss:.4f}, IoU: {iou:.4f})")
    else:
        logging.info(f"Checkpoint saved to {checkpoint_path} (Loss: {loss:.4f}, IoU: N/A)")


def train_network(config: Config):
    train_set_dataloader, val_set_dataloader = create_dataloaders(config)

    logging.info("train dataset size: %s", len(train_set_dataloader.dataset))
    logging.info("val dataset size: %s", len(val_set_dataloader.dataset))

    encoder = Encoder().to(config.misc.device)
    decoder = Decoder().to(config.misc.device)
    merger = Merger().to(config.misc.device)
    refiner = Refiner().to(config.misc.device)

    logging.info("encoder parameters: %s", count_parameters(encoder))
    logging.info("decoder parameters: %s", count_parameters(decoder))
    logging.info("merger parameters: %s", count_parameters(merger))
    logging.info("refiner parameters: %s", count_parameters(refiner))

    # Set up solver
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

    # Set up learning rate scheduler to decay learning rates dynamically
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

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    init_epoch = 0
    best_loss = float("inf")

    for epoch_idx in range(init_epoch, config.training.num_epochs):
        # Tick / tock
        epoch_start_time = time()

        # Batch average metrics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        encoder_losses = AverageMeter()
        refiner_losses = AverageMeter()

        # switch models to training mode
        encoder.train()
        decoder.train()
        merger.train()
        refiner.train()

        batch_end_time = time()
        n_batches = len(train_set_dataloader)

        for batch_idx, (
            input_images,
            ground_truth_volumes,
        ) in enumerate(train_set_dataloader):
            data_time.update(time() - batch_end_time)

            # Train the encoder, decoder, refiner, and merger
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

            # Append loss to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            # Log every 10 batches
            if batch_idx % 2 == 0:
                logging.info(
                    "[Epoch %d/%d][Batch %d/%d] EDLoss = %.4f RLoss = %.4f"
                    % (
                        epoch_idx + 1,
                        config.training.num_epochs,
                        batch_idx + 1,
                        n_batches,
                        encoder_loss.item(),
                        refiner_loss.item(),
                    )
                )

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        refiner_lr_scheduler.step()
        merger_lr_scheduler.step()

        # Calculate average loss for the epoch
        avg_encoder_loss = encoder_losses.avg
        avg_refiner_loss = refiner_losses.avg
        total_loss = avg_encoder_loss + avg_refiner_loss

        # Tick / tock
        epoch_end_time = time()
        logging.info(
            """[Epoch %d/%d] EpochTime = %.3f (s) EDLoss = %.4f
            RLoss = %.4f TotalLoss = %.4f"""
            % (
                epoch_idx + 1,
                config.training.num_epochs,
                epoch_end_time - epoch_start_time,
                avg_encoder_loss,
                avg_refiner_loss,
                total_loss,
            )
        )

        # Validate the network
        max_iou = validate_network(
            config, val_set_dataloader, encoder, decoder, merger, refiner
        )

        # Log epoch summary with IoU
        logging.info(
            """[Epoch %d/%d] EpochTime = %.3f (s) EDLoss = %.4f RLoss = %.4f
              TotalLoss = %.4f IoU = %.4f"""
            % (
                epoch_idx + 1,
                config.training.num_epochs,
                epoch_end_time - epoch_start_time,
                avg_encoder_loss,
                avg_refiner_loss,
                total_loss,
                max_iou,
            )
        )

        # Save checkpoint if this is the best loss so far
        if total_loss < best_loss:
            best_loss = total_loss
            save_checkpoint(
                config,
                encoder,
                decoder,
                merger,
                refiner,
                epoch_idx + 1,
                total_loss,
                max_iou,
            )

    # Save final checkpoint
    save_checkpoint(
        config,
        encoder,
        decoder,
        merger,
        refiner,
        config.training.num_epochs,
        total_loss,
        max_iou,
    )
    logging.info("Training completed! Final checkpoint saved.")


def validate_network(
    config: Config,
    val_set_dataloader: torch.utils.data.DataLoader,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    merger: torch.nn.Module,
    refiner: torch.nn.Module,
):
    """Validate the network and return mean IoU."""
    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(val_set_dataloader)
    encoder_losses = AverageMeter()
    refiner_losses = AverageMeter()
    all_ious = []  # Store IoU for each sample

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    for batch_idx, (input_images, ground_truth_volume) in enumerate(
        val_set_dataloader
    ):
        with torch.no_grad():
            image_features = encoder(input_images)
            raw_features, generated_volume = decoder(image_features)
            generated_volume = merger(raw_features, generated_volume)

            encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10

            generated_volume = refiner(generated_volume)

            refiner_loss = bce_loss(generated_volume, ground_truth_volume) * 10

            # Append loss to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            # IoU per sample
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

            # Log sample results
            logging.info(
                "Val[%d/%d] EDLoss = %.4f RLoss = %.4f IoU = %s"
                % (
                    batch_idx + 1,
                    n_samples,
                    encoder_loss.item(),
                    refiner_loss.item(),
                    ["%.4f" % si for si in sample_iou],
                )
            )

    # Calculate mean IoU across all samples
    mean_iou = np.mean(all_ious, axis=0)

    # Log validation summary
    logging.info(
        "Validation Summary - Avg EDLoss: %.4f, Avg RLoss: %.4f, Mean IoU: %s"
        % (
            encoder_losses.avg,
            refiner_losses.avg,
            ["%.4f" % mi for mi in mean_iou],
        )
    )

    # Return the maximum IoU (best performance across thresholds)
    max_iou = np.max(mean_iou)
    logging.info("Best IoU: %.4f" % max_iou)

    return max_iou
