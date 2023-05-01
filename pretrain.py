import os

# limit resource usage
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import random

import wandb
import torch
import numpy as np
from tqdm import tqdm

from src.vit_spatial_spectral import ViTSpatialSpectral
from src.vit_simmim_original import SimMIMSpatialSpectral
from src.utils import get_pretrain_config, get_optimizers, get_unsupervised_data

SEED = 5
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    config = get_pretrain_config(
        "configs/pretrain_config.yaml", "configs/config.yaml", SEED, device
    )

    # create encoder
    assert (
        config.encoder_name == "ViTSpatialSpectral"
    ), f"encoder {config.encoder_name} not available"

    spectral_pos = torch.arange(config.n_bands // config.band_patch_size)
    model = ViTSpatialSpectral(
        image_size=config.image_size,
        spatial_patch_size=config.patch_size,
        spectral_patch_size=config.band_patch_size,
        num_classes=config.n_classes,
        dim=config.transformer_dim,
        depth=config.transformer_depth,
        heads=config.transformer_n_heads,
        mlp_dim=config.transformer_mlp_dim,
        dropout=config.transformer_dropout,
        emb_dropout=config.transformer_emb_dropout,
        channels=config.n_bands,
        spectral_pos_embed=config.spectral_pos_embed,
        spectral_pos=spectral_pos,
        blockwise_patch_embed=config.blockwise_patch_embed,
        spectral_only=config.spectral_only,
    )

    # wrap encoder for masked pre-training
    model = SimMIMSpatialSpectral(
        encoder=model,
        intermediate_losses=config.mim_intermediate_losses,
        masking_ratio=config.mim_masking_ratio,
        mask_patch_size=config.mim_mask_patch_size,
        to_pixels_per_spectral_block=config.to_pixels_per_spectral_block,
        tube_masking=config.tube_masking,
    ).to(device)

    optimizer, scheduler = get_optimizers(model, config)

    if config.clip_grad_norm:
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    config.model_params = sum([p.numel() for p in model.parameters()])

    dataloader, val_dataloader = get_unsupervised_data(config, device)

    # set-up training run
    run = wandb.init(
        project="enmap-mim-spatial-spectral", config=config, save_code=True
    )
    config.run_id = run.id
    wandb.config.update(config)
    os.mkdir(f"models/{config.run_id}/")

    step = 0
    losses = []

    epochs_pbar = tqdm(range(config.epoch))
    for epoch in epochs_pbar:
        epochs_pbar.set_description(f"Epoch {epoch}")
        model.train()

        train_pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        for idx, batch in train_pbar:
            train_pbar.set_description(f"Training {step:,}")

            if config.image_size != 64 and config.dataset in ["dfc", "enmap"]:
                # select a image_size**2 patch at random location of the tile
                x, y = torch.randint(0, 64 - config.image_size, (2,))
            else:
                x, y = 0, 0

            img = batch["img"][
                :, :, x : x + config.image_size, y : y + config.image_size
            ].to(device)

            optimizer.zero_grad()

            loss = model(img)

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            loss.backward()
            optimizer.step()
            step += 1

            losses.append(loss.detach().item())

            if step % config.logging_freq == 0:
                wandb.log(
                    {
                        "epoch": epoch,
                        "loss": np.array(losses[-1 * config.logging_freq :]).mean(),
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )

        # log at end of training epoch (to same step as validation stats below)
        wandb.log({"epoch": epoch, "loss": loss.item()}, step=step)

        if epoch % config.model_save_freq == 0:
            # save model checkpoint along with some statistics
            stats = {
                "losses": torch.tensor(losses),
                "config": config,
                "model_state_dict": model.state_dict(),
                "lr_current": optimizer.param_groups[0]["lr"],
                "input": img.detach(),
                "transformer_input": img,
            }
            torch.save(
                stats,
                f"models/{config.run_id}/model_{config.encoder_name}_ep{epoch}.pth",
            )

            if epoch == 10 and config.model_save_freq == 1:
                config.model_save_freq = 10

        if not config.skip_val:
            # validation
            with torch.no_grad():
                val_losses = []
                val_accs = []
                model.eval()
                val_pbar = tqdm(
                    enumerate(val_dataloader), total=len(val_dataloader), leave=False
                )
                for idx, batch in val_pbar:
                    val_pbar.set_description(f"Validation {step:,}")
                    img_whole = batch["img"]

                    if config.image_size != 64 and config.dataset in ["dfc", "enmap"]:
                        # sliding window with stride == window size
                        for x in range(0, 64, config.image_size):
                            for y in range(0, 64, config.image_size):
                                img = img_whole[
                                    :,
                                    :,
                                    x : x + config.image_size,
                                    y : y + config.image_size,
                                ].to(device)

                                loss = model(img)

                                val_losses.append(loss.detach().item())

                    else:
                        img = img_whole.to(device)
                        loss = model(img)
                        val_losses.append(loss.detach().item())

                wandb.log(
                    {
                        "epoch": epoch,
                        "val_loss": torch.tensor(val_losses).mean().item(),
                    },
                    step=step,
                )

            if config.scheduler == "ReduceLROnPlateau":
                scheduler.step(torch.tensor(val_losses).mean().item())
        if config.scheduler == "cosine":
            scheduler.step()
