import os

# limit resource usage
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import random
import yaml

import wandb
import torch
import numpy as np
from tqdm import tqdm

from vit_spatial_spectral import ViTSpatialSpectral
from vit_simmim_original import SimMIMSpatialSpectral
from utilities import get_optimizers, get_unsupervised_data

SEED = 5
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
  
if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    hyperparams = {
        "dataset": "enmap", # houston or enmap
        "image_size": 8,
        "batch_size": 64,

        "clip_min": -200,
        "clip_max": 10000,
        "device": device,
        "logging_freq": 10, # steps
        "skip_val": False, # note: this affects the lr scheduler
        "model_save_freq": 1, # in epochs
        "encoder_name": "ViTSpatialSpectral", 

        "patch_size": 1,
        "band_patch_size": 10,
        "rgb_only": False,
        "spectral_pos_embed": False, # default True
        "blockwise_patch_embed": True,
        "spectral_only": False,
        "to_pixels_per_spectral_block": True,
        "tube_masking": True,

        "clip_grad_norm": True,

        "epoch": 800,
        "lr": 8e-3,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "weight_decay": 0.05,
        "seed": SEED,
    }
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    hyperparams.update(config["data"][hyperparams["dataset"]])
    hyperparams.update(config["transformer"])
    hyperparams.update(config["masked_modeling"])

    print("batch size:", hyperparams["batch_size"])

    if hyperparams["encoder_name"] == "ViTSpatialSpectral":
        spectral_pos = torch.arange(hyperparams["n_bands"] // hyperparams["band_patch_size"])
        model = ViTSpatialSpectral(
            image_size=hyperparams["image_size"],
            spatial_patch_size=hyperparams["patch_size"],
            spectral_patch_size=hyperparams["band_patch_size"],
            num_classes=hyperparams["n_classes"],
            dim=hyperparams["transformer_dim"],
            depth=hyperparams["transformer_depth"],
            heads=hyperparams["transformer_n_heads"],
            mlp_dim=hyperparams["transformer_mlp_dim"],
            dropout = hyperparams["transformer_dropout"],
            emb_dropout = hyperparams["transformer_emb_dropout"],
            channels=hyperparams["n_bands"],
            spectral_pos_embed=hyperparams["spectral_pos_embed"],
            spectral_pos=spectral_pos,
            blockwise_patch_embed=hyperparams["blockwise_patch_embed"],
            spectral_only=hyperparams["spectral_only"],
    )
    else:
        raise NotImplementedError(f"method {hyperparams['encoder_name']} not available")


    model = SimMIMSpatialSpectral( #SimMIMSpectralFormer(
        encoder=model,
        intermediate_losses=hyperparams["mim_intermediate_losses"],
        masking_ratio=hyperparams["mim_masking_ratio"],
        mask_patch_size=hyperparams["mim_mask_patch_size"],
        to_pixels_per_spectral_block=hyperparams["to_pixels_per_spectral_block"],
        tube_masking=hyperparams["tube_masking"],
        #reconstruction_loss_on_all_tokens=hyperparams["reconstruction_loss_on_all_tokens"],
    )

    model.to(device)

    optimizer, scheduler = get_optimizers(model, hyperparams)

    if hyperparams["clip_grad_norm"]:
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    print(f"Model encoder_name: {hyperparams['encoder_name']}")
    model_params = sum([p.numel() for p in model.parameters()])
    print(f"Model parameters: {model_params:,}")
    hyperparams["num_params"] = model_params

    dataloader, val_dataloader = get_unsupervised_data(hyperparams, device, hyperparams["seed"])

    run = wandb.init(project="enmap-mim-spatial-spectral", config=hyperparams, save_code=True)
    run_id = run.id
    hyperparams["run_id"] = run_id
    os.mkdir(f"models/{run_id}/")

    losses = []

    step = 0

    wandb.config.update(hyperparams)
    print("Epochs:", hyperparams["epoch"], "Batch size:", hyperparams["batch_size"])
    epochs_pbar = tqdm(range(hyperparams["epoch"]))
    for epoch in epochs_pbar:
        epochs_pbar.set_description(f"Epoch {epoch}")
        model.train()

        train_pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        for idx, batch in train_pbar:
            train_pbar.set_description(f"Training {step:,}")

            if hyperparams["image_size"] != 64 and hyperparams["dataset"] in ["dfc", "enmap"]:
                # select a image_size**2 patch at random location of the tile
                x,y = torch.randint(0, 64 - hyperparams["image_size"], (2,))
            else:
                x,y = 0,0

            img = batch["img"][:, :, x:x+hyperparams["image_size"], y:y+hyperparams["image_size"]].to(device)

            optimizer.zero_grad()
                
            loss = model(img)
            
            if torch.isnan(loss):
                raise ValueError("Loss is NaN")
                
            loss.backward()
            optimizer.step()
            step += 1

            losses.append(loss.detach().item())

            if step % hyperparams["logging_freq"] == 0:
                wandb.log({
                    "epoch": epoch,
                    "loss": np.array(losses[-1*hyperparams["logging_freq"]:]).mean(),
                    "lr": optimizer.param_groups[0]['lr'],
                },
                step=step,
            )

        # log at end of training epoch (to same step as validation stats below)
        wandb.log({"epoch": epoch, "loss": loss.item()}, step=step)


        if epoch % hyperparams["model_save_freq"] == 0:
            stats = {
                "losses": torch.tensor(losses),
                "config": hyperparams,
                "model_state_dict": model.state_dict(),
                "lr_current": optimizer.param_groups[0]['lr'],
                "input": img.detach(),
                "transformer_input": img,
            }
            torch.save(stats, f"models/{run_id}/model_{hyperparams['encoder_name']}_ep{epoch}.pth")

            if epoch == 10 and hyperparams["model_save_freq"] == 1:
                hyperparams["model_save_freq"] = 10

        if not hyperparams["skip_val"]:
            # validation 
            with torch.no_grad():
                val_losses = []
                val_accs = []
                model.eval()
                val_pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False)
                for idx, batch in val_pbar: 
                    val_pbar.set_description(f"Validation {step:,}")
                    img_whole = batch["img"]
                    
                    if hyperparams["image_size"] != 64 and hyperparams["dataset"] in ["dfc", "enmap"]:
                        # sliding window with stride == window size
                        for x in range(0, 64, hyperparams["image_size"]):
                            for y in range(0, 64, hyperparams["image_size"]):
                                img = img_whole[:, :, x:x+hyperparams["image_size"], y:y+hyperparams["image_size"]].to(device)
                            
                                loss = model(img)

                                val_losses.append(loss.detach().item())

                    else:
                        img = img_whole.to(device)
                        loss = model(img)
                        val_losses.append(loss.detach().item())

                wandb.log({
                    "epoch": epoch,
                    "val_loss": torch.tensor(val_losses).mean().item(),
                    },
                    step=step,
                )

            if hyperparams["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(torch.tensor(val_losses).mean().item())
        if hyperparams["scheduler"] == "cosine":
            scheduler.step()