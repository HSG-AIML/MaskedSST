import os

# limit resource usage
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import sys
import random

import wandb
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics import Accuracy

from DeepHyperX.models import get_model
from src.vit_original import ViTRGB
from src.vit_spatial_spectral import ViTSpatialSpectral
from src.utils import (
    get_supervised_data,
    load_checkpoint,
    get_finetune_config,
    get_val_epochs,
    train_step,
)
from src.utils import validate_downstream as validate

SEED = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset_name = sys.argv[1]
    valid_datasets = ["enmap", "houston2018"]
    assert (
        dataset_name in valid_datasets
    ), f"Please provide a valid dataset name from {valid_datasets}, provided: {dataset_name=}"

    config = get_finetune_config(
        f"configs/finetune_config_{dataset_name}.yaml",
        "configs/config.yaml",
        SEED,
        device,
    )

    run = wandb.init(config=config, project="downstream")
    config.run_id = run.id

    if config.method_name == "li":
        model, optimizer, criterion, model_params = get_model(
            name=config.method_name,
            n_classes=config.n_classes,
            n_bands=config.n_bands,
            ignored_labels=[config.ignored_label],
            patch_size=config.image_size - config.patch_sub,  # one prediction per patch
        )
    elif config.method_name == "ViTSpatialSpectral":
        model = ViTSpatialSpectral(
            image_size=config.image_size - config.patch_sub,
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
            spectral_pos=config.spectral_pos,
            spectral_pos_embed=config.spectral_pos_embed,
            blockwise_patch_embed=config.blockwise_patch_embed,
            spectral_only=config.spectral_only,
            pixelwise=config.pixelwise,
            pos_embed_len=config.pos_embed_len,
        )
    elif config.method_name == "ViTRGB":
        model = ViTRGB(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=config.n_classes,
            dim=config.transformer_dim,
            depth=config.transformer_depth,
            heads=config.transformer_n_heads,
            mlp_dim=config.transformer_mlp_dim,
            dropout=config.transformer_dropout,
            emb_dropout=config.transformer_emb_dropout,
            channels=config.n_bands,
            pixelwise=True,  # one prediction per pixel, not per patch
        )
    else:
        raise NotImplementedError(f"method {config.method_name} not available")

    classifier_name = "fc" if config.method_name == "li" else "mlp_head"

    if config.checkpoint_path is not None:
        model = load_checkpoint(config, model, classifier_name, device)

    model.to(device)

    if config.linear_eval:
        print("Linear evaluation... only training mlp_head")
        for n, p in model.named_parameters():
            if not classifier_name in n:
                p.requires_grad = False
        params = list(getattr(model, classifier_name).parameters())
    else:
        # fine-tuning
        params = list(model.parameters())
        # set different LR for transformer and MLP head
        if config.lr != config.mlp_head_lr:
            mlp_param_list = [
                p for n, p in model.named_parameters() if classifier_name in n
            ]
            rest_param_list = [
                p for n, p in model.named_parameters() if classifier_name not in n
            ]
            params = [
                {"params": mlp_param_list, "lr": config.mlp_head_lr},
                {"params": rest_param_list},
            ]

    if config.method_name != "li" or config.overwrite_li_optim:
        optimizer = torch.optim.Adam(
            params, lr=config.lr, weight_decay=config.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss(ignore_index=config.ignored_label)
    else:
        criterion = criterion.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.9, patience=5, verbose=True
    )

    acc_criterion = Accuracy(
        "multiclass", num_classes=config.n_classes, average="macro"
    ).to(device)
    model_params = sum([p.numel() for p in model.parameters()])
    config.num_params = model_params

    print(f"Model name: {config.method_name}")
    print(f"Model parameters: {model_params:,}")

    dataloader, val_dataloader = get_supervised_data(config, device)

    os.mkdir(f"models/{config.run_id}/")

    losses = []
    accs = []
    macro_accs = []
    acc_per_epoch = []
    current_val_acc = 0
    best_val_acc = 0
    step = 0
    epoch = 0
    validation_epochs = get_val_epochs(config, dataloader)

    wandb.config.update(config)

    epochs_pbar = tqdm(range(config.epoch + 1))
    while epoch < config.epoch + 1 or step < config.max_steps + 1:
        epochs_pbar.set_description(f"Epoch {epoch}")
        model.train()

        train_pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        for idx, batch in train_pbar:
            train_pbar.set_description(f"Training {step:,}")

            img = batch["img"]
            label = batch["label"]

            loss, acc, macro_acc = train_step(
                img, label, model, config, device, criterion, optimizer, acc_criterion
            )
            step += 1

            losses.append(loss.detach().item())
            accs.append(acc.detach().item())
            macro_accs.append(macro_acc.detach().item())

            if step % config.logging_freq == 0:
                wandb.log(
                    {
                        "epoch": epoch,
                        "acc": np.array(accs[-1 * config.logging_freq :]).mean(),
                        "macro_acc": np.array(
                            macro_accs[-1 * config.logging_freq :]
                        ).mean(),
                        "loss": np.array(losses[-1 * config.logging_freq :]).mean(),
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )

        # log at end of training epoch (to same step as validation stats below)
        wandb.log({"epoch": epoch, "acc": acc.item(), "loss": loss.item()}, step=step)
        if epoch in validation_epochs:
            val_losses, best_val_acc = validate(
                config,
                epoch,
                model,
                val_dataloader,
                criterion,
                acc_criterion,
                step,
                best_val_acc,
                optimizer.param_groups[0]["lr"],
                device,
                pixelwise=config.pixelwise,
            )

        scheduler.step(torch.tensor(val_losses).mean().item())
        epoch += 1
