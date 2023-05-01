import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import yaml
import wandb
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from einops import rearrange

from src.vit_spatial_spectral import ViTSpatialSpectral, get_pos_for_spectral_embedding
from src.data_enmap import (
    EnMAPWorldCoverDataset,
    StandardizeEnMAP,
    ToTensor,
    WorldCoverLabelTransform,
    DFCLabelTransform,
    invalid_l2_bands,
)
from src.data_houston2018 import (
    Houston2018Dataset,
    StandardizeHouston2018,
    Houston2018LabelTransform,
)
from src.data_enmap import wavelengths as enmaps_waves
from src.data_houston2018 import wavelengths as houston_waves


def get_optimizers(model, config):
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    if config.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.9, patience=5, verbose=True
        )
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=0,
            last_epoch=-1,
            verbose=False,
        )

    return optimizer, scheduler


def get_unsupervised_data(config, device):

    train_path = config.train_path
    if config.dataset == "enmap":
        label_transform = WorldCoverLabelTransform()
        standardizer = StandardizeEnMAP()
    elif config.dataset == "dfc":
        label_transform = DFCLabelTransform()
        standardizer = StandardizeEnMAP()
    elif config.dataset == "houston2018":
        train_label_path = config.train_label_path
        standardizer = StandardizeHouston2018()
        label_transform = Houston2018LabelTransform()

    transforms = torchvision.transforms.Compose(
        [
            standardizer,
            ToTensor(),
        ]
    )

    if config.dataset in ["dfc", "enmap"]:
        dataset = EnMAPWorldCoverDataset(
            train_path,
            transforms,
            label_transform,
            device,
            test=False,
            target_type="unlabeled",
            remove_bands=config.remove_bands,
            rgb_only=config.rgb_only,
        )
    elif config.dataset == "houston2018":
        dataset = Houston2018Dataset(
            train_path,
            train_label_path,
            transforms,
            label_transform,
            patch_size=config.image_size,
            test=False,
            drop_unlabeled=False,
            fix_train_patches=False,
        )

    num_train_samples = int(len(dataset) * config.train_fraction)
    num_val_samples = len(dataset) - num_train_samples
    num_train_samples = int(num_train_samples * config.data_fraction)

    val_dataset, train_dataset, _ = torch.utils.data.random_split(
        dataset,
        [
            num_val_samples,
            num_train_samples,
            len(dataset) - num_train_samples - num_val_samples,
        ],
        generator=torch.Generator().manual_seed(config.seed),
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=4,
    )

    return dataloader, val_dataloader


def get_supervised_data(config, device):

    train_path = config.train_path
    if config.dataset == "worldcover":
        label_transform = WorldCoverLabelTransform()
        standardizer = StandardizeEnMAP()
    elif config.dataset == "dfc":
        label_transform = DFCLabelTransform()
        standardizer = StandardizeEnMAP()
    elif config.dataset == "houston2018":
        train_label_path = config.train_label_path
        standardizer = StandardizeHouston2018()
        label_transform = Houston2018LabelTransform()

    transforms = torchvision.transforms.Compose(
        [
            # todo: rotate
            standardizer,
            ToTensor(),
        ]
    )

    if config.dataset in ["dfc", "enmap"]:
        dataset = EnMAPWorldCoverDataset(
            train_path,
            transforms,
            label_transform,
            device,
            test=False,
            target_type=config.dataset,
            remove_bands=config.remove_bands,
            rgb_only=config.rgb_only,
        )
    elif config.dataset == "houston2018":
        dataset = Houston2018Dataset(
            train_path,
            train_label_path,
            transforms,
            label_transform,
            patch_size=config.image_size - config.patch_sub,
            test=False,
            drop_unlabeled=True,
            fix_train_patches=False,
            pixelwise=config.pixelwise,
            rgb_only=config.rgb_only,
        )

    num_train_samples = int(
        len(dataset) * config.train_fraction
    )  # note: overwritten below
    num_val_samples = len(dataset) - num_train_samples
    num_train_samples = int(num_train_samples * config.data_fraction)

    val_dataset, train_dataset, rest = torch.utils.data.random_split(
        dataset,
        [
            num_val_samples,
            num_train_samples,
            len(dataset) - num_train_samples - num_val_samples,
        ],
        generator=torch.Generator().manual_seed(config.seed),
    )
    # patches_per_tile = (64-2*(config.patch_size//2))**2#
    print(f"{len(train_dataset)=}")
    print(f"{len(val_dataset)=}")

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=4,
    )

    return dataloader, val_dataloader


def verify_sweep_params(hyperparams):
    """Ensure that boolean flags are correctly handeled"""
    if hyperparams["checkpoint_path"] in ["none", "None"]:
        checkpoint_path = None
    else:
        checkpoint_path = hyperparams["checkpoint_path"]

    if hyperparams["linear_eval"] in [False, "false", "False"]:
        linear_eval = False
    else:
        linear_eval = True

    if hyperparams.get("spectral_pos_embed") in [False, "false", "False"]:
        spectral_pos_embed = False
    else:
        spectral_pos_embed = True

    if hyperparams.get("blockwise_patch_embed") in [False, "false", "False"]:
        blockwise_patch_embed = False
    else:
        blockwise_patch_embed = True

    if hyperparams.get("spectral_only") in [False, "false", "False", None]:
        spectral_only = False
    else:
        spectral_only = True

    if hyperparams.get("pixelwise") in [False, "false", "False", None]:
        pixelwise = False
    else:
        pixelwise = True

    if hyperparams["shifting_window"] in [False, "false", "False"]:
        shifting_window = False
    else:
        shifting_window = True

    if hyperparams["overwrite_li_optim"] in [False, "false", "False"]:
        overwrite_li_optim = False
    else:
        overwrite_li_optim = True

    return (
        checkpoint_path,
        linear_eval,
        spectral_pos_embed,
        blockwise_patch_embed,
        spectral_only,
        pixelwise,
        shifting_window,
        overwrite_li_optim,
    )


def load_checkpoint(config, model, classifier_name, device):
    print("Initializing pre-trained weights...")
    checkpoint = torch.load(config.checkpoint_path, map_location=device)

    encoder_weights = checkpoint["model_state_dict"]
    for k in list(encoder_weights.keys()):
        encoder_weights[k.replace("encoder.", "")] = encoder_weights[k]

        # delete old keys and those that are not part of the encoder
        del encoder_weights[k]

    # wrong output shape in pre-trained mpl_head
    if model.pixelwise:
        w, b = model.mlp_head[2].weight, model.mlp_head[2].bias
        linear_idx = 2
    else:
        w, b = model.mlp_head[1].weight, model.mlp_head[1].bias
        linear_idx = 1

    if config.patch_sub != 0 and isinstance(model, ViTSpatialSpectral):
        # pre_trained with different image_size
        if encoder_weights.get("pos_embed") is not None:
            print(f"{encoder_weights['pos_embed'].shape=}")
            assert (
                model.pos_embed.shape[1] == (config.image_size - config.patch_sub) ** 2
            )
            encoder_weights["pos_embed"] = encoder_weights["pos_embed"][
                :, : model.pos_embed.shape[1], :
            ]
            print(f"{encoder_weights['pos_embed'].shape=}")

    del encoder_weights[f"{classifier_name}.1.bias"]
    del encoder_weights[f"{classifier_name}.1.weight"]
    encoder_weights[f"{classifier_name}.{linear_idx}.bias"] = b
    encoder_weights[f"{classifier_name}.{linear_idx}.weight"] = w
    print(model.load_state_dict(encoder_weights))

    return model


def get_pretrain_config(pretrain_config_path, general_config_path, seed, device):
    # consolidate hyperparameters
    with open(pretrain_config_path, "r") as f:
        hyperparams = yaml.safe_load(f)
    with open(general_config_path, "r") as f:
        config = yaml.safe_load(f)

    hyperparams.update(config["data"][hyperparams["dataset"]])
    hyperparams.update(config["transformer"])
    hyperparams.update(config["masked_modeling"])
    hyperparams["seed"] = seed
    hyperparams["device"] = device

    return Dotdict(hyperparams)


class Dotdict(object):
    def __init__(self, data):
        self.__dict__.update(data)


def get_finetune_config(finetune_config_path, general_config_path, seed, device):
    with open(finetune_config_path, "r") as f:
        hyperparams = yaml.safe_load(f)
    with open(general_config_path, "r") as f:
        config = yaml.safe_load(f)

    hyperparams.update(config["data"][hyperparams["dataset"]])
    hyperparams.update(config["transformer"])

    hyperparams["seed"] = seed
    hyperparams["device"] = device

    if hyperparams["method_name"] == "li":
        assert hyperparams["pixelwise"]
    elif hyperparams["method_name"] == "ViTSpatialSpectral":
        hyperparams["spectral_pos"] = get_spectral_pos_embedding(
            hyperparams["dataset"],
            hyperparams["n_bands"],
            hyperparams["band_patch_size"],
        )

    if hyperparams["pixelwise"] and hyperparams["image_size"] % 2 == 0:
        # make sure there is a "center pixel"
        hyperparams["patch_sub"] = 1
    else:
        hyperparams["patch_sub"] = 0

    return Dotdict(hyperparams)


def get_sweep_finetune_config(finetune_config_path, general_config_path):
    with open(finetune_config_path, "r") as f:
        hyperparams_default = yaml.safe_load(f)
    with open(general_config_path, "r") as f:
        config = yaml.safe_load(f)

    hyperparams_default.update(config["data"][hyperparams_default["dataset"]])
    hyperparams_default.update(config["transformer"])

    run = wandb.init(config=hyperparams_default, project="enmap-simmim-downstream")
    hyperparams = wandb.config
    hyperparams["run_id"] = run.id

    if hyperparams["method_name"] == "li":
        assert hyperparams["pixelwise"]
    elif hyperparams["method_name"] == "ViTSpatialSpectral":
        hyperparams["spectral_pos"] = get_spectral_pos_embedding(
            hyperparams["dataset"],
            hyperparams["n_bands"],
            hyperparams["band_patch_size"],
        )

    (
        checkpoint_path,
        linear_eval,
        spectral_pos_embed,
        blockwise_patch_embed,
        spectral_only,
        pixelwise,
        shifting_window,
        overwrite_li_optim,
    ) = verify_sweep_params(hyperparams)
    hyperparams.update(
        {
            "checkpoint_path": checkpoint_path,
            "linear_eval": linear_eval,
            "spectral_pos_embed": spectral_pos_embed,
            "blockwise_patch_embed": blockwise_patch_embed,
            "spectral_only": spectral_only,
            "pixelwise": pixelwise,
            "shifting_window": shifting_window,
            "overwrite_li_optim": overwrite_li_optim,
        }
    )

    return run, Dotdict(hyperparams)


def get_spectral_pos_embedding(dataset, n_bands, band_patch_size):
    # configure spectral positional embedding
    if dataset in ["worldcover", "dfc"]:
        spectral_pos = torch.arange(n_bands // band_patch_size)
    elif dataset == "houston2018":
        # map spectral tokens to their positions in enmap spectral sequence
        spectral_pos = get_pos_for_spectral_embedding(
            band_patch_size,
            houston_waves,
            np.array(enmaps_waves)[~np.array(invalid_l2_bands)],
        )
    else:
        raise NotImplementedError(f"Unknown dataset {dataset=}")

    return spectral_pos


def get_val_epochs(config, dataloader):
    """fix the number of validation runs
    training will last for `epochs` or `max_steps`, whatever takes longer
    for small data_fraction and fixed batch_size, epochs will be very short
    and and validation time will dominate"""

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * config.epoch
    if total_steps > config.max_steps:
        # max epochs is reached first, eval after each epoch
        validation_epochs = torch.arange(config.epoch)
    else:
        # run will stop when max_steps is reached, still only eval `epoch` many times
        total_epochs = config.max_steps // steps_per_epoch
        validation_epochs = list(map(int, np.linspace(0, total_epochs, config.epoch)))

    return validation_epochs


def stack_image_batch(config, img, label):
    """tile image into multiple image_size,image_size patches
    and stack along batch dimension"""
    cutoff_h = img.shape[2] % (config.image_size - config.patch_sub)
    cutoff_w = img.shape[3] % (config.image_size - config.patch_sub)
    assert cutoff_h == cutoff_w
    if cutoff_h != 0:
        # remove border pixels s.t. image is divisible by patch size
        img = img[:, :, :-cutoff_h, :-cutoff_w]
        label = label[:, :-cutoff_h, :-cutoff_w]
    img = rearrange(
        img,
        "b c (h p1) (w p2) -> (b h w) c p1 p2",
        p1=config.image_size - config.patch_sub,
        p2=config.image_size - config.patch_sub,
    )
    label = rearrange(
        label,
        "b (h p1) (w p2) -> (b h w) p1 p2",
        p1=config.image_size - config.patch_sub,
        p2=config.image_size - config.patch_sub,
    )

    return img, label


def validate_downstream(
    config,
    epoch,
    model,
    val_dataloader,
    criterion,
    acc_criterion,
    step,
    best_val_acc,
    lr,
    device,
    pixelwise=False,
):
    with torch.no_grad():
        val_losses = []
        val_accs = []
        val_macro_accs = []
        model.eval()
        val_pbar = tqdm(
            enumerate(val_dataloader), total=len(val_dataloader), leave=False
        )
        for idx, batch in val_pbar:
            val_pbar.set_description(f"Validation {step:,}")
            img_whole = batch["img"]
            label_whole = batch["label"]

            if config.image_size != 64 and config.dataset in ["dfc", "worldcover"]:
                # validate each tile sub-patch
                for x in range(0, 64, config.image_size - config.patch_sub):
                    for y in range(0, 64, config.image_size - config.patch_sub):
                        img = img_whole[
                            :,
                            :,
                            x : x + config.image_size - config.patch_sub,
                            y : y + config.image_size - config.patch_sub,
                        ].to(device)
                        label = label_whole[
                            :,
                            x : x + config.image_size - config.patch_sub,
                            y : y + config.image_size - config.patch_sub,
                        ].to(device)
                        if x + config.image_size >= 64 or y + config.image_size > 64:
                            continue

                        if config.method_name == "li" or pixelwise:
                            # baseline model only predicts class for the center pixel of the patch
                            center_idx = (config.image_size - config.patch_sub) // 2

                            label = label[
                                :, center_idx, center_idx
                            ]  # .unsqueeze(1).unsqueeze(1)
                            if config.method_name == "li":
                                img = img.unsqueeze(1)

                        output = model(img)
                        loss = criterion(output, label)

                        pred = output.argmax(dim=1)
                        valid_idx = label != config.ignored_label
                        acc = (pred[valid_idx] == label[valid_idx]).sum() / pred[
                            valid_idx
                        ].numel()
                        macro_acc = acc_criterion(
                            pred[valid_idx].to(int), label[valid_idx]
                        )

            else:
                img = img_whole.to(device)
                label = label_whole.to(device)

                if config.method_name == "li" or pixelwise:
                    # baseline model only predicts class for the center pixel of the patch
                    if config.dataset != "houston2018":
                        center_idx = (config.image_size - config.patch_sub) // 2
                        label = label[
                            :, center_idx, center_idx
                        ]  # .unsqueeze(1).unsqueeze(1)
                    if config.method_name == "li":
                        img = img.unsqueeze(1)

                output = model(img)
                loss = criterion(output, label)
                pred = output.argmax(dim=1)
                valid_idx = label != config.ignored_label
                acc = (pred[valid_idx] == label[valid_idx]).sum() / pred[
                    valid_idx
                ].numel()
                if valid_idx.sum() != 0:
                    macro_acc = acc_criterion(pred[valid_idx].to(int), label[valid_idx])
                else:
                    macro_acc = acc

            val_losses.append(loss.detach().item())
            val_accs.append(acc.detach().item())
            val_macro_accs.append(macro_acc.detach().item())

        current_val_acc = torch.tensor(val_accs).mean().item()
        wandb.log(
            {
                "epoch": epoch,
                "val_acc": current_val_acc,
                "val_macro_acc": torch.tensor(val_macro_accs).mean().item(),
                "val_loss": torch.tensor(val_losses).mean().item(),
            },
            step=step,
        )

    if (
        epoch == config.epoch
        or current_val_acc > best_val_acc
        or epoch in config.checkpoint_save_epochs
    ):
        stats = {
            "config": config.__dict__,
            "model_state_dict": model.state_dict(),
            "lr_current": lr,
            "epoch": epoch,
        }

        if epoch == config.epoch or epoch in config.checkpoint_save_epochs:
            torch.save(
                stats, f"models/{config.run_id}/{config.method_name}_at_ep{epoch}.pth"
            )
        if current_val_acc > best_val_acc:
            torch.save(stats, f"models/{config.run_id}/best_{config.method_name}.pth")

            best_val_acc = current_val_acc

    return val_losses, best_val_acc


def train_step(img, label, model, config, device, criterion, optimizer, acc_criterion):
    if config.image_size != 64 and config.dataset in ["dfc", "worldcover"]:
        if config.shifting_window:
            # divide image into non-overlapping patches and stack them
            img, label = stack_image_batch(config, img, label)
        else:
            # train with one smaller random crop
            x, y = torch.randint(
                0, 64 - config.image_size - config.patch_sub, size=(2,)
            )
            img = img[
                :,
                :,
                x : x + config.image_size - config.patch_sub,
                y : y + config.image_size - config.patch_sub,
            ]
            label = label[
                :,
                x : x + config.image_size - config.patch_sub,
                y : y + config.image_size - config.patch_sub,
            ]

    if config.method_name == "li" or config.pixelwise:
        # baseline model only predicts class for the center pixel of the patch
        center_idx = (config.image_size - config.patch_sub) // 2
        if config.dataset in ["dfc", "worldcover"]:
            label = label[:, center_idx, center_idx]  # .unsqueeze(1).unsqueeze(1)

        # extra dim for 3D conv model
        if config.method_name == "li":
            img = img.unsqueeze(1)

    img = img.to(device)
    label = label.to(device)

    optimizer.zero_grad()

    output = model(img)
    loss = criterion(output, label)

    if torch.isnan(loss):
        ValueError("Loss is NaN")

    pred = output.argmax(dim=1)
    valid_idx = label != config.ignored_label
    acc = (pred[valid_idx] == label[valid_idx]).sum() / pred[valid_idx].numel()
    macro_acc = (
        acc_criterion(pred[valid_idx].to(int), label[valid_idx])
        if valid_idx.sum() != 0
        else acc
    )

    loss.backward()
    optimizer.step()

    return loss, acc, macro_acc
