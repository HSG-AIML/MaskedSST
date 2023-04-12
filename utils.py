import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import torch
import torchvision

from vit_spatial_spectral import ViTSpatialSpectral
from data_enmap import EnMAPWorldCoverDataset, StandardizeEnMAP, ToTensor, WorldCoverLabelTransform, DFCLabelTransform
from data_houston2018 import Houston2018Dataset, StandardizeHouston2018, Houston2018LabelTransform

def get_optimizers(model, hyperparams):
    if hyperparams["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
    elif hyperparams["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])

    if hyperparams["scheduler"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.9, patience=5, verbose=True
        )
    elif hyperparams["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=0, last_epoch=-1, verbose=False,
        )

    return optimizer, scheduler


def get_unsupervised_data(hyperparams, device):

    train_path = hyperparams["train_path"]
    if hyperparams["dataset"] == "enmap":
        label_transform = WorldCoverLabelTransform()
        standardizer = StandardizeEnMAP()
    elif hyperparams["dataset"] == "dfc":
        label_transform = DFCLabelTransform()
        standardizer = StandardizeEnMAP()
    elif hyperparams["dataset"] == "houston2018":
        train_label_path = hyperparams["train_label_path"]
        standardizer = StandardizeHouston2018()
        label_transform = Houston2018LabelTransform()

    transforms = torchvision.transforms.Compose([
            standardizer,
            ToTensor(),
    ])

    if hyperparams["dataset"] in ["dfc", "enmap"]:
        dataset = EnMAPWorldCoverDataset(train_path, transforms, label_transform, device, test=False, target_type="unlabeled", remove_bands=hyperparams["remove_bands"], rgb_only=hyperparams["rgb_only"])
    elif hyperparams["dataset"] == "houston2018":
        dataset = Houston2018Dataset(train_path, train_label_path, transforms, label_transform, patch_size=hyperparams["image_size"], test=False, drop_unlabeled=False, fix_train_patches=False)

    train_fraction = 0.9
    data_fraction = 1 # share of the train set that is actually used, does not affect val set size
    hyperparams.update({
        "train_fraction": train_fraction,
        "data_fraction": data_fraction,
    })

    num_train_samples = int(len(dataset) * train_fraction)
    num_val_samples = len(dataset) - num_train_samples
    num_train_samples = int(num_train_samples * data_fraction)

    print(f"{train_path=}")
    print(f"{train_fraction=}, {data_fraction=}")
    print(f"{num_train_samples=}, {num_val_samples=}, {len(dataset)=}")

    val_dataset, train_dataset, rest = torch.utils.data.random_split(dataset, [num_val_samples, num_train_samples, len(dataset) - num_train_samples - num_val_samples], generator=torch.Generator().manual_seed(hyperparams["seed"]))
    print(f"{len(train_dataset)=}")
    print(f"{len(val_dataset)=}")

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparams["batch_size"], drop_last=True, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=hyperparams["batch_size"], drop_last=True, shuffle=False, num_workers=4)

    return dataloader, val_dataloader


def get_supervised_data(hyperparams, pixelwise, device):

    train_path = hyperparams["train_path"]
    if hyperparams["dataset"] == "worldcover":
        label_transform = WorldCoverLabelTransform()
        standardizer = StandardizeEnMAP()
    elif hyperparams["dataset"] == "dfc":
        label_transform = DFCLabelTransform()
        standardizer = StandardizeEnMAP()
    elif hyperparams["dataset"] == "houston2018":
        train_label_path = hyperparams["train_label_path"]
        standardizer = StandardizeHouston2018()
        label_transform = Houston2018LabelTransform()

    transforms = torchvision.transforms.Compose([
            # todo: rotate
            standardizer,
            ToTensor(),
    ])

    if hyperparams["dataset"] in ["dfc", "enmap"]:
        dataset = EnMAPWorldCoverDataset(train_path, transforms, label_transform, device, test=False, target_type=hyperparams["dataset"], remove_bands=hyperparams["remove_bands"], rgb_only=hyperparams["rgb_only"])
    elif hyperparams["dataset"] == "houston2018":
        dataset = Houston2018Dataset(train_path, train_label_path, transforms, label_transform, patch_size=hyperparams["image_size"]-hyperparams["patch_sub"], test=False, drop_unlabeled=True, fix_train_patches=False, pixelwise=pixelwise, rgb_only=hyperparams["rgb_only"])

    num_train_samples = int(len(dataset) * hyperparams["train_fraction"]) # note: overwritten below
    num_val_samples = len(dataset) - num_train_samples
    num_train_samples = int(num_train_samples * hyperparams["data_fraction"])

    val_dataset, train_dataset, rest = torch.utils.data.random_split(dataset, [num_val_samples, num_train_samples, len(dataset) - num_train_samples - num_val_samples], generator=torch.Generator().manual_seed(hyperparams["seed"]))
    # patches_per_tile = (64-2*(hyperparams['patch_size']//2))**2# 
    print(f"{len(train_dataset)=}")
    print(f"{len(val_dataset)=}")

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparams["batch_size"], drop_last=False, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=hyperparams["batch_size"], drop_last=False, shuffle=False, num_workers=4)

    return dataloader, val_dataloader

def verify_sweep_params(hyperparams):
        # todo: handle this within wandb sweep init
    if hyperparams["pretrain_run_id"] in ["none", "None"]:
        pretrain_run_id = None
    else:
        pretrain_run_id = hyperparams["pretrain_run_id"]

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

    return pretrain_run_id, linear_eval, spectral_pos_embed, blockwise_patch_embed, spectral_only, pixelwise, shifting_window, overwrite_li_optim

def load_checkpoint(hyperparams, model, classifier_name, device):
    print("Intializing pre-trained weights...")
    checkpoint = torch.load(hyperparams["checkpoint_path"], map_location=device)

    encoder_weights = checkpoint["model_state_dict"]
    for k in list(encoder_weights.keys()):
        encoder_weights[k.replace("encoder.", "")] = encoder_weights[k]

        # delete old keys and those that are not part of the encoder 
        del encoder_weights[k]

    # wrong output shape in pre-trained mpl_head
    if model.pixelwise:
        w,b = model.mlp_head[2].weight, model.mlp_head[2].bias
        linear_idx = 2
    else:
        w,b = model.mlp_head[1].weight, model.mlp_head[1].bias
        linear_idx = 1

    if hyperparams["patch_sub"]!= 0 and isinstance(model, ViTSpatialSpectral):
        # pre_trained with different image_size
        if encoder_weights.get("pos_embed") is not None:
            print(f"{encoder_weights['pos_embed'].shape=}")
            assert model.pos_embed.shape[1] == (hyperparams["image_size"] - hyperparams["patch_sub"])**2
            encoder_weights["pos_embed"] = encoder_weights["pos_embed"][:, :model.pos_embed.shape[1], :]
            print(f"{encoder_weights['pos_embed'].shape=}")

    del encoder_weights[f"{classifier_name}.1.bias"]
    del encoder_weights[f"{classifier_name}.1.weight"]
    encoder_weights[f"{classifier_name}.{linear_idx}.bias"] = b
    encoder_weights[f"{classifier_name}.{linear_idx}.weight"] = w
    print(model.load_state_dict(encoder_weights))

    return model