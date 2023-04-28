import os

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
from einops import rearrange
from torchmetrics import Accuracy

from vit_original import ViTRGB
from vit_spatial_spectral import ViTSpatialSpectral, get_pos_for_spectral_embedding
from DeepHyperX.models import get_model

from src.data_enmap import wavelengths as enmaps_waves
from src.data_enmap import invalid_l2_bands
from src.data_houston2018 import wavelengths as houston_waves
from src.utils import get_supervised_data, verify_sweep_params


def validate(hyperparams, epoch, run_id, name, model, val_dataloader, criterion, acc_criterion, step, best_val_acc, lr, device, pixelwise=False):
    with torch.no_grad():
        val_losses = []
        val_accs = []
        val_macro_accs = []
        model.eval()
        val_pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False)
        for idx, batch in val_pbar: 
            val_pbar.set_description(f"Validation {step:,}")
            img_whole = batch["img"]
            label_whole = batch["label"]

            if hyperparams["image_size"] != 64 and hyperparams["dataset"] in ["dfc", "worldcover"]:
                # validate each tile sub-patch
                for x in range(0, 64, hyperparams["image_size"]-hyperparams["patch_sub"]):
                    for y in range(0, 64, hyperparams["image_size"]-hyperparams["patch_sub"]):
                        img = img_whole[:, :, x:x+hyperparams["image_size"]-hyperparams["patch_sub"], y:y+hyperparams["image_size"]-hyperparams["patch_sub"]].to(device)
                        label = label_whole[:, x:x+hyperparams["image_size"]-hyperparams["patch_sub"], y:y+hyperparams["image_size"]-hyperparams["patch_sub"]].to(device)
                        if x+hyperparams["image_size"] >= 64 or y + hyperparams["image_size"] > 64:
                            continue

                        if name == "li" or pixelwise:
                            # baseline model only predicts class for the center pixel of the patch
                            center_idx = (hyperparams["image_size"]-hyperparams["patch_sub"]) // 2

                            label = label[:, center_idx, center_idx]#.unsqueeze(1).unsqueeze(1)
                            if name == "li": img = img.unsqueeze(1)
            
                        output = model(img)
                        loss = criterion(output, label)

                        pred = output.argmax(dim=1)
                        valid_idx = label != hyperparams["ignored_label"]
                        acc = (pred[valid_idx] == label[valid_idx]).sum() / pred[valid_idx].numel()
                        macro_acc = acc_criterion(pred[valid_idx].to(int), label[valid_idx])

            else:
                img = img_whole.to(device)
                label = label_whole.to(device)

                if name == "li" or pixelwise:
                # baseline model only predicts class for the center pixel of the patch
                    if hyperparams["dataset"] != "houston2018":
                        center_idx = (hyperparams["image_size"]-hyperparams["patch_sub"]) // 2
                        label = label[:, center_idx, center_idx]#.unsqueeze(1).unsqueeze(1)
                    if name == "li": img = img.unsqueeze(1)

                output = model(img)
                loss = criterion(output, label)
                pred = output.argmax(dim=1)
                valid_idx = label != hyperparams["ignored_label"]
                acc = (pred[valid_idx] == label[valid_idx]).sum() / pred[valid_idx].numel()
                if valid_idx.sum() != 0:
                    macro_acc = acc_criterion(pred[valid_idx].to(int), label[valid_idx])
                else:
                    macro_acc = acc
                
            val_losses.append(loss.detach().item())
            val_accs.append(acc.detach().item())
            val_macro_accs.append(macro_acc.detach().item())

        current_val_acc = torch.tensor(val_accs).mean().item()
        wandb.log({
            "epoch": epoch,
            "val_acc": current_val_acc,
            "val_macro_acc": torch.tensor(val_macro_accs).mean().item(),
            "val_loss": torch.tensor(val_losses).mean().item(),
            },
            step=step,
        )
        
    if epoch == hyperparams["epoch"] or current_val_acc > best_val_acc or epoch in hyperparams["checkpoint_save_epochs"]:
        stats = {
            "config": dict(hyperparams),
            "model_state_dict": model.state_dict(),
            "lr_current": lr,
            "epoch": epoch,
        }

        if epoch == hyperparams["epoch"] or epoch in hyperparams["checkpoint_save_epochs"]:
            torch.save(stats, f"models/{run_id}/{name}_at_ep{epoch}.pth")
        if current_val_acc > best_val_acc:
            torch.save(stats, f"models/{run_id}/best_{name}.pth")

            best_val_acc = current_val_acc
        
    return val_losses, best_val_acc

  
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    name = "ViTSpatialSpectral"
    spectral_band_patch_size = 10

    hyperparams_default = {
        "dataset": "dfc", # dfc or enmap
        "ignored_label": -1,
        "device": device,
        "logging_freq": 10, # steps
        "val_freq": 1, # epochs
        "method_name": name,

        "train_fraction": 0.8,
        "data_fraction": 1., # does not affect val set size
        "pixelwise": False,
        "image_size": 8,
        "patch_size": 1,
        "band_patch_size": spectral_band_patch_size,
        "train_fraction": 0.8,
        "spectral_only": False,
        "rgb_only": False,
        "shifting_window": False, # on DFC dataset, if image_size < 64 don't take one random crop but move window image_sizeXimage_size window over 64x64 image

        "batch_size": 2, #
        "val_batch_size": 2,
        "epoch": 100,
        "checkpoint_save_epochs": [20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000],
        "max_steps": 1000,
        "lr": 5e-4, # was 5e-4
        "mlp_head_lr": 5e-3,
        "weight_decay": 5e-3,
        "pos_embed_len": None,

        "pretrain_run_id": None,
        "pretrain_run_epoch": 90,
        "linear_eval": False,
        "overwrite_li_optim": False,

    }

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    hyperparams_default.update(config["data"][hyperparams_default["dataset"]])
    hyperparams_default.update(config["transformer"])

    run = wandb.init(config=hyperparams_default, project="enmap-simmim-downstream")
    hyperparams = wandb.config

    print(f"{hyperparams['dataset']=}")
    print(f"{hyperparams['data_fraction']=}")

    pretrain_run_id, linear_eval, spectral_pos_embed, blockwise_patch_embed, spectral_only, pixelwise, shifting_window, overwrite_li_optim = verify_sweep_params(hyperparams)

    SEED = hyperparams["seed"]
    print(f"{SEED=}")
    print(f"{name=}")
    print(f"{hyperparams['method_name']=}")

    name = hyperparams['method_name']

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    if name == "li":
        assert pixelwise

    if pixelwise and hyperparams["image_size"] % 2 == 0:
        # make sure there is a "center pixel"
        hyperparams["patch_sub"] = 1
    else:
        hyperparams["patch_sub"] = 0
    
    print(f"{hyperparams['patch_sub']=}")

    if name == "li":

        model, optimizer, criterion, model_params = get_model(
            name=name,
            n_classes=hyperparams["n_classes"],
            n_bands=hyperparams["n_bands"],
            ignored_labels=[hyperparams["ignored_label"]],
            patch_size=hyperparams["image_size"] - hyperparams["patch_sub"], # one prediction per patch
        )
        print(f"{model.patch_size=}")

    elif name == "ViTSpatialSpectral":
        if hyperparams["dataset"] in ["worldcover", "dfc"]:
            # default
            spectral_pos = torch.arange(hyperparams["n_bands"] // hyperparams["band_patch_size"])
        elif hyperparams["dataset"] == "houston2018":
            # map spectral tokens to their positions in enmap spectral sequence
            spectral_pos = get_pos_for_spectral_embedding(
                hyperparams["band_patch_size"],
                houston_waves,
                np.array(enmaps_waves)[~np.array(invalid_l2_bands)],
            )
        else:
            raise NotImplementedError(f"Unknown dataset {hyperparams['dataset']=}")
        hyperparams["spectral_pos"] = spectral_pos
        model = ViTSpatialSpectral(
            image_size=hyperparams["image_size"]-hyperparams["patch_sub"],
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
            spectral_pos=spectral_pos,
            spectral_pos_embed=spectral_pos_embed,
            blockwise_patch_embed=blockwise_patch_embed,
            spectral_only=spectral_only,
            pixelwise=pixelwise,
            pos_embed_len=hyperparams["pos_embed_len"],
    )
    elif name == "ViTRGB":
        model = ViTRGB(
            image_size=hyperparams["image_size"],
            patch_size=hyperparams["patch_size"],
            num_classes=hyperparams["n_classes"],
            dim=hyperparams["transformer_dim"],
            depth=hyperparams["transformer_depth"],
            heads=hyperparams["transformer_n_heads"],
            mlp_dim=hyperparams["transformer_mlp_dim"],
            dropout = hyperparams["transformer_dropout"],
            emb_dropout = hyperparams["transformer_emb_dropout"],
            channels=hyperparams["n_bands"],
            pixelwise=True, # one prediction per pixel, not per patch
        )
    else:
        raise NotImplementedError(f"method {name} not available")

    classifier_name = "fc" if name == "li" else "mlp_head"

    if pretrain_run_id is not None:
        print("Intializing pre-trained weights...")
        pretrain_checkpoint_path = f"/netscratch/lscheibenreif/code/hyper/models/{pretrain_run_id}/model_{name}_ep{hyperparams['pretrain_run_epoch']}.pth"
        checkpoint = torch.load(pretrain_checkpoint_path, map_location=device)
        pretrain_run_config = checkpoint["config"]

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

    model.to(device)

    if linear_eval:
        print("Linear evaluation... only training mlp_head")
        for n,p in model.named_parameters():
            if not classifier_name in n:
                p.requires_grad = False
        if classifier_name == "mlp_head":
            params = list(model.mlp_head.parameters())
        elif classifier_name == "classifier":
            params = list(model.classifier.parameters())
        elif classifier_name == "fc":
            params = list(model.fc.parameters())
        print(f"Trainable params: {sum([p.numel() for p in params]):,}")
    else:
        params = list(model.parameters())
        print(f"Trainable params: {sum([p.numel() for p in params]):,}")
        
        # set different LR for transformer and MLP head
        if hyperparams["lr"] != hyperparams["mlp_head_lr"]:
            mlp_param_list = [p for n,p in model.named_parameters() if classifier_name in n]
            rest_param_list = [p for n,p in model.named_parameters() if classifier_name not in n]
            params = [
                {"params": mlp_param_list, "lr": hyperparams["mlp_head_lr"]},
                {"params": rest_param_list},
            ]

    if name != "li" or overwrite_li_optim:
        optimizer = torch.optim.Adam(params, lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
        criterion = torch.nn.CrossEntropyLoss(ignore_index=hyperparams["ignored_label"])
    else:
        criterion = criterion.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.9, patience=5, verbose=True
        )
    
    acc_criterion = Accuracy("multiclass", num_classes=hyperparams["n_classes"], average="macro").to(device)

    print(f"Model name: {name}")
    model_params = sum([p.numel() for p in model.parameters()])
    print(f"Model parameters: {model_params:,}")
    hyperparams["num_params"] = model_params

    dataloader, val_dataloader = get_supervised_data(hyperparams, pixelwise, device)

    run_id = run.id
    hyperparams["run_id"] = run_id
    os.mkdir(f"models/{run_id}/")

    losses = []
    accs = []
    macro_accs = []
    acc_per_epoch = []
    current_val_acc = 0
    best_val_acc = 0

    step = 0

    # fix the number of validation runs
    # training will last for `epochs` or `max_steps`, whatever takes longer
    # for small data_fraction and fixed batch_size, epochs will be very short
    # and it will take ages to run a validation loop after every single one
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * hyperparams["epoch"]
    if total_steps > hyperparams["max_steps"]:
        # max epochs is reached first, eval after each epoch
        validation_epochs = torch.arange(hyperparams["epoch"])
    else:
        # run will stop when max_steps is reached, still only eval `epoch` many times
        total_epochs = hyperparams["max_steps"] // steps_per_epoch
        validation_epochs = list(map(int, np.linspace(0, total_epochs, hyperparams["epoch"])))

    wandb.config.update(hyperparams)
    print("Epochs:", hyperparams["epoch"], "Batch size:", hyperparams["batch_size"])
    print(f"Seed: {hyperparams['seed']}")
    print(f"Linear eval: {linear_eval}", type(linear_eval))
    print(f"Pretrain run: {pretrain_run_id}", f"{pretrain_run_id is None}")
    print(f"{pixelwise=}")
    print(f"{hyperparams['linear_eval']=}")

    epochs_pbar = tqdm(range(hyperparams["epoch"] + 1))
    # for epoch in epochs_pbar:
    epoch = 0
    while epoch < hyperparams["epoch"] + 1 or step < hyperparams["max_steps"] + 1:
        epochs_pbar.set_description(f"Epoch {epoch}")
        model.train()

        train_pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        for idx, batch in train_pbar:
            train_pbar.set_description(f"Training {step:,}")
           
            img = batch["img"]
            label = batch["label"]

            if hyperparams["image_size"] != 64 and hyperparams["dataset"] in ["dfc", "worldcover"]:
                if shifting_window:
                    # tile image into multiple image_sizeXimage_size patches and stack along batch dimension
                    cutoff_h = img.shape[2] % (hyperparams["image_size"] - hyperparams["patch_sub"])
                    cutoff_w = img.shape[3] % (hyperparams["image_size"] - hyperparams["patch_sub"])
                    assert cutoff_h == cutoff_w
                    if cutoff_h != 0:
                        # remove border pixels s.t. image is divisible by patch size
                        img = img[:, :, :-cutoff_h, :-cutoff_w]
                        label = label[:, :-cutoff_h, :-cutoff_w]
                    img = rearrange(img, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=hyperparams["image_size"] - hyperparams["patch_sub"], p2=hyperparams["image_size"] - hyperparams["patch_sub"])
                    label = rearrange(label, 'b (h p1) (w p2) -> (b h w) p1 p2', p1=hyperparams["image_size"] - hyperparams["patch_sub"], p2=hyperparams["image_size"] - hyperparams["patch_sub"])
                else:
                    # train with one smaller random crop
                    x,y = torch.randint(0, 64 - hyperparams["image_size"] - hyperparams["patch_sub"], size=(2,))
                    img = img[:, :, x:x+hyperparams["image_size"]-hyperparams["patch_sub"], y:y+hyperparams["image_size"]-hyperparams["patch_sub"]] 
                    label = label[:, x:x+hyperparams["image_size"]-hyperparams["patch_sub"], y:y+hyperparams["image_size"]-hyperparams["patch_sub"]]


            if name == "li" or pixelwise:
                # baseline model only predicts class for the center pixel of the patch
                center_idx = (hyperparams["image_size"] - hyperparams["patch_sub"]) // 2
                if hyperparams["dataset"] in ["dfc","worldcover"]:
                    label = label[:, center_idx, center_idx]#.unsqueeze(1).unsqueeze(1)

                # extra dim for 3D conv model
                if name == "li": img = img.unsqueeze(1)

            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(img)
            loss = criterion(output, label) 

            if torch.isnan(loss):
                 ValueError("Loss is NaN")
                    
            pred = output.argmax(dim=1)
            valid_idx = label != hyperparams["ignored_label"]
            acc = (pred[valid_idx] == label[valid_idx]).sum() / pred[valid_idx].numel()
            if valid_idx.sum() != 0:
                macro_acc = acc_criterion(pred[valid_idx].to(int), label[valid_idx])
            else:
                macro_acc = acc

            loss.backward()
            optimizer.step()
            step += 1

            losses.append(loss.detach().item())
            accs.append(acc.detach().item())
            macro_accs.append(macro_acc.detach().item())

            if step % hyperparams["logging_freq"] == 0:
                wandb.log({
                        "epoch": epoch,
                        "acc": np.array(accs[-1*hyperparams["logging_freq"]:]).mean(),
                        "macro_acc": np.array(macro_accs[-1*hyperparams["logging_freq"]:]).mean(),
                        "loss": np.array(losses[-1*hyperparams["logging_freq"]:]).mean(),
                        "lr": optimizer.param_groups[0]['lr'],
                },
                step=step,
            )

        # log at end of training epoch (to same step as validation stats below)
        wandb.log({"epoch": epoch, "acc": acc.item(), "loss": loss.item()}, step=step)

        if epoch in validation_epochs:
            val_losses, best_val_acc = validate(
                hyperparams, 
                epoch, run_id, 
                name, model, 
                val_dataloader, 
                criterion, 
                acc_criterion, 
                step, 
                best_val_acc, 
                optimizer.param_groups[0]['lr'],
                pixelwise=pixelwise,
                )

        scheduler.step(torch.tensor(val_losses).mean().item())
        epoch += 1