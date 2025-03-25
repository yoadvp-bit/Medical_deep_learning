"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser
import pytorch_lightning as pl

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, MaskyVarNetModule
import time

import wandb
from pytorch_lightning.loggers import WandbLogger

import matplotlib.pyplot as plt

def cli_main(args):
    pl.seed_everything(args.seed)

    # Wandb
    logger = WandbLogger(name=args.experiment_name, project='Exercise3')

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, [1]
    )

    # use random masks for train transform, fixed masks for val transform
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform()
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # print how large the training/validation set is
    train_set = data_module.train_dataloader().dataset
    print("Size of trainingset:", len(train_set))
    val_set = data_module.val_dataloader().dataset
    print("Size of validationset:", len(val_set))

    # ------------
    # model
    # ------------
    model = MaskyVarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        mask_slope=args.mask_slope,
        mask_sparsity=1/args.accelerations[0],
        mask_lr=args.mask_lr
    )
    # model = model.double()
    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        print("Start training")
        start = time.time()
        trainer.fit(model, datamodule=data_module)
        end = time.time()
        print('Training time:', end-start)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("save_model/fastmri_dirs.yaml")
    # HERE VERANDERD TO DO
    # path_config = pathlib.Path("/content/gdrive/MyDrive/DL_4_MI/Assigment3/Recon_exercise_2024/save_model/fastmri_dirs.yaml")
    num_gpus = 1
    batch_size = 1

    # set path to logs and saved model
    default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"
    data_path = "/gpfs/work5/0/prjs1312/Recon_exercise/FastMRIdata/"
    # data_path = "C:/Users/jonas/Desktop/Mri_data/"

    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )

    # how much of the center to keep from subsampling
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[2],
        type=int,
        help="Acceleration rates to use for masks",
    )

    parser.add_argument(
        "--experiment_name",
        default='Train_VarNet 2 cascades',
        type=str,
        help="Name of Experiment in WandB",
    )

    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--num_epochs",
        default=10,
        type=int,
        help="Number of epochs to train",
    )

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)

    # module config
    parser = MaskyVarNetModule.add_model_specific_args(parser)

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()

    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type=args.mask_type,
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
    )

    parser.set_defaults(
        num_cascades=2,  # number of unrolled iterations
        pools=4,  # number of pooling layers for U-Net
        chans=18,  # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=args.learning_rate,  # Adam learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
        mask_slope=3.0,
        mask_lr=0.001,
        mask_sparsity=1/args.accelerations[0],
        thresh_slope=0.1
    )

    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=args.num_epochs,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = args.default_root_dir / f"checkpoints_{timestamp}"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    # if args.resume_from_checkpoint is None:
    #     ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
    #     if ckpt_list:
    #         args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()
    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    wandb.login()
    run_cli()
