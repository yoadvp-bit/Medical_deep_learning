"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser
from evaluation_metrics import ssim, nmse
import h5py

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.data.transforms import center_crop


from fastmri.pl_modules import FastMriDataModule, VarNetModule


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations,
    )
    # mask = create_mask_for_mask_type(
    #     12, args.center_fractions, args.accelerations,
    # )
    # use random masks for train transform, fixed masks for val transform
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform(mask_func=mask)
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
    test_set = data_module.test_dataloader().dataset
    print("Size of trainingset:", len(test_set))


    # ------------
    # model
    # ------------

    model = VarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)
    return


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("save_model/fastmri_dirs.yaml")
    #path_config = pathlib.Path("/content/gdrive/MyDrive/DL_4_MI/Assigment3/Recon_exercise_2024/save_model/fastmri_dirs.yaml")
    num_gpus = 1
    batch_size = 1

    # set defaults based on optional directory config
    #data_path = "/content/gdrive/MyDrive/DL_4_MI/Assigment3/Recon_exercise_2024/FastMRIdata/"
    data_path = '/projects/prjs1312/Recon_exercise/FastMRIdata/multicoil_test'
    default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"

    # client arguments
    parser.add_argument(
        "--mode",
        default="test",
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
        default=[1.5],
        type=int,
        help="Acceleration rates to use for masks",
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
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        # mask_type="equispaced_fraction",  # VarNet uses equispaced mask
        mask_type="random",  # VarNet uses equispaced mask
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,
    )

    # module config
    parser = VarNetModule.add_model_specific_args(parser)
    args = parser.parse_args()
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
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        # strategy=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])
    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TESTING
    # ---------------------
    cli_main(args)


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

# fourier transform copied to prevent importing error

def fourier_transform(kspace):
    """This function reconstrucs an image using the fourier transform. """
    dim1 = 0
    dim2 = 1
    # dofftshift and ifftshift because data is spread over four corners
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(dim1, dim2)),
                    axes=(dim1, dim2)), axes=(dim1, dim2))
    return image

def evaluate_test_data_quantitatively(data_dir, recon_dir):
    gt_files = sorted(pathlib.Path(data_dir).glob('*.h5'))
    rec_files = sorted(pathlib.Path(recon_dir).glob('*.h5'))

    mse_vals = []
    nmse_vals = []
    psnr_vals = []
    ssim_vals = []
    
    for gt_path, rec_path in zip(gt_files, rec_files):
        with h5py.File(gt_path, 'r') as f:
            gt_img = f['/kspace'][:]
        with h5py.File(rec_path, 'r') as f:
            rec_img = f['/reconstruction'][:]
        
        gt_img = np.squeeze(gt_img, axis=1)
        gt_img = center_crop(gt_img, rec_img.shape[1:])
        
        gt_img = fourier_transform(gt_img)
        rec_img = fourier_transform(rec_img)
        
        if np.iscomplexobj(gt_img):
            gt_img = np.abs(gt_img)
        if np.iscomplexobj(rec_img):
            rec_img = np.abs(rec_img)
        
        mse_vals.append(mse(gt_img, rec_img))
        nmse_vals.append(nmse(gt_img, rec_img))
        psnr_vals.append(psnr(gt_img, rec_img))
        ssim_vals.append(ssim(gt_img, rec_img))
    
    print(f"Mean MSE: {np.mean(mse_vals)}")
    print(f"Mean NMSE: {np.mean(nmse_vals)}")
    print(f"Mean PSNR: {np.mean(psnr_vals)}")
    print(f"Mean SSIM: {np.mean(ssim_vals)}")

    return

def store_img(fig, output_path: str):
    """ Save the figure as a PNG file to the specified path. """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  
    fig.savefig(output_path, bbox_inches='tight', dpi=300)  
    plt.close(fig)  


def evaluate_test_data_qualitatively(data_dir, rec_dir, save_dir):
    gt_list = sorted(pathlib.Path(data_dir).glob('*.h5'))
    rec_list = sorted(pathlib.Path(rec_dir).glob('*.h5'))

    for gt_file, rec_file in zip(gt_list, rec_list):
        print(f"Processing: {gt_file.name} and {rec_file.name}")
        with h5py.File(gt_file, 'r') as file:
            kspace_data = file['/kspace'][:]
        with h5py.File(rec_file, 'r') as file:
            rec_data = file['/reconstruction'][:]

        kspace_data = np.squeeze(kspace_data, axis=1)
        kspace_data = center_crop(kspace_data, rec_data.shape[1:])
        gt_img = fourier_transform(kspace_data)

        mag_gt = np.abs(gt_img)
        mag_rec = np.abs(rec_data)

        mid_slice = mag_gt.shape[0] // 2

        slice_gt_mag = mag_gt[mid_slice]
        slice_rec_mag = mag_rec[mid_slice]

        slice_gt_phase = np.angle(gt_img[mid_slice])
        slice_gt_real = np.real(gt_img[mid_slice])
        slice_gt_imag = np.imag(gt_img[mid_slice])

        slice_rec_phase = np.angle(rec_data[mid_slice])
        slice_rec_real = np.real(rec_data[mid_slice])
        slice_rec_imag = np.imag(rec_data[mid_slice])

        fig, axs = plt.subplots(2, 4, figsize=(15, 8))

        axs[0, 0].imshow(slice_gt_mag, cmap='gray')
        axs[0, 0].set_title('GT Magnitude')
        axs[0, 1].imshow(slice_gt_phase, cmap='gray')
        axs[0, 1].set_title('GT Phase')
        axs[0, 2].imshow(slice_gt_real, cmap='gray')
        axs[0, 2].set_title('GT Real')
        axs[0, 3].imshow(slice_gt_imag, cmap='gray')
        axs[0, 3].set_title('GT Imaginary')

        axs[1, 0].imshow(slice_rec_mag, cmap='gray')
        axs[1, 0].set_title('Rec Magnitude')
        axs[1, 1].imshow(slice_rec_phase, cmap='gray')
        axs[1, 1].set_title('Rec Phase')
        axs[1, 2].imshow(slice_rec_real, cmap='gray')
        axs[1, 2].set_title('Rec Real')
        axs[1, 3].imshow(slice_rec_imag, cmap='gray')
        axs[1, 3].set_title('Rec Imaginary')

        out_file = os.path.join(save_dir, f"{gt_file.stem}_comparison.png")
        store_img(fig, out_file)
        print(f"Saved: {out_file}")

    return

if __name__ == "__main__":
    # run testing the network
    run_cli()
    datapath = '/projects/0/gpuuva035/reconstruction/'
    reconpath = 'varnet/varnet_demo/reconstructions/'
    # # quantitativaly evaluate data
    evaluate_test_data_quantitatively(datapath, reconpath)
    # # qualitatively
    evaluate_test_data_qualitatively(datapath, reconpath)


