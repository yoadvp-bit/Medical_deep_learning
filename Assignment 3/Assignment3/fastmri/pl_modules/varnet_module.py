"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import torch
import wandb

import fastmri
from fastmri.data import transforms
from fastmri.models import VarNet
from fastmri.models import ProbMask, RescaleProbMap, ThresholdRandomMask, RandomMask, UnderSample

from .mri_module import MriModule


class VarNetModule(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.varnet = VarNet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
        )

        self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, mask, num_low_frequencies):
        return self.varnet(masked_kspace, mask, num_low_frequencies)

    def training_step(self, batch, batch_idx):
        output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies)

        target, output = transforms.center_crop_to_smallest(batch.target, output)
        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(
            batch.masked_kspace, batch.mask, batch.num_low_frequencies
        )
        target, output = transforms.center_crop_to_smallest(batch.target, output)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            ),
        }

    def test_step(self, batch, batch_idx):
        output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies)

        # check for FLAIR 203
        if output.shape[-1] < batch.crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        else:
            crop_size = batch.crop_size

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser


class MaskyVarNetModule(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        mask_slope: float = 3,
        mask_sparsity: float = 0.5,
        thresh_slope: float = 4,
        mask_lr: float = 0.001,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.mask_lr = mask_lr
        self.mask_slope = mask_slope
        self.mask_sparsity = mask_sparsity
        self.thresh_slope = thresh_slope

        self.varnet = VarNet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
        )

        self.loss = fastmri.SSIMLoss()

        # Mask generation components
        self.prob_mask = ProbMask(slope=self.mask_slope, shape=(1,1,184,160,1))
        self.threshold_random_mask = ThresholdRandomMask(slope=self.thresh_slope)
        self.random_mask = RandomMask()
        self.under_sample = UnderSample()
        self.rescale_prob_map = RescaleProbMap(sparsity=self.mask_sparsity)


    def forward(self, kspace, num_low_frequencies):
        # Generate the mask
        prob_map = self.prob_mask(kspace)
        rescaled_prob_map = self.rescale_prob_map(prob_map)
        random_mask = self.random_mask(prob_map)
        thresholded_mask = self.threshold_random_mask(rescaled_prob_map, random_mask)
        undersampled_kspace = self.under_sample(kspace, thresholded_mask)

        # Log the histogram of the thresholded mask values
        self.logger.experiment.log({
            "thresholded_mask_histogram": wandb.Histogram(thresholded_mask.cpu().detach().numpy())
        })

        # print("undersampled kspace: ", undersampled_kspace.shape, " thresholded mask: ", thresholded_mask.shape, " prob_map: ", prob_map.shape, " rescaled_prob_map: ", rescaled_prob_map.shape, " random_mask: ", random_mask.shape)

        # # Log the thresholded mask as an image
        # self.logger.experiment.log({
        #     "thresholded_mask": wandb.Image(thresholded_mask[0,0,:,:,0].cpu().detach().numpy())
        # })  
        
        # # Log the raw mask as an image
        # self.logger.experiment.log({
        #     "prob_mask": wandb.Image(prob_map[0,0,:,:,0].cpu().detach().numpy())
        # })

        # Log the raw mask as an image
        self.logger.experiment.log({
            "rescaled_prob_mask": wandb.Image(rescaled_prob_map[0,0,:,:,0].cpu().detach().numpy())
        })

        # # Log the random mask as an image
        # self.logger.experiment.log({
        #     "random_mask": wandb.Image(random_mask[0,0,:,:,0].cpu().detach().numpy())
        # })

        # # Log the thresholded mask as an image
        # self.logger.experiment.log({
        #     "thresholded_mask": wandb.Image(thresholded_mask[0,0,:,:,0].cpu().detach().numpy())
        # })

        # Apply VarNet
        return self.varnet(undersampled_kspace, thresholded_mask, num_low_frequencies)


    def training_step(self, batch, batch_idx):
        output = self(batch.masked_kspace, batch.num_low_frequencies)

        target, output = transforms.center_crop_to_smallest(batch.target, output)
        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )

        self.log("train_loss", loss)

        # # Perform backward pass to compute gradients
        # self.manual_backward(loss)

        # Log gradients for the mask parameters
        if self.prob_mask.slope.grad is not None:
            self.logger.experiment.log({"prob_mask.slope_grad": wandb.Histogram(self.prob_mask.slope.grad.cpu().numpy())})
        if hasattr(self.prob_mask, 'mult') and self.prob_mask.mult.grad is not None:
            self.logger.experiment.log({"prob_mask.mult_grad": wandb.Histogram(self.prob_mask.mult.grad.cpu().numpy())})

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(
            batch.masked_kspace, batch.num_low_frequencies
        )
        target, output = transforms.center_crop_to_smallest(batch.target, output)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            ),
        }

    def test_step(self, batch, batch_idx):
        output = self(batch.masked_kspace, batch.num_low_frequencies)

        # check for FLAIR 203
        if output.shape[-1] < batch.crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        else:
            crop_size = batch.crop_size

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        print("Named parameters: ", [name for name, param in self.named_parameters()])

        mask_param_names = [
            name for name, param in self.named_parameters()
            if any(param is p for p in self.prob_mask.parameters()) or
               any(param is p for p in self.threshold_random_mask.parameters()) or
               any(param is p for p in self.random_mask.parameters()) or
               any(param is p for p in self.under_sample.parameters()) or
               any(param is p for p in self.rescale_prob_map.parameters())
        ]
        other_param_names = [
            name for name, param in self.named_parameters()
            if name not in mask_param_names
        ]
        print("Mask param names:", mask_param_names)
        print("Other param names:", other_param_names)
        overlap = set(mask_param_names).intersection(other_param_names)
        print("Overlap:", overlap)

        mask_params = [
            param for name, param in self.named_parameters()
            if name in mask_param_names
        ]
        other_params = [
            param for name, param in self.named_parameters()
            if name in other_param_names
        ]

        optimizer = torch.optim.Adam(
            [
                {'params': other_params, 'lr': self.lr},
                {'params': mask_params, 'lr': self.mask_lr}
            ],
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.lr_step_size, self.lr_gamma
        )

        return [optimizer], [scheduler]

    def configure_optimizers2(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        has_mult = any("mult" in name for name, _ in self.named_parameters())
        print(f"Does self.parameters include 'self.mult'? {'Yes' if has_mult else 'No'}")

        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        print("Arguments in parser:")
        for arg in vars(parser.parse_args()):
            print(f"{arg}: {getattr(parser.parse_args(), arg)}")

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        # mask training params
        parser.add_argument(
            "--mask_slope",
            default=3,
            type=float,
            help="Slope for the mask training",
        )
        parser.add_argument(
            "--thresh_slope",
            default=4,
            type=float,
            help="Slope for the thresholding of the mask",
        )
        # parser.add_argument(
        #     "--mask_sparsity",
        #     default=0.5,
        #     type=float,
        #     help="Sparsity for the mask training",
        # )
        parser.add_argument(
            "--mask_lr",
            default=0.001,
            type=float,
            help="Learning rate for the mask training",
        )

        print("Arguments in parser:")
        for arg in vars(parser.parse_args()):
            print(f"{arg}: {getattr(parser.parse_args(), arg)}")

        return parser
