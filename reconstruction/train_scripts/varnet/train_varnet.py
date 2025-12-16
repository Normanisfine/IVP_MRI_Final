"""
Custom VarNet training script for your MRI dataset
Modified from the original FastMRI VarNet demo
Compatible with modern PyTorch Lightning versions
"""

import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, VarNetModule


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
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
        distributed_sampler=(args.strategy in ("ddp", "ddp_cpu")),
    )

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
    # callbacks
    # ------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.default_root_dir / "checkpoints",
        filename="{epoch}-{validation_loss:.4f}",
        monitor="validation_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="validation_loss",
        patience=10,
        mode="min",
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ------------
    # logger
    # ------------
    logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        name="lightning_logs",
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        strategy=args.strategy,
        devices=args.num_gpus,
        accelerator="gpu" if args.num_gpus > 0 else "cpu",
        precision=16 if args.use_amp else 32,
        log_every_n_steps=50,
        enable_progress_bar=True,
        gradient_clip_val=args.gradient_clip_val,
    )

    # ------------
    # run
    # ------------
    trainer.fit(model, datamodule=data_module)


def build_args():
    parser = ArgumentParser()

    # data arguments
    parser.add_argument(
        "--data_path",
        default="/scratch/ml8347/MRI/train/train_dataset",
        type=pathlib.Path,
        help="Path to FastMRI data root",
    )
    parser.add_argument(
        "--test_path",
        default=None,
        type=pathlib.Path,
        help="Path to test data (if different from data_path)",
    )
    parser.add_argument(
        "--challenge",
        default="multicoil",
        choices=("singlecoil", "multicoil"),
        type=str,
        help="Which challenge to run",
    )
    parser.add_argument(
        "--test_split",
        default="test",
        choices=("val", "test"),
        type=str,
        help="Which split to use for testing",
    )
    parser.add_argument(
        "--sample_rate",
        default=1.0,
        type=float,
        help="Fraction of total volumes to use",
    )

    # mask arguments
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="equispaced_fraction",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Center fraction for mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration factors for mask",
    )

    # model arguments
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
        help="Number of channels in VarNet blocks",
    )
    parser.add_argument(
        "--sens_pools",
        default=4,
        type=int,
        help="Number of pooling layers for sense map estimation U-Net",
    )
    parser.add_argument(
        "--sens_chans",
        default=8,
        type=int,
        help="Number of channels for sense map estimation U-Net",
    )

    # training arguments
    parser.add_argument(
        "--lr",
        default=0.0003,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--lr_step_size",
        default=40,
        type=int,
        help="Epoch at which to decrease learning rate",
    )
    parser.add_argument(
        "--lr_gamma",
        default=0.1,
        type=float,
        help="Amount to decrease learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--max_epochs",
        default=50,
        type=int,
        help="Number of training epochs",
    )

    # trainer arguments
    parser.add_argument(
        "--default_root_dir",
        default=pathlib.Path("./varnet_training"),
        type=pathlib.Path,
        help="Default directory for logs and checkpoints",
    )
    parser.add_argument(
        "--num_gpus",
        default=1,
        type=int,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        type=str,
        help="Distributed training strategy (ddp, ddp_cpu, etc.)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision (16-bit training)",
    )
    parser.add_argument(
        "--gradient_clip_val",
        default=0.0,
        type=float,
        help="Gradient clipping value (0 for no clipping)",
    )

    # reproducibility
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = build_args()
    cli_main(args)
