#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A


class SlicedProstateDataset(Dataset):

    def __init__(self, root_dir: str, augment: bool = False):
        self.samples = []
        case_dirs = sorted([d for d in os.listdir(root_dir) if d.endswith("_slice")])
        for cd in case_dirs:
            d = os.path.join(root_dir, cd)
            npys = [f for f in os.listdir(d) if f.endswith('.npy')]
            base_to_files = {}
            for fname in npys:
                full = os.path.join(d, fname)
                if fname.endswith("_mask.npy"):
                    key = fname.replace("_mask.npy", "")
                    base_to_files.setdefault(key, {})['mask'] = full
                else:
                    key = fname.replace(".npy", "")
                    base_to_files.setdefault(key, {})['img'] = full
            for key, files in base_to_files.items():
                if 'img' in files and 'mask' in files:
                    self.samples.append((files['img'], files['mask']))
        if len(self.samples) == 0:
            print("Warning: no image/mask npy pairs found under", root_dir)

        self.augment = augment
        if self.augment:
            self.transform = A.Compose([
                A.Resize(height=270, width=270, p=1),
                A.ElasticTransform(alpha=150, sigma=100, p=0.5),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=20, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.3),
                A.pytorch.ToTensorV2(transpose_mask=True),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=270, width=270, p=1),
                A.pytorch.ToTensorV2(transpose_mask=True),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        imin, imax = float(np.min(img)), float(np.max(img))
        if imax > imin:
            img = (img - imin) / (imax - imin)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        if img.ndim == 2:
            img = img[..., None]

        if mask.ndim == 2:
            mask_ch0 = (mask == 1).astype(np.float32)
            mask_ch1 = (mask == 2).astype(np.float32)
            mask = np.stack([mask_ch0, mask_ch1], axis=-1)
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            m = mask[..., 0]
            mask_ch0 = (m == 1).astype(np.float32)
            mask_ch1 = (m == 2).astype(np.float32)
            mask = np.stack([mask_ch0, mask_ch1], axis=-1)
        elif mask.ndim == 3 and mask.shape[-1] == 2:
            mask = mask.astype(np.float32)
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        return img, mask


def run_epoch(model, loader, device, loss_fn, optimizer, sigmoid, train: bool, save_dir=None):
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    with torch.set_grad_enabled(train):
        for idx, (imgs, masks) in tqdm(enumerate(loader), leave=False, desc="Batch"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                probs = sigmoid(logits)
                preds = (probs > 0.5).float()

                if save_dir:
                    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                    axes[0, 0].imshow(imgs.cpu()[0, 0, :, :], cmap="gray")
                    axes[0, 0].imshow(masks.cpu()[0, 0, :, :], cmap="winter", vmin=0, vmax=1, alpha=0.15)
                    axes[0, 0].axis("off")
                    axes[0, 0].set_title("GT ch0")
                    axes[0, 1].imshow(imgs.cpu()[0, 0, :, :], cmap="gray")
                    axes[0, 1].imshow(preds.cpu()[0, 0, :, :], cmap="winter", vmin=0, vmax=1, alpha=0.15)
                    axes[0, 1].axis("off")
                    axes[0, 1].set_title("Pred ch0")
                    if masks.shape[1] > 1:
                        axes[1, 0].imshow(imgs.cpu()[0, 0, :, :], cmap="gray")
                        axes[1, 0].imshow(masks.cpu()[0, 1, :, :], cmap="winter", vmin=0, vmax=1, alpha=0.15)
                        axes[1, 0].axis("off")
                        axes[1, 0].set_title("GT ch1")
                        axes[1, 1].imshow(imgs.cpu()[0, 0, :, :], cmap="gray")
                        axes[1, 1].imshow(preds.cpu()[0, 1, :, :], cmap="winter", vmin=0, vmax=1, alpha=0.15)
                        axes[1, 1].axis("off")
                        axes[1, 1].set_title("Pred ch1")
                    fig.tight_layout()
                    fig.savefig(save_dir + f"_batch{idx}.png")
                    plt.close(fig)

                eps = 1e-7
                intersection = (probs * masks).sum(dim=(2, 3))
                denom = (probs + masks).sum(dim=(2, 3))
                dice = ((2 * intersection + eps) / (denom + eps)).mean()

            total_loss += loss.item()
            total_dice += float(dice)
            num_batches += 1

    return total_loss / max(1, num_batches), total_dice / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser(description="Train ResNet18-UNet for prostate segmentation")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Training data directory with sliced 2D data")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Validation data directory with sliced 2D data")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Log/output directory for checkpoints and TensorBoard")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs (default: 150)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers (default: 8)")
    args = parser.parse_args()

    print("Starting data loading...")
    train_ds = SlicedProstateDataset(args.train_dir, augment=True)
    val_ds = SlicedProstateDataset(args.val_dir, augment=False)
    print(f"Train slices: {len(train_ds)}")
    print(f"Val slices: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Initializing model...")
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=2,
    ).to(device)

    loss_fn = smp.losses.DiceLoss(mode="multilabel")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sigmoid = torch.nn.Sigmoid()

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_dice = 0.0
    print(f"Training for {args.epochs} epochs...")

    for epoch in tqdm(range(args.epochs), desc="Epochs", leave=False):
        train_loss, train_dice = run_epoch(
            model, train_loader, device, loss_fn, optimizer, sigmoid,
            train=True, save_dir=f"{args.log_dir}/train_images/epoch_{epoch}"
        )
        print(f"Train - loss: {train_loss:.4f} | dice: {train_dice:.4f}")

        val_loss, val_dice = run_epoch(
            model, val_loader, device, loss_fn, optimizer, sigmoid,
            train=False, save_dir=f"{args.log_dir}/val_images/epoch_{epoch}"
        )
        print(f"Val   - loss: {val_loss:.4f} | dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), f"{args.log_dir}/best_model.pth")

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Dice", train_dice, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Dice", val_dice, epoch)
        writer.flush()

    writer.close()
    print(f"Training complete. Best val dice: {best_val_dice:.4f}")
    print(f"Model saved to: {args.log_dir}/best_model.pth")


if __name__ == "__main__":
    main()
