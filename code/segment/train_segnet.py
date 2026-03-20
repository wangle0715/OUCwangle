import argparse
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset
from evaluate import dice_loss, evaluate
from SegNet import SegNet

# 数据路径配置
dir_img = Path('/media/disk1/wl/Unet_tactile/data/imgs')
dir_mask = Path('/media/disk1/wl/Unet_tactile/data/masks')
dir_checkpoint = Path('/media/disk1/wl/Unet_tactile/checkpoints')


def evaluate_test(net, loader, device):
    net.eval()
    total_pixels = 0
    correct_pixels = 0
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)

            pred = net(images)
            probs = torch.sigmoid(pred[:, 1, ...])  # 取第2类
            preds = (probs > 0.5).float()

            # Pixel Accuracy
            correct_pixels += (preds == true_masks).sum().item()
            total_pixels += torch.numel(preds)

            # Dice
            intersection = (preds * true_masks).sum(dim=(1, 2))
            union = preds.sum(dim=(1, 2)) + true_masks.sum(dim=(1, 2))
            dice = (2. * intersection + 1e-7) / (union + 1e-7)
            dice_scores.extend(dice.cpu().numpy())

            # IoU
            i = (preds * true_masks).sum(dim=(1, 2))
            u = (preds + true_masks - preds * true_masks).sum(dim=(1, 2))
            iou = (i + 1e-7) / (u + 1e-7)
            iou_scores.extend(iou.cpu().numpy())

    PA = correct_pixels / total_pixels
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)

    return PA, mean_dice, mean_iou


def train_net(net, device, epochs: int = 10, batch_size: int = 1, learning_rate: float = 1e-05,
              val_percent: float = 0.1, test_percent: float = 0.1,
              save_checkpoint: bool = True, img_scale: float = 0.5, amp: bool = False):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation / test
    n_total = len(dataset)
    n_val = int(n_total * val_percent)
    n_test = int(n_total * test_percent)
    n_train = n_total - n_val - n_test

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test],
                                                generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)

    # Logging
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Test size:       {n_test}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Image scale:     {img_scale}
        Mixed precision: {amp}
    ''')

    # 4. Optimizer & loss
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0
    epoch_losses = []
    validation_dice_scores = []

    # 5. Training loop
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred[:, 1, ...], true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if (n_train // (10 * batch_size)) > 0 and global_step % (n_train // (10 * batch_size)) == 0:
                    val_score = evaluate(net, val_loader, device)
                    validation_dice_scores.append(val_score)
                    scheduler.step(val_score)
                    logging.info('Validation Dice score: {:.4f}'.format(val_score))

            epoch_losses.append(epoch_loss / len(train_loader))

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

    # 6. Test after training
    PA, dice, miou = evaluate_test(net, test_loader, device)
    logging.info(f'Test Results:\nPixel Accuracy: {PA:.4f}\nDice Score: {dice:.4f}\nmIoU: {miou:.4f}')

    # 7. Draw training curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Train Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    validation_dice_scores_cpu = [score.cpu().item() for score in validation_dice_scores]
    plt.plot(range(1, len(validation_dice_scores_cpu) + 1), validation_dice_scores_cpu, label='Validation Dice', marker='o')
    plt.xlabel('Validation Step')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice Curve')
    plt.grid(True)
    plt.legend()

    plt.savefig('training_validation_curves.png')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10)
    parser.add_argument('--batch-size', '-b', metavar='B', type=int, default=16)
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=2e-5, dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5)
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--bilinear', action='store_true', default=False)
    parser.add_argument('--classes', '-c', type=int, default=2)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = SegNet(num_classes=args.classes)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  device=device,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  val_percent=args.val / 100,
                  test_percent=0.1,
                  img_scale=args.scale,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
