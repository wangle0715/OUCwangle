import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


def dice_loss(input: Tensor, target: Tensor, epsilon=1e-6):

    inter = torch.sum(input * target)
    sets_sum = torch.sum(input) + torch.sum(target)

    return 1 - (2 * inter + epsilon) / (sets_sum + epsilon)


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to binary predictions (foreground vs background)
            binary_pred = (F.sigmoid(mask_pred[:, 1, ...]) > 0.5).float()
            binary_true = (mask_true > 0).float()

            # compute the Dice score
            dice_score += 1 - dice_loss(binary_pred, binary_true, epsilon=1e-6)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches

