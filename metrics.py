import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

class BinaryDiceLoss(nn.Module):
	def __init__(self):
		super(BinaryDiceLoss, self).__init__()
	
	def forward(self, input, targets):
		N = targets.size()[0]
		smooth = 1
		input_flat = input.view(N, -1)
		targets_flat = targets.view(N, -1)
	
		intersection = input_flat * targets_flat 
		N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
		loss = 1 - N_dice_eff.sum() / N
		return loss

def bce_dice_loss(y_true, y_pred):
  bce = F.binary_cross_entropy(torch.sigmoid(y_pred),y_true)
  dice_loss = BinaryDiceLoss()
  dice_loss = dice_loss(y_pred,y_true)
  loss = bce + dice_loss
  return loss


def get_iou(y_true, y_pred):
    """Calculates the intersection over union for the images"""

    y_true = np.asarray(y_true).astype(np.bool)
    y_pred = np.asarray(y_pred).astype(np.bool)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    y_true = y_true > 0.5
    y_pred = y_pred > 0.5

    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)

    return intersection.sum() / float(union.sum())


def iou_score(y_true, y_pred):
    return get_iou(y_true, y_pred)
