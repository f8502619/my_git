'''
This script is used to:
(1) visualise the images and the masks.
(2) train the model.
(3) test the model.
'''

import glob
from random import shuffle, random

# Importing the required libraries
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
from utils import EarlyStopping, save_pred
from loader import DataFolder
from sklearn.model_selection import train_test_split
from model import UNet
import torch.nn.functional as F
import torch
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torchsummary
import os
import pdb
import sys
import random

# create the paths for the images and the masks
train_paths = sorted(glob.glob('dataset/train_images_256/*'))
mask_paths = sorted(glob.glob('dataset/train_masks_256/*'))
print("total number of images", len(train_paths))
print("total number of masks", len(mask_paths))

if __name__ == '__main__':
    # Part1: Visualise the images and the masks
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate.')
    parser.add_argument('--min_delta', type=float, default=0.0001,
                        help='Minimum loss improvement for each epoch.')
    parser.add_argument('--patience', type=float, default=10,
                        help='Early stopping patience.')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='Training batch size.')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='Evaluation batch size. Same as validation batch size.')
    parser.add_argument('--epochs', default=1000, type=int, help='Maximum number of epochs for training')
    parser.add_argument('--exp_root', type=str, default='./dataset/experiments/',
                        help='Path to the experiment root.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--eval', default=1, type=int, help='Evaluation mode')
    args = parser.parse_args()

    # Training the model -  Create the data loaders
    all_loader = data.DataLoader(
        dataset=DataFolder('dataset/train_images_256/', 'dataset/train_masks_256/', 'all'),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=2
    )

    # Training the model - Convert the data to tensors
    GPU = torch.cuda.is_available()
    device = torch.device("cuda" if GPU else "cpu")
    model = UNet(3, shrink=1).cuda()
    nets = [model]
    params = [{'params': net.parameters()} for net in nets]
    # use Adam optimizer
    optimiser = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    es = EarlyStopping(min_delta=args.min_delta, patience=args.patience)
    random_seed = 42

    all_dataset = DataFolder('dataset/train_images_256/', 'dataset/train_masks_256/', 'all')
    num_all = len(all_dataset)
    indices = list(range(num_all))
    split = int(np.floor(0.2 * num_all))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = data.DataLoader(
        dataset=all_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=2
    )

    valid_loader = data.DataLoader(
        dataset=all_dataset,
        batch_size=args.eval_batch_size,
        sampler=valid_sampler,
        num_workers=2
    )

    # Creating a folder for the experiment
    exp_dir = os.path.join(args.exp_root, 'exp_{}'.format(args.seed))
    os.makedirs(exp_dir, exist_ok=True)

    # Creating a folder for the model
    model_dir = os.path.join(exp_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    # Setting the folder of saving the predictions
    pred_dir = os.path.join(model_dir, 'pred')
    os.makedirs(pred_dir, exist_ok=True)

    if args.eval == 0:
        # Training the model -Training the model
        for epoch in range(1, args.epochs + 1):

            train_loss = []
            valid_loss = []
            for batch_idx, (img, mask, _) in enumerate(train_loader):
                optimiser.zero_grad()

                img = img.cuda()
                mask = mask.cuda()

                pred = model(img)
                loss = criterion(pred, mask)

                loss.backward()
                optimiser.step()

                train_loss.append(loss.item())

            with torch.no_grad():
                for batch_idx, (img, mask, _) in enumerate(valid_loader):
                    img = img.cuda()
                    mask = mask.cuda()

                    pred = model(img)
                    loss = criterion(pred, mask)

                    valid_loss.append(loss.item())

            print('[EPOCH {}/{}] Train Loss: {:.4f}; Valid Loss: {:.4f}'.format(
                epoch, args.epochs, np.mean(train_loss), np.mean(valid_loss)
            ))

            flag, best, bad_epochs = es.step(torch.Tensor([np.mean(valid_loss)]))
            if flag:
                print('Early stopping criterion met')
                break
            else:
                if bad_epochs == 0:
                    torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
                    print('Saving current best model')

                print('Current Valid loss: {:.4f}; Current best: {:.4f}; Bad epochs: {}'.format(
                    np.mean(valid_loss), best.item(), bad_epochs
                ))

        print('Training done... start evaluation')

    # Training the model - Evaluation
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))
    model.eval()

    # save the predictions
    with torch.no_grad():
        all_loss = []
        for batch_idx, (img, mask, img_fns) in enumerate(all_loader):
            img = img.cuda()
            mask = mask.cuda()

            pred = model(img)
            loss = criterion(pred, mask)

            all_loss.append(loss.item())

            pred_mask = torch.argmax(F.softmax(pred, dim=1), dim=1)
            pred_mask = torch.chunk(pred_mask, chunks=args.eval_batch_size, dim=0)

            save_pred(pred_mask, img_fns, pred_dir)

            print('[PREDICT {}/{}] Loss: {:.4f}'.format(
                batch_idx + 1, len(all_loader), loss.item()
            ))

    print('FINAL PREDICT LOSS: {:.4f}'.format(np.mean(all_loss)))
