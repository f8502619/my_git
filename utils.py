'''
This script is used to create utility functions for the project.
part 1: functions to pre-process the images and the masks, then create the dataset
part 2: functions to
'''
# Importing the required libraries
import os
import torch


#  Functions to save the predictions and early stop the training
def save_pred(preds, fns, out_dir):
    if not os.path.exists(out_dir):
        print('Creating output directory: {}'.format(out_dir))
        os.makedirs(out_dir)

    for idx, pred in enumerate(preds):
        import matplotlib.pyplot as plt
        pred = torch.squeeze(pred, dim=0)
        sp = pred.cpu().numpy()
        plt.figure()
        plt.imshow(sp, cmap='Greys_r')
        plt.savefig(out_dir + '/{}'.format(fns[idx]), dpi=300)
        plt.close()


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.patience == 0:
            return False, self.best, self.num_bad_epochs

        if self.best is None:
            self.best = metrics
            return False, self.best, 1

        if torch.isnan(metrics):
            return True, self.best, self.num_bad_epochs

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True, self.best, self.num_bad_epochs

        return False, self.best, self.num_bad_epochs

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)
