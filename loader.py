import os
import os.path
import glob
import random
import torch
from torch.utils.data import Dataset
import cv2


class DataFolder(Dataset):
    def __init__(self, path_img, path_mask, mode):
        self.mask_fns = None
        self.img_fns = None
        self.path_img = path_img
        self.path_mask = path_mask
        self.mode = mode
        self.get_data_list()

    def get_data_list(self):
        self.img_fns = []
        self.mask_fns = []
        for fn in glob.iglob(self.path_img + '*tif'):
            img_id = os.path.basename(fn)
            self.img_fns.append(img_id)

        for fn in glob.iglob(self.path_mask + '*tif'):
            mask_id = os.path.basename(fn)
            self.mask_fns.append(mask_id)

    def __getitem__(self, index):

        img_id = self.img_fns[index]
        img = cv2.cvtColor(cv2.imread(self.path_img + img_id), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.path_mask + img_id), cv2.COLOR_BGR2GRAY)
        h, w = mask.shape
        w = random.randint(0, w-256)
        h = random.randint(0, h-256)
        image = img[h:h+256, w:w+256]
        mask = mask[h:h + 256, w:w + 256]
        img = torch.from_numpy(img).float().permute(2, 0, 1)/255
        mask = torch.from_numpy(mask).type(torch.LongTensor)

        return img, mask, img_id

    def __len__(self):
        return len(self.img_fns)
