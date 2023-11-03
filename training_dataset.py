import os

import torch
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset, DataLoader
"""
- Load all datasets into one huge list
- iterate over this list as method of getitem
Pro:
    faster access to data (already in ram)
    no awkwardly large getitem method
    no multilist (dict of list of lists) as item but a simple dict
Cons:
    needs more ram 
    longer setup time (init should load stuff into RAM)
"""

class TrainingDataset(Dataset):
    def __init__(self, settings, split, transform=None, norm=None):
        self.settings = settings

        # Get paths
        nii_path = settings["dataset"]["raw_path"]
        gt_path = settings["dataset"]["gt_path"]

        # create list
        nii_list = []
        gt_list = []

        name_list = []

        # Load data
        for item in split:
            item_nii_path   = os.path.join(nii_path, item)
            item_gt_path    = os.path.join(gt_path, item)

            image       = np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1)
            image_gt    = np.swapaxes(nib.load(item_gt_path).dataobj, 0, 1).astype(np.int64)

            image = image.astype(np.int64)

            image = image[:128,:128,:128]
            image_gt = image_gt[:128,:128,:128]

            if transform:
                image       = transform(image)
                image_gt    = transform(image_gt)

            if norm:
                image       = norm(image)
                # image_gt    = norm(image_gt)
            

            # Pad
            #TODO
            # padding_value = int(self.settings["preprocessing"]["padding"])

            # image = np.pad(image, padding_value, "reflect")
            
            image_gt[image_gt == np.min(image_gt)] = 0
            image_gt[image_gt > np.min(image_gt)] = 1
            # Torchify
            image = torch.tensor(image).float().unsqueeze(0)
            # image_gt = torch.tensor(image_gt).float().unsqueeze(0)
            image_gt = torch.tensor(image_gt).float().unsqueeze(0)
            nii_list.append(image)
            gt_list.append(image_gt)
            name_list.append(item)

        self.item_list = [nii_list, gt_list, name_list]

    def __len__(self):
        return len(self.item_list[0])

    def __getitem__(self, idx):
        return {"volume":self.item_list[0][idx], "segmentation":self.item_list[1][idx], "name":self.item_list[2][idx]}



