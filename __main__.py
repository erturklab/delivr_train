# import neccessary stuff
import os
import shutil
import json
import datetime
import torch
import torchmetrics

from collections import OrderedDict

from lightning.pytorch.loggers import TensorBoardLogger

import argparse

import numpy as np
import nibabel as nib
import lightning as L

import pandas as pd

from random import shuffle
from monai.data import CacheDataset, ArrayDataset
from monai.data import DataLoader
from lightning_unet import LightningUnet
from training_dataset import TrainingDataset

import torch.nn.functional as F
from cell_analysis import get_patch_overlap

from monai.transforms import (
    Compose,
    LoadImaged, 
    ToTensord,
    Orientationd,
    ThresholdIntensityd,
    AsDiscrete,
    RandGaussianNoised,
    RandFlipd,
    RandAffined,
    CastToTyped
)

import monai

class Binarized(monai.transforms.Transform):
    def __init__(self, keys, threshold=0.5):
        self.threshold = threshold
        self.keys = keys[0]

    def __call__(self, data):
        img = data[self.keys]
        img[img >= self.threshold] = self.threshold
        img[img < self.threshold] = 0
        data[self.keys] = img
        return data

# Sneeds Seed & Feed
torch.manual_seed(42)

def get_scores(tp, fp, fn):
    eps = 0.00001
    dice        = tp / (eps + tp + 0.5*(fp + fn))
    precision   = tp / (eps + tp + fp) if (tp + fp) > 0 else 0
    recall      = tp / (eps + tp + fn)
    return dice, precision, recall

def create_overlay(vol_pred, vol_gt):
    vol_gt[vol_gt > 1] = 1 
    vol_pred[vol_pred > 1] = 1 

    vol_out = vol_gt*2 + vol_pred
    vol_out[vol_out==2] = 10
    vol_out[vol_out==3] = 2 
    vol_out[vol_out==10]= 3
    return vol_out

def save_nifti(path, volume):
    # Create a new NIfTI image from the output data
    affmat = np.eye(4)
    affmat[0,0] = affmat[1,1] = -1
    NiftiObject = nib.Nifti1Image(np.swapaxes(volume,0,1), affine=affmat)
    nib.save(NiftiObject, path)

def test_instance(checkpoint_path, test_dataset, settings):
    """Perform a test round on a model and calculate the voxel scores
    """
    model = LightningUnet.load_from_checkpoint(checkpoint_path)
    tp, fp, fn = 0, 0, 0
    batch_size = settings["network"]["batch_size"]
    num_workers = settings["network"]["num_workers"]
    if num_workers == -1:
        num_workers = os.cpu_count()
    
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers)

    model.eval()

    for i, data in enumerate(test_dataset):
        with torch.no_grad():
            print(f"{i}/{len(test_dataset)}")
            logits = model(data['volume'].unsqueeze(0)) # Assumed model expects single image tensor
            
            # Assuming the model output is a tensor, detach it from the computation graph and convert to numpy
            inference = logits.detach().cpu().numpy()
            inference[inference > 0.5] = 1
            inference[inference <= 0.5] = 0
            inference = np.squeeze(np.squeeze(inference.astype(np.uint8)))

            # Load the segmentation for instance scores
            segmentation = np.squeeze(np.squeeze(data["segmentation"].cpu().numpy().astype(np.uint8)))

            # Calculate instance scores
            tp_, fp_, fn_ = get_patch_overlap(inference, segmentation)
            tp += tp_
            fp += fp_
            fn += fn_

            dice, precision, recall = get_scores(tp_, fp_, fn_)
            print(f"Dice: {dice:.2f} Precision {precision:.2f} Recall {recall:.2f}")

            # Save the inference
            orig_file_name = data["name"]
            base_name, _ = os.path.splitext(orig_file_name)
            new_file_name = f"{base_name}_output.nii"
            output_dir = settings["dataset"]["output_path"]
            path = os.path.join(output_dir, new_file_name)
            save_nifti(path, inference)
            
            #TODO Wrong here...
            # Create an overlay, save it, too
            overlay = create_overlay(inference, segmentation)
            new_file_name = new_file_name.replace("_output.nii", "_overlay.nii")
            path = os.path.join(output_dir, new_file_name)
            save_nifti(path, overlay)

    dice, precision, recall = get_scores(tp, fp, fn)
    return dice, precision, recall


def tta_voxel_stats(checkpoint_path, test_dataset, settings):
    """Perform a test round on a model with test time augmentation where
    the volume is flipped several times and gaussian noise is added.
    This calculates *voxel scores*
    """
    model = LightningUnet.load_from_checkpoint(checkpoint_path)

    voxel_dice        = torchmetrics.Dice(zero_division = 1)
    voxel_acc         = torchmetrics.classification.BinaryAccuracy() 
    voxel_precision   = torchmetrics.classification.BinaryPrecision() 
    voxel_recall      = torchmetrics.classification.BinaryRecall() 

    model.eval()
    for i, data in enumerate(test_dataset):
        with torch.no_grad():
            print(f"{i}/{len(test_dataset)}")
            inference = model(data['volume'].unsqueeze(0)) # Assumed model expects single image tensor
            
            print(f"{inference.shape} {np.min(inference)} {np.mean(inference)} {np.max(inference)} {inference.dtype}")
            
            n = 1.0
            for _ in range(4):
                # Test time augmentation
                volume = RandGaussianNoised(keys="volume", prob=1.0, std=0.001)(data)[
                        "volume"
                ]
                inference_ = model(volume.unsqueeze(0))
                inference_[inference_ > 0.5] = 1
                inference_[inference_ <= 0.5] = 0
                inference += inference_
                n += 1.0
                for dims in [[2], [3]]:
                    flip_inference = model(torch.flip(volume, dims=dims).unsqueeze(0))
                    inference_ = torch.flip(flip_inference, dims=dims)
                    inference_[inference_ > 0.5] = 1
                    inference_[inference_ <= 0.5] = 0
                    inference += inference_
                    n += 1.0
                inference = inference / n
            
            # inference = F.sigmoid(inference)
            inference[inference > 0.5] = 1
            inference[inference <= 0.5] = 0
            # Calculate voxel scores
            segmentation = data["segmentation"]
            inference = inference.squeeze()
            segmentation = segmentation.squeeze()
            voxel_dice.update(inference, segmentation.int())
            voxel_acc.update(inference, segmentation.int())
            voxel_precision.update(inference, segmentation.int())
            voxel_recall.update(inference, segmentation.int())

            inference = inference.detach().cpu().numpy()
            inference = np.squeeze(np.squeeze(inference.astype(np.uint8)))

            segmentation = np.squeeze(np.squeeze(data["segmentation"].cpu().numpy().astype(np.uint8)))

    print(f"Voxel:\nDice {voxel_dice.compute():.2f} Precision {voxel_precision.compute():.2f} Recall {voxel_recall.compute():.2f}")

def tta_instance_stats(checkpoint_path, test_dataset, settings):
    """Perform a test round on a model with test time augmentation where
    the volume is flipped several times and gaussian noise is added.
    This calculates *instance scores*
    """
    model = LightningUnet.load_from_checkpoint(checkpoint_path)
    tp, fp, fn = 0, 0, 0
    batch_size = settings["network"]["batch_size"]
    num_workers = settings["network"]["num_workers"]
    if num_workers == -1:
        num_workers = os.cpu_count()
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers)

    model.eval()

    for i, data in enumerate(test_dataset):
        with torch.no_grad():
            print(f"{i}/{len(test_dataset)}")
            inference = model(data["volume"].unsqueeze(0))
            n = 1.0
            for _ in range(4):
                # Test time augmentation
                volume = RandGaussianNoised(keys="volume", prob=1.0, std=0.001)(data)[
                        "volume"
                ]
                inference_ = model(volume.unsqueeze(0))
                inference_[inference_ > 0.5] = 1
                inference_[inference_ < 0.5] = 0
                inference += inference_
                n += 1.0
                for dims in [[2], [3]]:
                    flip_inference = model(torch.flip(volume, dims=dims).unsqueeze(0))
                    inference_ = torch.flip(flip_inference, dims=dims)
                    inference_[inference_ > 0.5] = 1
                    inference_[inference_ < 0.5] = 0
                    inference += inference_
                    n += 1.0
                inference = inference / n
            
            # Assuming the model output is a tensor, detach it from the computation graph and convert to numpy
            inference = inference.detach().cpu().numpy()
            inference[inference > 0.5] = 1
            inference[inference <= 0.5] = 0
            inference = np.squeeze(np.squeeze(inference.astype(np.uint8)))

            # Load the segmentation for instance scores
            segmentation = np.squeeze(np.squeeze(data["segmentation"].cpu().numpy().astype(np.uint8)))

            # Calculate instance scores
            tp_, fp_, fn_ = get_patch_overlap(inference, segmentation)
            tp += tp_
            fp += fp_
            fn += fn_

            dice, precision, recall = get_scores(tp_, fp_, fn_)
            print(f"Dice: {dice:.2f} Precision {precision:.2f} Recall {recall:.2f}")

            # Save the inference
            orig_file_name = data["name"]
            base_name, _ = os.path.splitext(orig_file_name)
            new_file_name = f"{base_name}_output.nii"
            output_dir = settings["dataset"]["output_path"]
            path = os.path.join(output_dir, new_file_name)
            save_nifti(path, inference)

    dice, precision, recall = get_scores(tp, fp, fn)

    print(f"Dice: {dice:.2f} Precision {precision:.2f} Recall {recall:.2f}")

def split_list(input_list, split_rate, split_amount=-1):
    """Splits a list into n = 1 / split_rate pairs of two disjunct sublists
    Args:
        input_list (list):   List containing all elements
        split_rate (float):   Percentage of elements contained in the small list
    Returns:
        A list containing n tuples of lists
    """
    if split_amount == -1:
        split_amount = int(1/split_rate)
    split_size = int(np.ceil(len(input_list) * split_rate))
    shuffle(input_list)
    result_list = []
    for iteration in range(split_amount):
        small_split = input_list[iteration*split_size:(iteration+1)*split_size]
        big_split = [i for i in input_list if not i in small_split]
        result_list.append((big_split, small_split))
    return result_list

def export_model_weights(model_path,output_path):
    print(f"Exporting the model to {output_path}")
    #load training result 
    new_model = torch.load(model_path,map_location="cpu")

    #create an empty dict and orderedDict
    new_model_subset = {} 
    state_dict = OrderedDict()

    #replace 'network' with 'module', write to state_dict
    for name,value in new_model['state_dict'].items():
        name = name.replace('network','module')
        state_dict[name] = value

    #add to subset 
    new_model_subset['state_dict'] = state_dict

    #save result as weights 
    torch.save(new_model_subset,output_path)
    


if __name__ == "__main__":
    parser =  argparse.ArgumentParser(description="DELIVR training pipeline")
    parser.add_argument("config", metavar="config", type=str, nargs="*", default="config.json", help="Path for the config file; default is in the same folder as the __main__.py file (./config_train.json)")

    args = parser.parse_args()

    config_location = args.config[0]
    print(config_location)
    # load config
    settings = {}
    with open(config_location, "r") as file:
        settings = json.load(file)


    raw_list    = [x for x in sorted(os.listdir(settings["dataset"]["raw_path"]))]
    gt_list     = [x for x in sorted(os.listdir(settings["dataset"]["gt_path"]))]

    # Training parameters
    max_epochs = settings["training"]["epochs"]
    learning_rate = settings["training"]["learning_rate"]

    batch_size = settings["network"]["batch_size"]
    num_workers = settings["network"]["num_workers"]
    if num_workers == -1:
        num_workers = os.cpu_count()
    
    print(f"MAXEPOCHS:{max_epochs}")

    # splits? If not, split yourself
    if not "test_list" in settings["training"].keys() or settings["training"]["test_list"] == "":
        train_test_splits = split_list(gt_list, 0.2)
        len_train_test = len(train_test_splits)
    else:
        train_test_splits = [[-1, -1]]
        if os.path.isdir(settings["training"]["test_list"]):
            test_list = os.listdir(settings["training"]["test_list"])
        else:
            test_list = settings["training"]["test_list"]
        train_test_splits[0][1] = test_list
        train_test_splits[0][0] = [x for x in gt_list if not x in test_list]


    # create data loaders
    best_fold, best_dice, best_scores = -1, 0, 0
    best_model_path = ""

    # Transforms
    keys=["volume", "segmentation"]

    transforms_arr = [
        LoadImaged(keys, ensure_channel_first=True, image_only=True),
        ToTensord(keys)
        ]

    transforms_train_arr = [
        LoadImaged(keys, ensure_channel_first=True, image_only=True),
        ToTensord(keys),
        RandGaussianNoised(keys=["volume"], prob=0.2, std=0.01),
        RandFlipd(keys, spatial_axis=0, prob=0.5),
        RandFlipd(keys, spatial_axis=1, prob=0.5),
        RandFlipd(keys, spatial_axis=2, prob=0.5),
        RandAffined(
            keys,
            prob=0.25,
            # 3 parameters control the transform on 3 dimensions
            rotate_range=(0.05, 0.05, 0.05, None),
            scale_range=(0.1, 0.1, 0.1, None),
            mode=("bilinear", "nearest")
            )
        ]

    if settings["training"]["normalization"]:
        transforms_arr.append(ScaleIntensityd(keys))
        transforms_train_arr.append(ScaleIntensityd(keys))


    transforms_train = Compose(transforms_train_arr)
    transforms = Compose(transforms_arr)

    len_train_test = len(train_test_splits)
    torch.multiprocessing.set_sharing_strategy('file_system')


    # Logger
    #TODO Fix run dir
    # logger = TensorBoardLogger("/home/rami/runs/graphs/", name = f"DELIVR {datetime.datetime.now()}")

    for test_fold, train_test_list in enumerate(train_test_splits[:1]):
        test_list = train_test_list[1]
        train_list = train_test_list[0]
        settings["training"]["test_list"] = test_list
        settings["training"]["train_list"] = train_list

        train_val_splits = split_list(train_list, 0.2)
        len_train_val = len(train_val_splits)
        best_val_fold, best_val_dice = 0, 0
        for val_fold, train_val_list in enumerate(train_val_splits[:2]):
            print(f"Test Fold :{test_fold}:{len_train_test}| Train_Val Fold:{val_fold}:{len_train_val}")
            #Use monai.CacheDataset
            val_list = train_val_list[1]
            train_list = train_val_list[0]
            train_list_dict = [{"volume":settings["dataset"]["raw_path"] + i,\
                                "segmentation":settings["dataset"]["gt_path"] + i,
                                "name":i}\
                                for i in train_list]
            val_list_dict = [{"volume":settings["dataset"]["raw_path"] + i,\
                                "segmentation":settings["dataset"]["gt_path"] + i,
                                "name":i}\
                                for i in val_list]



            train_dataset = CacheDataset(train_list_dict, transforms_train)
            val_dataset = CacheDataset(val_list_dict, transforms_train)  

            train_dataloader    = DataLoader(train_dataset, batch_size =batch_size, num_workers=num_workers)
            val_dataloader      = DataLoader(val_dataset, batch_size =batch_size,num_workers=num_workers)

            # train_dataloader = DataLoader(TrainingDataset(settings, train_list), batch_size =batch_size, num_workers=num_workers)
            # val_dataloader = DataLoader(TrainingDataset(settings, val_list), batch_size =batch_size,num_workers=num_workers)

            unet = LightningUnet(learning_rate, batch_size=batch_size, epochs=max_epochs, len_dataloader=len(train_dataloader))
            if settings["training"]["retrain"]:
                checkpoint_path = settings["dataset"]["delivr_model_path"]
                checkpoint = torch.load(checkpoint_path)
                state_dict = dict(checkpoint["model_state"].copy())
                new_dict = {}
                keys = state_dict.keys()
                for key in keys:
                    new_dict[key.replace("module","network")] = state_dict[key]
                unet.load_state_dict(new_dict)

            trainer = L.Trainer(log_every_n_steps = 1,
                               max_epochs = max_epochs,
                               check_val_every_n_epoch = 1,
                               default_root_dir = settings["dataset"]["output_path"])#,
                               #logger = logger)

            # logger.log_graph(unet)

            trainer.fit(unet, train_dataloader, val_dataloader)
            trainer.save_checkpoint(settings["dataset"]["checkpoint_path"] + f"{test_fold}-{val_fold}.ckpt")
            val_dice = trainer.validate(unet, dataloaders=val_dataloader)[0]["val_dice"]
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_val_fold = val_fold

        print(f"{test_fold} {best_val_fold} {best_val_dice}")
        train_dataloader = 0
        val_dataloader = 0

        test_list_dict = [{"volume":settings["dataset"]["raw_path"] + i,\
                            "segmentation":settings["dataset"]["gt_path"] + i,
                           "name":i}\
                            for i in train_list]
        test_dataset = CacheDataset(test_list_dict, transforms)
        batch_size = settings["network"]["batch_size"]
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers)

        ckpt_path = settings["dataset"]["checkpoint_path"] + f"{test_fold}-{best_val_fold}.ckpt"
        test_scores = trainer.test(unet, test_dataloader, ckpt_path=ckpt_path,verbose=True)[0]
        test_dice = test_scores["test_dice"]

        if test_dice > best_dice:
            best_dice       = test_dice
            best_fold       = test_fold
            best_model_path = ckpt_path
            best_scores     = test_scores

        # if settings["training"]["tta"]:
        #     tta_voxel_stats(ckpt_path, test_dataset, settings)
        #     tta_instance_stats(ckpt_path, test_dataset, settings)
        # else:
        #     instance_dice, instance_precision, instance_recall = test_instance(ckpt_path, test_dataset, settings)
        #     print(f"Dice: {instance_dice:.2f} Precision {instance_precision:.2f} Recall {instance_recall:.2f}")

    model_datetime = datetime.datetime.now()
    with open(settings["dataset"]["output_path"] + f"{model_datetime}_config.json","x") as file:
        json.dump(settings, file, indent=2)
    
    print(f"Best scores dict:\n{best_scores}")
    best_scores_df = pd.DataFrame([best_scores.values()], columns=best_scores.keys())
    best_scores_df.to_csv(settings["dataset"]["output_path"] + f"{model_datetime}_scores.csv")

    print(f"Best Fold {best_fold} \t {best_dice}")
    print(f"best_model_path:\n{best_model_path}")

    easy_access = settings["dataset"]["output_path"] + best_model_path.split("/")[-1]

    model_name = best_model_path.split("/")[-1]

    print(f"Model name : {best_model_path.split('/')[-1]}")
    print(f"easy_access {easy_access}")
    best_path = settings["dataset"]["output_path"] + f"{model_datetime}_best_model.ckpt"
    print(f"Proposed name {best_path}")

    #TODO Save CSV in output
    #Load model, extract state_dict, rename 'network' -> '
    export_model_weights(best_model_path, output_path=best_path)

