from torch.utils.data import Dataset
from classification.utils.config import config
from itertools import chain
from glob import glob
from tqdm import tqdm
from .augmentations import get_train_transform,get_test_transform
import random 
import numpy as np 
import pandas as pd 
import os 
import cv2
import torch
import csv

#1.set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


#2.define dataset
class ChaojieDataset(Dataset):
    def __init__(self,label_list,train=False, val=False, test=False):
        self.test = test 
        self.train = train
        self.val = val
        imgs = []
        if self.test:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"]))
            self.imgs = imgs 
        else:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"],row["label"]))
            self.imgs = imgs

    def __getitem__(self,index):
        if self.test:
            filename = self.imgs[index]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # cv2.imshow('', img.astype(np.uint8))
            # cv2.waitKey()
            # img = cv2.resize(img,(int(config.img_height*1.5),int(config.img_weight*1.5)))
            # cv2.imshow('', img.astype(np.uint8))
            # cv2.waitKey()
            # print(img.shape)
            img = get_test_transform((config.img_height, config.img_weight))(image=img)["image"]
            # cv2.imshow('', img)
            # cv2.waitKey()
            return img,filename
        else:
            filename, label = self.imgs[index] 
            img = cv2.imread(filename)
            # print(filename)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img,(int(config.img_height*1.5),int(config.img_weight*1.5)))
            if self.val:
                img = get_test_transform((config.img_height, config.img_weight))(image=img)["image"]
            if self.train:
                img = get_train_transform((config.img_height, config.img_weight),augmentation=config.augmen_level)(image=img)["image"]
            return img, label

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), label


def get_files(root,mode):
    #for test
    if mode == "test":
        from pathlib import Path
        files = []
        # for img in os.listdir(root):
        #     files.append(root + img)
        files = [str(p) for p in Path(root).rglob('*.png')]
        files = pd.DataFrame({"filename":files})
        return files
    elif mode != "test": 
        #for train and val       
        all_data_path,labels = [],[]
        image_folders = list(map(lambda x:root+x,os.listdir(root)))
        all_images = list(chain.from_iterable(list(map(lambda x:glob(x+"/*"),image_folders))))
        print("loading train dataset")
        for file in tqdm(all_images):
            all_data_path.append(file)
            labels.append(int(file.split("/")[-2]))
        all_files = pd.DataFrame({"filename":all_data_path,"label":labels})
        return all_files
    else:
        print("check the mode please!")
    

def get_files_from_csv(root,mode):
    #for test
    if mode == "test":
        with open(root, 'r') as f:
            files = list(map(lambda x: x[0], list(csv.reader(f))))
        f.close()
        files = pd.DataFrame({"filename":files[int(len(files) * 0.8):]})
        return files
    elif mode != "test":
        #for train and val
        all_data_path,labels = [],[]
        with open(root, 'r') as f:
            r = csv.reader(f)
            for line in r:
                path, label = line
                all_data_path.append(path)
                labels.append(int(label))
        f.close()
        print("loading train dataset")
        # for file in tqdm(all_data_path):
        #     if '-neg' in file: label = 0
        #     else: label = 1
        #     labels.append(label)
        all_files = pd.DataFrame({"filename":all_data_path, "label":labels})
        # if mode == 'train':
        #     all_files = pd.DataFrame({"filename":all_data_path[: int(len(labels) * 0.6)],"label":labels[: int(len(labels) * 0.6)]})
        # else:
        #     all_files = pd.DataFrame({"filename":all_data_path[int(len(labels) * 0.6): int(len(labels) * 0.8)],
        #                               "label":labels[int(len(labels) * 0.6): int(len(labels) * 0.8)]})
        return all_files
    else:
        print("check the mode please!")