# '''
# Author: DRQ
# Date: 2021-12-15 20:30:26
# LastEditTime: 2021-12-16 08:52:00
# LastEditors: DRQ
# Description: 
# FilePath: \udacityControl\data.py
# drq2015@outlook.com
# '''
import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import csv

# 导入pytorch
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
import torchvision
from torchvision.transforms.transforms import ToTensor

from net import config

class UdacityDataset(BaseDataset):
    """Udacity. Read images, apply augmentation and preprocessing transformations.
    
    Args:
    
    """
    
    def __init__(
            self, 
            csv_path,
            is_train,
            config,
    ):
        self.csv_file = pd.read_csv(csv_path)
        self.images_fps = self.csv_file["center"]
        self.speed_fps = self.csv_file["speed"]
        self.angle_fps = self.csv_file["angle"]
        
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])

        self.crop_size = config["crop_size"] # (h, w)
        self.center_crop_size = config["center_crop_size"]
        self.rotate_degree = config["degree"]
        self.is_train = is_train

        self.tsf_feature = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])    # 对于RGB的feature有归一化操作
        self.tsf_label = transforms.Compose([transforms.ToTensor()])                # 对于L的label无归一化操作，故有俩tsf


    def __getitem__(self, i):      
        # read data
        image = Image.open(self.images_fps[i])
        angle = self.angle_fps[i]
        speed = self.speed_fps[i]
       
        if self.is_train:
            image = self.rand_crop(image,self.crop_size[0],self.crop_size[1],self.rotate_degree)
        else:
            image = self.center_crop(image,self.center_crop_size)

        return self.tsf_feature(image), torch.Tensor(np.array(angle)), torch.Tensor(np.array(speed))   

    def __len__(self):
        return len(self.images_fps)
    
    def rand_crop(self, feature, height, width, degree):
        """
        随机裁剪、概率旋转和概率水平翻转（p=0.5）feature(PIL image) 和 label(PIL image).
        为了使裁剪的区域相同，不能直接使用RandomCrop，而要像下面这样做
        Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        feature = transforms.functional.center_crop(feature,(height,width))
        if random.random() > 0.5: 
            r_degree = random.randint(-degree, degree)
            feature = feature.rotate(r_degree)

        return feature

    def center_crop(self, feature,crop_size):
        feature = transforms.functional.center_crop(feature,crop_size)
        return feature


class UdacityDataset2(BaseDataset):
    """Udacity. Read images, apply augmentation and preprocessing transformations.
    
    Args:
    
    """
    
    def __init__(
            self, 
            csv_path,
            is_train,
            config,
    ):
        self.csv_file = pd.read_csv(csv_path)
        self.images_fps = self.csv_file["center"]
        # self.speed_fps = self.csv_file["speed"]
        self.angle_fps = self.csv_file["angle"]
        
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])

        self.crop_size = config["crop_size"] # (h, w)
        self.center_crop_size = config["center_crop_size"]
        self.rotate_degree = config["degree"]
        self.is_train = is_train

        self.tsf_feature = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])    # 对于RGB的feature有归一化操作
        self.tsf_label = transforms.Compose([transforms.ToTensor()])                # 对于L的label无归一化操作，故有俩tsf


    def __getitem__(self, i):      
        # read data
        image = Image.open(self.images_fps[i])
        angle = self.angle_fps[i]
        # speed = self.speed_fps[i]
       
        if self.is_train:
            image = self.rand_crop(image,self.crop_size[0],self.crop_size[1],self.rotate_degree)

        return self.tsf_feature(image), torch.Tensor(np.array(angle))   

    def __len__(self):
        return len(self.images_fps)
    
    def rand_crop(self, feature, height, width, degree):
        """
        随机裁剪、概率旋转和概率水平翻转（p=0.5）feature(PIL image) 和 label(PIL image).
        为了使裁剪的区域相同，不能直接使用RandomCrop，而要像下面这样做
        Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        # feature = transforms.functional.center_crop(feature,(height,width))
        if random.random() > 0.5: 
            r_degree = random.randint(-degree, degree)
            feature = feature.rotate(r_degree)

        return feature


if __name__ == "__main__":
    csv_path = "D:\\udacity\\term1-simulator-windows\\dataRecord\\driving_log2.csv"
    udata = UdacityDataset(
        csv_path=csv_path,
        is_train=True,
        config=config
    )
    uloader = DataLoader(udata,batch_size=5,shuffle=True)
    for data in uloader:
        img,ang,spd = data
        print(type(img),type(ang),type(spd),img.size(),ang.size(),spd.size(),img.shape,ang.shape,spd.shape)
        quit()