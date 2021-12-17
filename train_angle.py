# '''
# Author: DRQ
# Date: 2021-12-16 09:49:48
# LastEditTime: 2021-12-16 10:03:40
# LastEditors: DRQ
# Description: train unet2control
# FilePath: \udacityControl\train_unet2control.py
# drq2015@outlook.com
# '''
import os
import time
from typing import Dict
import numpy as np
from tqdm import tqdm
import argparse

# 导入pytorch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.models.segmentation.segmentation import fcn_resnet50

from data import UdacityDataset2
from net import Unet2Angle, Conv2Angle, config, load_ckpt, save_ckpt

def train_model(net:nn.Module, criterion, optimizer, data_loader:Dict, config:Dict):
    for epoch in range(config["num_epochs"]):
    
        net.train()
        epoch_loss = 0
        for data in tqdm(data_loader["train"]):
            
            img,ang = data

            images = img.to(device=config["device"]).float() 
            angs = ang.to(device=config["device"]).float()

            pred_angs = net(images)

            # print(pred_spds.size(),spds.size(),"|",pred_angs.size(),angs.size())
            loss = criterion(pred_angs, angs)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("No.", epoch, " done. epoch loss: ", epoch_loss)

        if epoch %config["save_freq"] == 0 or epoch+1 >= config["num_epochs"]:
            save_ckpt(net, optimizer,os.path.join(config["save_dir"],"model_save"),epoch)
            print("No.", epoch, " epoch model saved... ")

        if epoch %config["val_freq"] == 0 or epoch+1 >= config["num_epochs"]:
            net.eval()
            val_epoch_loss = 0

            start_time = time.time()
            for i, data in tqdm(enumerate(data_loader["val"])):
                img,ang = data

                images = img.to(device=config["device"]).float() 
                angs = ang.to(device=config["device"]).float()

                with torch.no_grad():
                    pred_angs = net(images)
                    
                    loss = criterion(pred_angs, angs)
                    val_epoch_loss += loss.item()
                
            dt = time.time() - start_time
            #TODO 此处需要展示文字性的预测结果，如评价指标，耗时等
            print("val take time: ", dt, "s, val total loss: ",val_epoch_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_bs', default=5, type=int, help='batch size')
    parser.add_argument('-n_epoch', default=36, type=int, help='num of epoch')
    parser.add_argument('-resume', default=None, type=str, help='path to saved model, if not None, load model, continue to train')
    args = parser.parse_args()

    config["train_batch_size"]=args.train_bs
    config["val_batch_size"] = config["train_batch_size"]
    config["num_epochs"] = args.n_epoch

    utrain_data = UdacityDataset2(
        csv_path=config["train_csv_path"],#train_csv_path
        is_train=True,
        config=config
    )
    utrain_loader = DataLoader(utrain_data,batch_size=config["train_batch_size"],shuffle=True)

    uval_data = UdacityDataset2(
    csv_path=config["val_csv_path"],#train_csv_path
    is_train=True,
    config=config
    )
    uval_loader = DataLoader(uval_data,batch_size=config["train_batch_size"],shuffle=True)

    data_loader={}
    data_loader["train"] = utrain_loader
    data_loader["val"] = uval_loader

    net = Unet2Angle(config=config)

    if args.resume is not None:
        net, _ = load_ckpt(net, None, args.resume)
        
    net = net.to(device=config["device"])  #.cuda()需要在构建优化器之前执行

    optimizer = optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    criterion = nn.SmoothL1Loss()   

    train_model(net, criterion, optimizer, data_loader, config)





