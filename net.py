# '''
# Author: your name
# Date: 2021-12-06 14:49:18
# LastEditTime: 2021-12-15 20:24:19
# LastEditors: DRQ
# Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# FilePath: \udacityControl\net.py
# '''
import os
import numpy as np
# 导入pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18
from torchvision.models.segmentation import fcn_resnet50

# 设置数据集路径
cur_path = os.path.abspath(os.path.curdir)

### config ###
config = dict()

"""Dataset"""
# Raw Dataset
# TODO gai
# config["train_csv_path"] = "D:\\udacity\\term1-simulator-windows\\dataRecord\\driving_log2.csv"
config["train_csv_path"] = "D:\\udacity\\term1-simulator-windows\\dataTest\\driving_log.csv"
config["val_csv_path"] = "D:\\udacity\\term1-simulator-windows\\dataTest\\driving_log.csv"

config["crop_size"] = (320, 320)
config["center_crop_size"] = 320
config["degree"] = 30

"""unet"""
config["unet_n_channels"] = 3
config["unet_n_classes"] = 2

"""fcnres50"""
config["n_classes"] = 2


"""Model"""
config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
config["save_dir"] = os.path.join(cur_path, "results")
config["test_img_save_dir"] = os.path.join(config["save_dir"],"test_img")

config["colormaps"] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]#TODO 根据具体场景或者数据集来确定
config["classes"] = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']#TODO 根据具体场景或者数据集来确定

"""Train"""
config["train_batch_size"] = 6
config["val_batch_size"] = config["train_batch_size"]

config["workers"] = 0
config["val_workers"] = config["workers"]

config["lr"] = 1e-3
config["weight_decay"] = 1e-8


config["num_epochs"] = 36
config["save_freq"] = 5.0
config["val_freq"] = 5

"""test"""
config["test_batch_size"] = 1



### end of config ###

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# UNet
class UNet(nn.Module):
    """
    in:(b,c,h,w);out:(b,num_classes,h,w)
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.out  = torch.sigmoid #此处记得有sigmoid
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.out(logits)
        return logits

# FCN_Res50
class FcnRes50(nn.Module):
    """
    in:(b,c,h,w);out:(b,num_classes,h,w) 
    """
    def __init__(self, n_classes):
        super(FcnRes50, self).__init__()
        self.net = fcn_resnet50(num_classes=n_classes)

    def forward(self, x):
        logits = self.net(x)
        return logits["out"]

class PostProcess(nn.Module):
    """
    post process for semantic seg:(b,num_classes,h,w)2(b,1,h,w)
    """
    def __init__(self, config) -> None:
        super(PostProcess, self).__init__()
        self.config = config
        self.colormaps = torch.Tensor(config["colormaps"])

    def forward(self, preds):
        seg_idx_img = torch.argmax(preds,dim=1)
        seg_idx_img = seg_idx_img.view(seg_idx_img.size()[0],1,seg_idx_img.size()[-2],seg_idx_img.size()[-1])
        return seg_idx_img  # (b,1,h,w)

    def pred_metrics(self,seg_idx_imgs,labels,):
        # 这是一个batch的正确率，整个epoch的需要做一个列表，然后求平均
        seg_idx_imgs = seg_idx_imgs.cpu().numpy()
        labels = labels.cpu().numpy()
        return np.sum(seg_idx_imgs==labels)/np.prod(labels.shape)
    
    def label2color(self, label):
        """
        label:(b,1,h,w)
        return:(b,1,h,w,3)
        """
        return self.colormaps[label,:]


class Fcn2Control(nn.Module):
    def __init__(self, config):
        super(Fcn2Control, self).__init__()
        self.fcn_net = fcn_resnet50(num_classes=config["n_classes"])
        self.resnet = resnet18()
        
        self.RGB2GrayConv = torch.nn.Conv2d(3,1,1)
        
        self.fc1 = torch.nn.Linear(1000,256)
        self.fc2 = torch.nn.Linear(256,32)
        self.fc3 = torch.nn.Linear(1000,128)
        self.fc4 = torch.nn.Linear(128,32)
        self.fc5 = torch.nn.Linear(32,1)
        self.fc6 = torch.nn.Linear(64,1)

        self.colormap = torch.Tensor(config["colormaps"])
        
    def forward(self, x):
        #1
        fcn_out = self.fcn_net(x)
        seg_idx_img = torch.argmax(fcn_out["out"],dim=1)
        seg_idx_img = seg_idx_img.view(seg_idx_img.size()[0],1,seg_idx_img.size()[-2],seg_idx_img.size()[-1])
        #2
        gray_img = self.RGB2GrayConv(x)
        #3
        color_mask = self.label2color(seg_idx_img, self.colormap)
        color_mask =  color_mask.view( color_mask.shape[0], color_mask.shape[-1], color_mask.shape[-3], color_mask.shape[-2])
        gray_mask = self.RGB2GrayConv(color_mask)
        # 统一到torch.Tensor
        seg_idx_img = seg_idx_img.type(torch.Tensor)
        
        con_img =torch.cat((seg_idx_img,gray_img,gray_mask),dim=1)
        # print(seg_idx_img.type(),gray_img.type(),color_mask.type(),gray_mask.type(),seg_idx_img.shape,gray_img.shape,color_mask.shape,gray_mask.shape)
        
        res_out = self.resnet(con_img)
        
        # velocity
        v_feature = self.fc1(res_out)
        v_feature = self.fc2(v_feature)
        v = self.fc5(v_feature)
        # angle
        a_feature = self.fc3(res_out)
        a_feature = self.fc4(a_feature)
        a_feature = torch.cat((v_feature,a_feature),dim=1)
        a = self.fc6(a_feature)

        return v.squeeze(),a.squeeze()
    
    def label2color(self, label, colormap):
        """
        label:(b,1,h,w)
        return:(b,1,h,w,3)
        """
        return colormap[label,:]

class Unet2Control(nn.Module):
    """
    in(b,3,h,w),out(v,a)
    """
    def __init__(self, config):
        super(Unet2Control, self).__init__()
        self.unet = UNet(n_channels=config["unet_n_channels"],n_classes=config["unet_n_classes"])
        self.resnet = resnet18()
        
        self.RGB2GrayConv = torch.nn.Conv2d(3,1,1)
        
        self.fc1 = torch.nn.Linear(1000,256)
        self.fc2 = torch.nn.Linear(256,32)
        self.fc3 = torch.nn.Linear(1000,128)
        self.fc4 = torch.nn.Linear(128,32)
        self.fc5 = torch.nn.Linear(32,1)
        self.fc6 = torch.nn.Linear(64,1)

        self.colormap = torch.Tensor(config["colormaps"]).to(config["device"])
        
    def forward(self, x):
        #1
        unet_out = self.unet(x)
        seg_idx_img = torch.argmax(unet_out,dim=1)
        seg_idx_img = seg_idx_img.view(seg_idx_img.size()[0],1,seg_idx_img.size()[-2],seg_idx_img.size()[-1])
        #2
        gray_img = self.RGB2GrayConv(x)
        #3
        color_mask = self.label2color(seg_idx_img, self.colormap)
        color_mask =  color_mask.view(color_mask.shape[0], color_mask.shape[-1], color_mask.shape[-3], color_mask.shape[-2])
        gray_mask = self.RGB2GrayConv(color_mask)
        # 统一到torch.Tensor
        seg_idx_img = seg_idx_img.to(device=config["device"]).float()

        con_img =torch.cat((seg_idx_img,gray_img,gray_mask),dim=1)
        # print(seg_idx_img.type(),gray_img.type(),color_mask.type(),gray_mask.type(),seg_idx_img.shape,gray_img.shape,color_mask.shape,gray_mask.shape)
        
        res_out = self.resnet(con_img)
        
        # velocity
        v_feature = self.fc1(res_out)
        v_feature = self.fc2(v_feature)
        v = self.fc5(v_feature)
        # angle
        a_feature = self.fc3(res_out)
        a_feature = self.fc4(a_feature)
        a_feature = torch.cat((v_feature,a_feature),dim=1)
        a = self.fc6(a_feature)

        return v.squeeze(),a.squeeze()
    
    def label2color(self, label, colormap):
        """
        label:(b,1,h,w)
        return:(b,1,h,w,3)
        """
        return colormap[label,:]

class Unet2Angle(nn.Module):
    """
    in(b,3,h,w),out(a)
    """
    def __init__(self, config):
        super(Unet2Angle, self).__init__()
        self.unet = UNet(n_channels=config["unet_n_channels"],n_classes=config["unet_n_classes"])
        self.resnet = resnet18()
        
        self.RGB2GrayConv = torch.nn.Conv2d(3,1,1)
        
        self.fc3 = torch.nn.Linear(1000,128)
        self.fc4 = torch.nn.Linear(128,32)
        self.fc5 = torch.nn.Linear(32,1)

        self.colormap = torch.Tensor(config["colormaps"]).to(config["device"])
        
    def forward(self, x):
        #1
        unet_out = self.unet(x)
        seg_idx_img = torch.argmax(unet_out,dim=1)
        seg_idx_img = seg_idx_img.view(seg_idx_img.size()[0],1,seg_idx_img.size()[-2],seg_idx_img.size()[-1])
        #2
        gray_img = self.RGB2GrayConv(x)
        #3
        color_mask = self.label2color(seg_idx_img, self.colormap)
        color_mask =  color_mask.view(color_mask.shape[0], color_mask.shape[-1], color_mask.shape[-3], color_mask.shape[-2])
        gray_mask = self.RGB2GrayConv(color_mask)
        # 统一到torch.Tensor
        seg_idx_img = seg_idx_img.to(device=config["device"]).float()

        con_img =torch.cat((seg_idx_img,gray_img,gray_mask),dim=1)
        # print(seg_idx_img.type(),gray_img.type(),color_mask.type(),gray_mask.type(),seg_idx_img.shape,gray_img.shape,color_mask.shape,gray_mask.shape)
        
        res_out = self.resnet(con_img)
        
        # angle
        a_feature = self.fc3(res_out)
        a_feature = self.fc4(a_feature)
        a = self.fc5(a_feature)

        return a.squeeze()
    
    def label2color(self, label, colormap):
        """
        label:(b,1,h,w)
        return:(b,1,h,w,3)
        """
        return colormap[label,:]


class Conv2Angle(nn.Module):
    """NVIDIA model used in the paper."""

    def __init__(self):
        """Initialize NVIDIA model.
        NVIDIA model used
            Image normalization to avoid saturation and make gradients work better.
            Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Drop out (0.5)
            Fully connected: neurons: 100, activation: ELU
            Fully connected: neurons: 50, activation: ELU
            Fully connected: neurons: 10, activation: ELU
            Fully connected: neurons: 1 (output)
        the convolution layers are meant to handle feature engineering.
        the fully connected layer for predicting the steering angle.
        the elu activation function is for taking care of vanishing gradient problem.
        """
        super(Conv2Angle, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 13 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        """Forward pass."""
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output


class LeNet(nn.Module):
    """LeNet architecture."""

    def __init__(self):
        """Initialization."""
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass."""
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    opt_state_dict = opt.state_dict()
    # for key in opt_state_dict.keys():
    #     opt_state_dict[key] = opt_state_dict[key].cpu()

    save_name = net.__class__.__name__ + "%3.1f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt_state_dict},
        os.path.join(save_dir, save_name),
    )

def load_ckpt(net, opt, save_path):
    """
    net:定义好的网络，necessary
    opt:优化器（定义好的优化器 or None）；当None时，表示此时只需要保存的模型参数
    save_path:模型保存路径，
    在载入模型参数之前的初始模型，需要先设定好模型的位置（cpu&gpu）
    不能导入参数之后，再对模型位置进行设置
    """
    if os.path.exists(save_path):
        model_ckpt = torch.load(save_path)
        print("loading checkpoint...")

        state_dict = net.state_dict()
        for key in model_ckpt["state_dict"].keys():
            if key in state_dict and (model_ckpt["state_dict"][key].size() == state_dict[key].size()):
                value = model_ckpt["state_dict"][key]
                if not isinstance(value, torch.Tensor):
                    value = value.data
                state_dict[key] = value
        net.load_state_dict(state_dict)
        if opt is not None:
            opt.load_state_dict(model_ckpt["opt_state"])
        else:
            opt = opt
        print(os.path.split(save_path)[-1]," is loaded...")
        return net, opt
    else:
        print("*"*40)
        print("No checkpoint is included...")
        print("*"*40)
        return net, opt

def load_ckpt_with_filter(model, optimizer, checkpoint, loadOptimizer):
    if os.path.exists(checkpoint):
        print("loading checkpoint...")
        model_dict = model.state_dict()
        modelCheckpoint = torch.load(checkpoint)
        pretrained_dict = modelCheckpoint["state_dict"]
        # 过滤操作
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
        # 如果不需要更新优化器那么设置为false
        if loadOptimizer is True:
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            print('loaded! optimizer')
        else:
            print('not loaded optimizer')
    else:
        print('No checkpoint is included')
    return model, optimizer

if __name__ == "__main__":
    import numpy as np
    unet = UNet(3,2)
    unet2control = Unet2Control(config)
    in_tensor = torch.randn(5,3,320,320)
    out_tensor = unet2control(in_tensor)
    print(out_tensor)
    total = sum([param.nelement() for param in unet2control.parameters()])
    trainable = sum(p.numel() for p in unet2control.parameters() if p.requires_grad)
    print("num of parameter: %.2fM,%.2fM"%(total/1e6,trainable/1e6))
    quit()

    model = Fcn2Control(2)
    fcn = fcn_resnet50(num_classes=2)
    unet = UNet(3,2)
    total = sum([param.nelement() for param in model.parameters()])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num of parameter: %.2fM,%.2fM"%(total/1e6,trainable/1e6))
    total = sum([param.nelement() for param in fcn.parameters()])
    trainable = sum(p.numel() for p in fcn.parameters() if p.requires_grad)
    print("num of parameter: %.2fM,%.2fM"%(total/1e6,trainable/1e6))
    total = sum([param.nelement() for param in unet.parameters()])
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print("num of parameter: %.2fM,%.2fM"%(total/1e6,trainable/1e6))
    quit()
    # 设置数据集路径
    cur_path = os.path.abspath(os.path.curdir)
    dataset_path = os.path.join(cur_path, "data", "camvid")
    dataset_images_path = os.path.join(cur_path,"data","camvid","images")
    dataset_labels_path = os.path.join(cur_path,"data","camvid","labels")


    x_train_dir = os.path.join(dataset_path, 'train_images')
    y_train_dir = os.path.join(dataset_path, 'train_labels')

    x_valid_dir = os.path.join(dataset_path, 'valid_images')
    y_valid_dir = os.path.join(dataset_path, 'valid_labels')

    x_test_dir = os.path.join(dataset_path, 'test_images')
    y_test_dir = os.path.join(dataset_path, 'test_labels')


    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation()
        )
    img,label = train_dataset[1]
    img = np.expand_dims(img, 0)
    label = np.expand_dims(label, 0)
    img = np.transpose(img, (0,3,1,2)) # 交换通道顺序
    img = img/255. # 把image的值归一化到[0,1]
    images = torch.from_numpy(img).float()
    labels = torch.from_numpy(label).float()
    # images = Variable(img.to(dtype=torch.float32))
    # labels = Variable(label.to(dtype=torch.float32))

    net = UNet(n_channels=3, n_classes=1)
    net1 = FcnRes50(n_classes=1)
    net1.eval()

    pred = net1(images)

    print(pred.shape)

