# '''
# Author: DRQ
# Date: 2021-12-15 20:34:17
# LastEditTime: 2021-12-15 20:37:52
# LastEditors: DRQ
# Description: try one try
# FilePath: \udacityControl\trialThings.py
# drq2015@outlook.com
# '''
# import csv
# import pandas as pd
# csv_path = "D:\\udacity\\term1-simulator-windows\\dataTest\\driving_log.csv"

# csv_file = csv.reader(open(csv_path,'r'))
# # for line in csv_file:
# #     print(type(line),'\n',line,'\n',type(csv_file))
# #     print(len(csv_file))
# #     break

# reader = csv.DictReader(csv_path)
# for row in reader:
#     print(row)
#     break

# csv_file = pd.read_csv(csv_path)
# print(len(csv_file['speed']),type(csv_file['speed']),csv_file.keys())
#######################################################################
# class SimplePIController:
#     def __init__(self, Kp, Ki):
#         self.Kp = Kp
#         self.Ki = Ki
#         self.set_point = 0.
#         self.error = 0.
#         self.integral = 0.

#     def set_desired(self, desired):
#         self.set_point = desired

#     def update(self, measurement):
#         # proportional error
#         self.error = self.set_point - measurement

#         # integral error
#         self.integral += self.error

#         return self.Kp * self.error + self.Ki * self.integral


# controller = SimplePIController(0.1, 0.002)
# set_speed = 6
# controller.set_desired(set_speed)
# for speed in range(12):
#     throttle = controller.update(float(speed))
#     print(throttle)
#####################
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# import pandas as pd
# from PIL import Image
# import matplotlib.pyplot as plt
# import torchvision.transforms.functional as TF

# csv_path = "D:\\udacity\\term1-simulator-windows\\dataRecord\\driving_log2.csv"

# csv_file = pd.read_csv(csv_path)
# img = Image.open(csv_file['center'][0])

# print(img.height,img.width)

# img=TF.center_crop(img,320)

# plt.figure("dog")
# plt.imshow(img)
# plt.show()
############

# from net import LeNet,Unet2Control
# epoch=1
# net = LeNet()
# name=net.__class__.__name__ + "%3.1f.ckpt" % epoch
# print(net.__class__.__name__,type(net.__class__.__name__),name)
###############################
from net import LeNet,Unet2Control,config
import torch

img = torch.randn(4,3,320,160).to(device=config["device"])
net = Unet2Control(config).to(device=config["device"])
a = net(img)
print(a)