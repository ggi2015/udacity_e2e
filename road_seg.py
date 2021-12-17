# '''
# Author: DRQ
# Date: 2021-12-15 16:35:17
# LastEditTime: 2021-12-15 17:14:49
# LastEditors: DRQ
# Description: from udacity get pic and semantic seg
# FilePath: \udacityControl\road_seg.py
# drq2015@outlook.com
# '''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np

import argparse
import base64
from datetime import datetime

import time
import shutil
import matplotlib.pyplot as plt
plt.ion()

import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

sio = socketio.Server()
app = Flask(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models import resnet18
from torchvision.models.segmentation import fcn_resnet50

from net import load_ckpt, config, PostProcess, Unet2Control

root_path = os.path.abspath(os.curdir)
model_path = os.path.join(root_path,"results","model_save","75.0.ckpt")

# net = fcn_resnet50()
# net, _ = load_ckpt(net,None,model_path)
# post_process = PostProcess(config)
# net.to(device=config["device"])
# post_process.to(device=config["device"])

net = Unet2Control(config)
net, _ = load_ckpt(net,None,model_path)
net.to(device=config["device"])


center_crop_size = 320
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
tsf = transforms.Compose([transforms.CenterCrop(center_crop_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean,std=std)])
tsf_full_img = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=mean,std=std)])

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    # plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        # plt.pause(.1)
        
    plt.show()
    plt.clf()

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 10
controller.set_desired(set_speed)



@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        spd, ang = net(tsf(image).unsqueeze(dim=0).to(device=config["device"]))
        # print(type(spd),spd.shape,spd.size(),spd)

        ang = ang.detach().cpu().numpy()
        spd = spd.detach().cpu().numpy()
        print(ang, spd)
        #steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        steering_angle = ang
        throttle = controller.update(float(speed))

        # throttle = float(spd)

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
       # if args.image_folder != '':
          #  timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
           # image_filename = os.path.join(args.image_folder, timestamp)
          #  image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

if __name__ == "__main__":
        # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)