import os
import sys
from numpy.core.fromnumeric import _cumprod_dispatcher
import pandas as pd

cur_dir_path = os.path.abspath(os.curdir)
# csv_path = os.path.join(cur_dir_path,)
csv_path = "C:\\Users\\DRQ\\Desktop\\driving_log.csv"
csv_file = pd.read_csv(csv_path)
center_images_fps = csv_file["center"]
left_images_fps = csv_file["left"]
right_images_fps = csv_file["right"]
angle_fps = csv_file["angle"]
fore_fps = csv_file["fore"]
back_fps = csv_file["back"]
speed_fps = csv_file["speed"]
# center	left	right	angle	fore	back	speed

r_path = "\\udacity\\term1-simulator-windows"
r2_path = os.path.curdir    #TODO 改
r2_csv_name = "todo"    #TODO gai


for idx in range(len(center_images_fps)):
    center_images_fps[idx]=center_images_fps[idx].replace(r_path,r2_path)
    center_images_fps[idx]=center_images_fps[idx].replace('\','/'')
for idx in range(len(left_images_fps)):
    left_images_fps[idx]=left_images_fps[idx].replace(r_path,r2_path)
    left_images_fps[idx]=left_images_fps[idx].replace('\','/'')
for idx in range(len(right_images_fps)):
    right_images_fps[idx]=right_images_fps[idx].replace(r_path,r2_path)
    right_images_fps[idx]=right_images_fps[idx].replace('\','/'')

csv_file.to_csv("test.csv",index=False)

