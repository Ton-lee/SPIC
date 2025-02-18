import os
import numpy as np
import cv2
import tqdm

# 定义数据路径
oar_root = "/home/Users/dqy/Dataset/Cityscapes/format_yolo/labels/"
layout_save_root = "/home/Users/dqy/Dataset/Cityscapes/layout_ssm/"

target_dim = 19

size = [512, 256]
for phase in ["train", "val"]:
    layout_save_folder = os.path.join(layout_save_root, phase)
    os.makedirs(layout_save_folder, exist_ok=True)
    oar_folder = os.path.join(oar_root, phase)
    files = sorted(os.listdir(oar_folder))
    for file in tqdm.tqdm(files):
        oar_file = os.path.join(oar_folder, file)
        layout_save_path = os.path.join(layout_save_folder, file.replace("boundingBox.txt", "layout.npy"))
        layout = np.zeros([size[1], size[0], target_dim], dtype="int")
        oars = np.loadtxt(oar_file)
        for category, x_mid, y_mid, w, h in oars:
            x_min = x_mid - w / 2
            y_min = y_mid - h / 2
            x_max = x_mid + w / 2
            y_max = y_mid + h / 2
            x_min = max(0, int(x_min * size[0]))
            y_min = max(0, int(y_min * size[1]))
            x_max = min(size[0], int(x_max* size[0]))
            y_max = min(size[1], int(y_max* size[1]))
            layout[y_min:y_max, x_min:x_max, int(category)] = 1
        np.save(layout_save_path, layout)
