import os
import numpy as np
from sklearn.decomposition import PCA
import cv2
import tqdm

# 定义数据路径
data_path = "/home/Users/dqy/Dataset/Cityscapes/Embeddings/"
oar_root = "/home/Users/dqy/Dataset/Cityscapes/format_yolo/labels/"
layout_save_root = "/home/Users/dqy/Dataset/Cityscapes/layout_prior/"

# 定义优先级列表
semantic_priority = [11, 12, 18, 17, 13, 14, 15, 16, 6, 7, 4, 5, 8, 0, 1, 9, 2, 3, 10]

# 读取所有文件
file_names = [f for f in os.listdir(data_path) if f.endswith('.npy')]

# 用于存储文件内容和标签
embeddings = []
labels = []

# 加载文件内容
for file_name in file_names:
    file_path = os.path.join(data_path, file_name)
    embedding = np.load(file_path)  # 加载 512 维向量
    label = file_name.split('-')[0]  # 使用文件名按 '-' 分割的第一个元素作为标签
    embeddings.append(embedding)
    labels.append(label)

# 转换为 numpy 数组
embeddings = np.array(embeddings)
labels = np.array(labels)

# PCA 降维
target_dim = 16
pca = PCA(n_components=target_dim)
reduced_features = pca.fit_transform(embeddings)

size = [512, 256]
for phase in ["train", "val"]:
    layout_save_folder = os.path.join(layout_save_root, phase)
    os.makedirs(layout_save_folder, exist_ok=True)
    oar_folder = os.path.join(oar_root, phase)
    files = sorted(os.listdir(oar_folder))
    for file in tqdm.tqdm(files):
        oar_file = os.path.join(oar_folder, file)
        layout_save_path = os.path.join(layout_save_folder, file.replace("boundingBox.txt", "layout.npy"))
        layout = np.zeros([size[1], size[0], target_dim])
        oars = np.loadtxt(oar_file)
        for semantic in semantic_priority[::-1]:  # 先填充低优先级的区域，这样会保证后填充的高优先级区域将低优先级区域覆盖
            oars_fill = oars[oars[:, 0].astype("int") == semantic]
            for category, x_mid, y_mid, w, h in oars_fill:
                x_min = x_mid - w / 2
                y_min = y_mid - h / 2
                x_max = x_mid + w / 2
                y_max = y_mid + h / 2
                x_min = max(0, int(x_min * size[0]))
                y_min = max(0, int(y_min * size[1]))
                x_max = min(size[0], int(x_max* size[0]))
                y_max = min(size[1], int(y_max* size[1]))
                layout[y_min:y_max, x_min:x_max] = reduced_features[int(category)]
        np.save(layout_save_path, layout)
