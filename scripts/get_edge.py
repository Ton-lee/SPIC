"""边缘检测。对数据集进行处理，即目标文件夹中包含多个子文件夹，表示多段视频
代码框架继承自 sketch_extracting_dataset
"""
import sys
import warnings

from tqdm import tqdm
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("/home/Users/dqy/Projects/VideoSketch")
from tools.utils import Source, Logger
from tools.detector import EdgeDetector
from tools.utils import AnyOptions
from tools.SVG_Sketch import Sketch

data_root = sys.argv[1]
save_root = sys.argv[2]
phase = sys.argv[3]
start_index = int(sys.argv[4])
end_index = int(sys.argv[5])  # not included in the range

# 参数指定
method = "SED"  # 边缘检测方法
load_size = None  # 图片加载宽高。None表示不做resize处理

root_path = f"{data_root}/{phase}/"
save_root_path_edge = f"{save_root}/{phase}/"
os.makedirs(save_root_path_edge, exist_ok=True)

files = os.listdir(root_path)
files.sort()
detector = EdgeDetector(method=method)
index = start_index
if end_index == -1: end_index = len(files)
for file in tqdm(files[start_index:end_index]):
    save_path = os.path.join(save_root_path_edge, file)
    name = file.split(".")[0]
    threshold = [30, 50]
    image = cv2.imread(os.path.join(root_path, file))[:, :, ::-1]
    # 边缘检测
    edge = detector.detect(image, nms=True, binary=True, name=name, threshold=threshold)
    cv2.imwrite(save_path, 255 - edge)  # 线条以黑色显示
    index += 1
