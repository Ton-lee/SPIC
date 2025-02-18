# 对轮廓图像进行压缩，通过素描图算法实现数据量的减小
import sys
import os
import numpy as np
import tqdm
sys.path.append("/home/Users/dqy/Projects/SPIC/")
import cv2
from tools.sketch_func import get_sketch
from tools.sketch_utils import save_branch_compressed
from tools.utils import save_graph_compressed, rearrange_graph


edge_root = "/home/Users/dqy/Dataset/Cityscapes/edges/"
save_root = "/home/Users/dqy/Dataset/Cityscapes/edges/"
phases = ["train", "val"]
size1 = (1024, 512)
size2 = (512, 256)
thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]


def compress_sketch(edge, threshold, code_path, decode_path):
    H, W = edge.shape
    points, connections, branch_object, edge_sketch = get_sketch(edge,
                                                                 neighbor=4,
                                                                 threshold_fit=threshold,
                                                                 return_recorder=True,
                                                                 threshold_length=1,
                                                                 valid_shift=[0, 1, 3],
                                                                 abort_stick=False)
    # 以分支抽象的格式保存压缩后的素描图
    # save_path = code_path
    # save_branch_compressed(branch_object, save_path, [H, W])
    # bpp = os.path.getsize(save_path) * 8 / H / W
    # 以图格式保存压缩后的素描图
    graph = {"positions": points, "connections": connections}
    sketch_graph_rearranged = rearrange_graph(graph)
    save_graph_compressed(sketch_graph_rearranged, code_path, [H, W])
    bpp = os.path.getsize(code_path) * 8 / H / W
    cv2.imwrite(decode_path, (1 - (edge_sketch.sum(axis=2)>0).astype('uint8')) * 255)
    return bpp


def main():
    for phase in phases:
        phase_folder = os.path.join(edge_root, phase, "ori")
        save_folder = os.path.join(save_root, phase)
        bpp_list = {}
        pbar = tqdm.tqdm(total=2975 if phase == "train" else 500, desc=f"Processing {phase}")
        edge_files = sorted(os.listdir(phase_folder))
        for edge_file in edge_files:
            edge_file_path = os.path.join(phase_folder, edge_file)
            edge = cv2.imread(edge_file_path)[:, :, 0]
            edge = (edge == 255).astype("uint8")
            # 2. 按指定阈值进行素描图压缩
            edge = 1 - edge
            for threshold in thresholds:
                code_path = os.path.join(save_folder, f"compressed_th{threshold}", edge_file.replace(".png", ".bin"))
                decode_path = os.path.join(save_folder, f"decompressed_th{threshold}", edge_file)
                os.makedirs(os.path.dirname(code_path), exist_ok=True)
                os.makedirs(os.path.dirname(decode_path), exist_ok=True)
                bpp = compress_sketch(edge, threshold, code_path, decode_path)
                if threshold not in bpp_list:
                    bpp_list[threshold] = []
                bpp_list[threshold].append(bpp)
            pbar.update()
        print("phase\tthreshold\tmbpp")
        for threshold in thresholds:
            mean_bpp = sum(bpp_list[threshold]) / len(bpp_list[threshold])
            print(f"{phase}\t{threshold}\t{mean_bpp:.4f}")

main()