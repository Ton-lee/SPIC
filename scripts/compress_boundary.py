# 对轮廓图像进行压缩，通过素描图算法实现数据量的减小
import sys
import os
import tqdm
sys.path.append("/home/Users/dqy/Projects/SPIC/")
import cv2
from tools.sketch_func import get_sketch
from tools.sketch_utils import save_branch_compressed
from tools.utils import save_graph_compressed, rearrange_graph, save_graph_compressed_bounding


boundary_root = "/home/Users/dqy/Dataset/Cityscapes/gtFine/"
save_root = "/home/Users/dqy/Dataset/Cityscapes/boundary_compressed/"
phases = ["train", "val"]
size1 = (1024, 512)
size2 = (512, 256)
thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]


def compress_sketch(boundary, threshold, code_path, decode_path):
    H, W = boundary.shape
    points, connections, branch_object, edge_sketch = get_sketch(boundary,
                                                                 neighbor=4,
                                                                 threshold_fit=threshold,
                                                                 return_recorder=True,
                                                                 threshold_length=1,
                                                                 valid_shift=[0, 1, 3],
                                                                 abort_stick=True,
                                                                 merge_neighbor=True)
    # 以分支抽象的格式保存压缩后的素描图
    # save_path = code_path
    # save_branch_compressed(branch_object, save_path, [H, W])
    # bpp = os.path.getsize(save_path) * 8 / H / W
    # 以图格式保存压缩后的素描图
    graph = {"positions": points, "connections": connections}
    sketch_graph_rearranged = rearrange_graph(graph)
    # save_graph_compressed(sketch_graph_rearranged, code_path, [H, W])
    save_graph_compressed_bounding(sketch_graph_rearranged, code_path, [H, W], branch_object)
    bpp = os.path.getsize(code_path) * 8 / H / W
    cv2.imwrite(decode_path, (1 - (edge_sketch.sum(axis=2)>0).astype('uint8')) * 255)
    return bpp


def main():
    for phase in phases:
        phase_folder = os.path.join(boundary_root, phase)
        cities = sorted(os.listdir(phase_folder))
        save_folder = os.path.join(save_root, phase)
        bpp_list = {}
        pbar = tqdm.tqdm(total=2975 if phase == "train" else 500, desc=f"Processing {phase}")
        for city in cities:
            city_folder = os.path.join(phase_folder, city)
            files = sorted(os.listdir(city_folder))
            boundary_files = [file for file in files if file.endswith("_instanceBoundary.png")]
            for boundary_file in boundary_files:
                boundary_file_path = os.path.join(city_folder, boundary_file)
                boundary = cv2.imread(boundary_file_path)[:, :, 0]
                boundary = cv2.resize(boundary, size1)
                boundary = cv2.resize(boundary, size2)
                boundary = (boundary == 255).astype("uint8")
                # 1. 保存指定尺寸的原始轮廓图
                save_path = os.path.join(save_folder, "ori", boundary_file)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, boundary * 255)
                # 2. 按指定阈值进行素描图压缩
                edge = 1 - boundary
                for threshold in thresholds:
                    code_path = os.path.join(save_folder, f"compressed_th{threshold}", boundary_file.replace(".png", ".bin"))
                    decode_path = os.path.join(save_folder, f"decompressed_th{threshold}", boundary_file)
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
