"""基于区域分割与多边形拟合的语义分割图压缩算法测试"""
import os
import numpy as np
import imageio
from tools.base_utils import ssm2boundary, get_mIoU, get_ssm_colored_cityscapes, cityscapes_semantic_orders
from tools.base_utils import get_semantic_orders
from tools.sketch_utils import save_branch_compressed, preprocess_ssm
from tools.sketch_func import get_sketch
from tools.utils import save_graph_compressed, rearrange_graph
import cv2


def test_SSM_compression(ssm: np.ndarray, threshold=2):
    """调用基本函数对语义分割图进行压缩"""
    H, W = ssm.shape
    # 获取语义优先级
    semantic_order = cityscapes_semantic_orders
    semantic_order = get_semantic_orders(ssm, ignore_label=19)
    # 预处理语义分割图，使斜连但不连通的像素连通
    ssm_preprocessed = preprocess_ssm(ssm, semantic_order, ignore_label=19)
    # 根据语义分割图获取语义边界轮廓
    boundary = ssm2boundary(ssm_preprocessed, semantic_order)
    # 对语义边界轮廓进行分支抽象并执行直线段拟合。部分分支线对区域划分起到关键作用，因此不能根据分支长度进行舍弃
    points, connections, branch_object, edge_sketch = get_sketch(boundary,
                                                                 neighbor=4,
                                                                 threshold_fit=threshold,
                                                                 return_recorder=True,
                                                                 threshold_length=1,
                                                                 valid_shift=[1, 3],
                                                                 abort_stick=True)
    # 以分支抽象的格式保存压缩后的素描图
    save_path = "/home/Users/dqy/Projects/SPIC/temp/compressed_branch.bin"
    save_branch_compressed(branch_object, save_path, [H, W])
    bpp_branch = os.path.getsize(save_path) * 8 / H / W
    # 以图格式保存压缩后的素描图
    save_path = "/home/Users/dqy/Projects/SPIC/temp/compressed_sketch.bin"
    graph = {"positions": points, "connections": connections}
    sketch_graph_rearranged = rearrange_graph(graph)
    save_graph_compressed(sketch_graph_rearranged, save_path, [H, W])
    bpp_sketch = os.path.getsize(save_path) * 8 / H / W
    # 编码区域信息，包括围成区域的分支序号和区域的标签
    assigned_index = branch_object.arrange_branch()  # 为各个分支分配唯一的序号
    ssm_regions = branch_object.generate_regions_connection(ssm, assigned_index)
    bpp_region = sum([len(region[0]) * np.ceil(np.log2(branch_object.count)) \
                      + np.ceil(np.log2(19)) for region in ssm_regions]) / H / W
    # 恢复语义分割图
    ssm_recovered = branch_object.recover_ssm(ssm_regions, assigned_index, H, W, ignore_label=19,
                                              semantic_orders=semantic_order)
    ssm_colored = get_ssm_colored_cityscapes(ssm_recovered)
    # 计算语义指标
    mIoU = get_mIoU(ssm, ssm_recovered, labels=semantic_order, ignore_label=19)
    print(f"BPP: {bpp_sketch + bpp_region} \t\t mIoU: {mIoU}")


def main():
    # 读取语义分割图并做标签预处理
    ssm_path = ("/home/Users/dqy/Dataset/Cityscapes/"
                "gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png")
    ssm = imageio.v2.imread(ssm_path)  # (1024x2048)
    ssm[ssm == 255] = 19
    ssm_resample = cv2.resize(ssm, [512, 256], interpolation=cv2.INTER_NEAREST)
    test_SSM_compression(ssm_resample)


if __name__ == "__main__":
    main()
