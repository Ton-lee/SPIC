"""基本工具，包括图像压缩命令等"""
import collections
import os
import cv2
import numpy as np
import time
import shutil
from openjpeg import encode, decode
from typing import Union
from shapely import Polygon, MultiPoint
from matplotlib.path import Path
from skimage import measure

temp_folder = "/home/Users/dqy/Projects/SPIC/temp"
suffix = {
    "jpeg": '.jpeg',
    "png": '.png',
    "webp": '.webp',
    "bpg": '.bpg',
    "flif": '.flif',
    "j2k": '.j2k',
    "cheng2020-anchor": '.bin',
    "cheng2020-attn": '.bin',
    "hific": '.tfci',
    "PostFilter-jpeg": ".log",
    "PostFilter-webp": ".log"
}


def compress_image(image: Union[np.ndarray, str],
                   method: str,
                   compress_factor: int,
                   compressed_path: str = None,
                   decompressed_path: str = None,
                   save_compressed: bool = False,
                   save_decompressed: bool = False,
                   return_array: bool = True):
    """图像压缩编码"""
    # 判断参数合法性
    if isinstance(image, str):
        assert os.path.isfile(image), f"Image not found: {image}"
        image: np.ndarray = cv2.imread(image)  # BGR image (H,W,3)
    assert (not save_compressed or compressed_path is not None
            and os.path.isdir(os.path.dirname(compressed_path))), "Compressed path not assigned!"
    assert (not save_decompressed or decompressed_path is not None
            and os.path.isdir(os.path.dirname(decompressed_path))), "Decompressed path not assigned!"
    # 获取临时路径
    name_stamp = method
    time_stamp = time.time()
    process_stamp = os.getpid()
    original_save_name = f"{name_stamp}-{process_stamp}#{time_stamp}_original.png"
    compressed_save_name = f"{name_stamp}-{process_stamp}#{time_stamp}_compressed{suffix[method.lower()]}"
    decompressed_save_name = f"{name_stamp}-{process_stamp}#{time_stamp}_decompressed.png"
    temp_abs_path_original = os.path.join(temp_folder, original_save_name)
    temp_abs_path_compressed = os.path.join(temp_folder, compressed_save_name)
    temp_abs_path_decompressed = os.path.join(temp_folder, decompressed_save_name)
    image_compressed = None  # RGB image
    # 执行压缩
    if method.lower() in ["jpeg", "png", "webp"]:
        if method.lower() == 'jpeg':  # JPEG 压缩
            assert 0 <= compress_factor <= 100, (f"Compression factor for JPEG must be 0-100 "
                                                 f"(higher for better, got {compress_factor})")
            # 编码
            cv2.imwrite(temp_abs_path_compressed, image, [int(cv2.IMWRITE_JPEG_QUALITY), compress_factor])
        if method.lower() == 'png':  # PNG
            assert 0 <= compress_factor <= 9, (f"Compression factor for PNG must be 0-9 "
                                               f"(lower for better, got {compress_factor})")
            # 编码
            cv2.imwrite(temp_abs_path_compressed, image, [int(cv2.IMWRITE_PNG_COMPRESSION), compress_factor])
        if method.lower() == 'webp':  # WEBP
            assert 0 <= compress_factor <= 100, (f"Compression factor for WEBP must be 0-100 "
                                                 f"(higher for better, got {compress_factor})")
            cv2.imwrite(temp_abs_path_compressed, image, [int(cv2.IMWRITE_WEBP_QUALITY), compress_factor])
        # 解码
        image_compressed = cv2.imread(temp_abs_path_compressed)[:, :, ::-1]
        cv2.imwrite(temp_abs_path_decompressed, image_compressed[:, :, ::-1])
    if method.lower() == 'bpg':  # BPG
        assert 0 <= compress_factor <= 51, (f"Compression factor for BPG must be 0-51 "
                                            f"(lower for better, got {compress_factor})")
        cv2.imwrite(temp_abs_path_original, image)
        command = (f"bash '/home/Users/dqy/scripts/compress_BPG.sh' "
                   f"'{temp_abs_path_original}' "
                   f"{compress_factor} "
                   f"'{temp_abs_path_compressed}' "
                   f"'{temp_abs_path_decompressed}'")
        os.system(command)
        image_compressed = cv2.imread(temp_abs_path_decompressed)[:, :, ::-1]
    if method.lower() == 'flif':  # FLIF
        assert 0 <= compress_factor <= 100, (f"Compression factor for FLIF must be 0-100 "
                                             f"(higher for better, got {compress_factor})")
        cv2.imwrite(temp_abs_path_original, image)
        command = (f"bash '/home/Users/dqy/scripts/compress_FLIF.sh' "
                   f"'{temp_abs_path_original}' "
                   f"{compress_factor} "
                   f"'{temp_abs_path_compressed}' "
                   f"'{temp_abs_path_decompressed}'")
        os.system(command)
        image_compressed = cv2.imread(temp_abs_path_decompressed)[:, :, ::-1]
    if method.lower() == 'j2k':  # J2K
        assert 1 <= compress_factor, (f"Compression ratio for J2K must be 1- "
                                      f"(1 for best, got {compress_factor})")
        codes = encode(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), compression_ratios=[compress_factor])
        with open(temp_abs_path_compressed, 'wb') as f:
            f.write(codes)
        image_compressed = decode(codes)  # RGB
        cv2.imwrite(temp_abs_path_decompressed, cv2.cvtColor(image_compressed, cv2.COLOR_RGB2BGR))
    # 处理保存与返回信息
    if save_compressed:
        shutil.copy(temp_abs_path_compressed, compressed_path)
    if save_decompressed:
        shutil.copy(temp_abs_path_decompressed, decompressed_path)
    # 清除临时文件
    if os.path.exists(temp_abs_path_original):
        os.remove(temp_abs_path_original)
    if os.path.exists(temp_abs_path_compressed):
        os.remove(temp_abs_path_compressed)
    if os.path.exists(temp_abs_path_decompressed):
        os.remove(temp_abs_path_decompressed)
    if return_array:
        return image_compressed
    else:
        return None


def get_mIoU(label_gt: np.ndarray, label_pred: np.ndarray, labels=None, ignore_label=19):
    """
    计算两个语义分割图的 mIoU
    """
    if labels is None:
        labels = list(set(label_gt.flatten().tolist()) | set(label_pred.flatten().tolist()))
    IoU = []
    for label_index in labels:
        union = np.bitwise_or(label_gt == label_index, label_pred == label_index) * (label_gt != ignore_label)
        intersection = np.bitwise_and(label_gt == label_index, label_pred == label_index) * (label_gt != ignore_label)
        union = np.sum(union)
        intersection = np.sum(intersection)
        if union == 0:
            continue
        IoU.append(intersection / union)
    return np.mean(IoU)


def pad_same(image: np.ndarray, pad_width=1) -> np.ndarray:
    """以与边界值相同的值对图像进行扩展"""
    map_pad = np.pad(image, pad_width=pad_width, mode='edge')
    return map_pad


def ssm2boundary_seed(ssm: np.ndarray) -> np.ndarray:
    """
    通过种子生长算法获取语义分割图的边界线
    """
    # 具体算法为，初始化边界蒙版为 0，采用区域生长的方法，在边界蒙版上以某一非边界点为种子向8邻域生长。生长规则为：
    # 1. 若种子为非边界点，且与其根类别相同，则将8邻域设置为种子；
    # 2. 若种子为边界点，则该种子不生长；
    # 3. 若种子为非边界点，但与其根类别不同，则将该种子修正为边界点。
    pass


def ssm2boundary(ssm: np.ndarray, semantic_orders=None, ignore_label=19) -> (np.ndarray, np.ndarray):
    """由语义分割图获取边界，采用迭代式边界检测的方法"""
    if semantic_orders is None:
        semantic_orders = set(ssm.flatten().tolist())
    # 具体算法为：遍历每一个标签类别并设置边界，设置边界时考虑已经存在的边界的影响
    assert len(ssm.shape) == 2, "dim of ssm.shape must be 2"
    H, W = ssm.shape
    boundary = np.zeros_like(ssm)
    for semantic_label in semantic_orders:  # 考察一个区域类别
        if semantic_label not in ssm:  # 只考虑存在于语义分割图中的标签
            continue
        inner_map = ssm == semantic_label  # 当前考察标签对应区域
        outer_map = ssm != semantic_label  # 考察标签以外的区域
        # 找到考察区域外，与考察标签相邻的像素点
        inner_map_pad = pad_same(inner_map, 1)
        boundary_pad = pad_same(boundary, 1, )
        shifts_x, shifts_y = (0, 0, 0, 1, 1, 2, 2, 2), (0, 1, 2, 0, 2, 0, 1, 2)
        candidate_map = np.zeros_like(ssm)  # 区域外与考察标签相邻的像素点为 1
        for shift_x, shift_y in zip(shifts_x, shifts_y):
            neighbor_map = inner_map_pad[shift_y: shift_y + H, shift_x: shift_x + W]
            neighbor_boundary = boundary_pad[shift_y: shift_y + H, shift_x: shift_x + W]
            candidate_map += outer_map * neighbor_map * (1 - neighbor_boundary) * (1 - boundary)  # 区域外的非边界点，且邻点为区域内的非边界点
        boundary[candidate_map > 0] = 1
    boundary[:, [0, -1]] = np.logical_or(boundary[:, [0, -1]], ssm[:, [0, -1]] != ignore_label)
    boundary[[0, -1], :] = np.logical_or(boundary[[0, -1], :], ssm[[0, -1], :] != ignore_label)
    return boundary


def get_inner_points(points):
    """获得多边形内部（包括边界）的点"""
    p = Polygon(points)
    xmin, ymin, xmax, ymax = p.bounds
    x = np.arange(np.floor(xmin), np.ceil(xmax) + 1)
    y = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
    grids = MultiPoint(np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))]))
    result: MultiPoint = grids.intersection(p)
    return [(int(p.x), int(p.y)) for p in result.geoms]


def get_inner_mask(polygon):
    """获得多边形内部点的蒙版"""
    min_l, min_c = np.min(polygon, axis=0)
    max_l, max_c = np.max(polygon, axis=0)
    h, w = max_l - min_l + 1, max_c - min_c + 1
    local_mask = np.zeros([h, w])
    path = Path(polygon)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    points = np.vstack((y.flatten(), x.flatten())).T + np.array([min_l, min_c])
    mask = path.contains_points(points, radius=0.5).reshape((h, w))  # 设置扩展半径使其包含路径上的点
    return mask, min_l, min_c


def neighbored(p1, p2, neighbor=8):
    """两个点是否相邻或相同"""
    if neighbor == 8:
        return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) <= 1
    else:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) <= 1


def get_cityscapes_ssm_paths():
    """获取 Cityscapes 数据集 val 子集的所有语义分割图的路径"""
    root_folder = "/home/Users/dqy/Dataset/Cityscapes/gtFine/val/"
    # 获取文件夹中所有 labelTrainIds.png 为后缀的文件绝对路径
    all_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith("labelTrainIds.png"):
                all_paths.append(os.path.join(root, file))
    return all_paths


def get_ade20k_ssm_paths():
    """获取 ADE20k 数据集 val 子集的所有语义分割图的路径"""
    root_path = "/home/Users/dqy/Dataset/ADE20K/ADEChallengeData2016/annotations/validation/"
    files = sorted(os.listdir(root_path))
    return [os.path.join(root_path, file) for file in files]


def parse_cityscapes_labels():
    cityscapes_colors_map = {
        0: (128, 64, 128),  # road
        1: (244, 35, 232),  # sidewalk
        2: (70, 70, 70),  # building
        3: (102, 102, 156),  # wall
        4: (190, 153, 153),  # fence
        5: (153, 153, 153),  # pole
        6: (250, 170, 30),  # traffic light
        7: (220, 220, 0),  # traffic sign
        8: (107, 142, 35),  # vegetation
        9: (152, 251, 152),  # terrain
        10: (70, 130, 180),  # sky
        11: (220, 20, 60),  # person
        12: (255, 0, 0),  # rider
        13: (0, 0, 142),  # car
        14: (0, 0, 70),  # truck
        15: (0, 60, 100),  # bus
        16: (0, 80, 100),  # train
        17: (0, 0, 230),  # motorcycle
        18: (119, 11, 32)  # bicycle
    }
    color2label = {}
    for key, val in color2label.items():
        color2label[val] = key
    cityscapes_semantic_orders = [
        6, 7, 4, 5,  # 交通灯、交通标志、篱笆和杆等细小物
        11, 12, 18, 17,  # 行人、骑者、自行车与摩托车等小前景
        13, 14, 15, 16,  # 汽车、卡车、公交、火车等交通工具前景
        0, 1, 2, 3, 8, 9,  # 路、墙、建筑等中景
        10  # 天空，大背景
    ]  # 遍历类别的顺序，高优先级的顺序倾向于保留区域内的所有像素点，保证区域不收缩（如路灯）
    return cityscapes_colors_map, color2label, cityscapes_semantic_orders

def parse_ade20k_labels():
    """解析 ADE20k 的标签结构，包括获取类别到颜色、颜色到类别字典和类别优先级列表"""
    label2color = {}
    color2label = {}
    semantic_orders = [i+1 for i in range(150)]
    return label2color, color2label, semantic_orders


cityscapes_label2color, cityscapes_color2label, cityscapes_semantic_orders = parse_cityscapes_labels()
ade20k_label2color, ade20k_color2label, ade20k_semantic_orders = parse_ade20k_labels()


def get_ssm_colored_cityscapes(ssm):
    """对语义分割图上色"""
    H, W = ssm.shape[:2]
    color_image = np.zeros((H, W, 3), dtype=np.uint8)
    for key, color in cityscapes_label2color.items():
        mask = ssm == key
        color_image[mask] = color
    return color_image


def get_semantic_orders(ssm: np.ndarray, ignore_label=None):
    """根据像素数占比决定语义优先级，优先保证小目标的完整性"""
    if ignore_label is None:
        ignore_label = -1
    counter = collections.Counter(ssm.flatten().tolist())
    frequencies = counter.most_common()
    semantic_order = [frequency[0] for frequency in frequencies if frequency[0] != ignore_label]
    return semantic_order[::-1]  # 低频次标签具有更高优先级


def sample_color(image, boundary, alpha=1.0):
    """
    从给定的图像中采样像素点作为图像的稀疏表征
    :param image: RGB 图像，(H,W,3)
    :param boundary: 边界图，(H,W)，0 表示非边界、1表示边界像素点，边界四邻域连通
    :param alpha: 码率损失与 MSE 损失的权重，优化目标为码率损失+alpha*MSE，更高的alpha表示更高的采样细节
    :return: 采样点坐标和颜色
    """
    # 根据边界像素点获取图像区域连通性
    labeled_image, num_labels = measure.label(1 - boundary, connectivity=1)
    # 考察每个连通区域，以像素平均 RGB 值作为初始颜色采样点
    sample_points = []
    for label in range(1, num_labels + 1):
        mask = labeled_image == label
        region_points = np.argwhere(mask)
        region_color = np.mean(image[region_points[:, 0], region_points[:, 1], :], axis=0)
        sample_points.append((region_points[0, 0], region_points[0, 1], tuple(region_color.astype(int))))

