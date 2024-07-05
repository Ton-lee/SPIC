"""基本工具，包括图像压缩命令等"""

import os
import cv2
import numpy as np
import time
import shutil
from openjpeg import encode, decode
from typing import Union

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


def get_mIoU(label_gt: np.ndarray, label_pred: np.ndarray):
    """
    计算两个语义分割图的 mIoU，其中 255 表示忽略的区域
    """
    IoU = []
    label_pred[label_pred > 18] = 255
    for label_index in range(19):
        union = np.bitwise_or(label_gt == label_index, label_pred == label_index) * (label_gt != 255)
        intersection = np.bitwise_and(label_gt == label_index, label_pred == label_index) * (label_gt != 255)
        union = np.sum(union)
        intersection = np.sum(intersection)
        if union == 0:
            continue
        IoU.append(intersection / union)
    return np.mean(IoU)
