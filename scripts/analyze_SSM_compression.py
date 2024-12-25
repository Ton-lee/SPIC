"""测试通用图像压缩算法在语义分割图压缩的性能"""
import os.path
import cv2
import tqdm
import sys
sys.path.append("/home/Users/dqy/Projects/SPIC")
from tools.base_utils import suffix, compress_image, get_mIoU

dataset = "ADE20k"

if dataset == "Cityscapes":
    # 对 Cityscapes，采用以下路径设置
    root_folder = "/home/Users/dqy/Dataset/Cityscapes/SSM_compressed/val/"  # 保存结果的路径
    save_name = ""  # 保存结果的子目录
    # label_suffix = "(extended)"
    label_suffix = ""
    if label_suffix == "":
        gt_folder = f"/home/Users/dqy/Dataset/Cityscapes/SSM_compressed/val/labels_ori/"
    else:
        gt_folder = f"/home/Users/dqy/Projects/SPIC/Result/pretrained/ori/labels/"
    test_total = 500
elif dataset == "ADE20k":
    # 对 ADE20k，采用以下路径设置
    root_folder = "/home/Users/dqy/Projects/SPIC/Result_ADE20k/"
    save_name = "annotations_compressed"
    gt_folder = "/home/Users/dqy/Dataset/ADE20K/ADEChallengeData2016/annotations/validation/"  # 保存结果的路径
    label_suffix = ""
    test_total = 2000
else:
    raise NotImplementedError(f"Dataset {dataset} not implemented!")
# 路径设置结束

compressed_root = f"{root_folder}/{save_name}"
decompressed_root = f"{root_folder}/{save_name}"
test_factor = {
    "jpeg": list(range(0, 101, 5)),
    "png": list(range(0, 10, 1)),
    "webp": list(range(0, 100, 5)),
    "bpg": list(range(0, 52, 3)),
    "flif": list(range(0, 101, 10)),
    "j2k": [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200],
    "cheng2020-anchor": [1, 2, 3, 4, 5, 6],
    "cheng2020-attn": [1, 2, 3, 4, 5, 6],
    "hific": ["hific-hi", "hific-mi", "hific-lo"],
    "PostFilter-jpeg": list(range(0, 101, 5)),
    "PostFilter-webp": list(range(0, 101, 5)),
}

print("method\tfactor\tbpp\tmIoU")
# for method in ["jpeg", "png", "webp", "bpg", "flif", "j2k", "cheng2020-anchor", "cheng2020-attn", "hific", "PostFilter-jpeg", "PostFilter-webp"]:
# for method in ["jpeg", "png", "webp", "bpg", "flif", "j2k", "cheng2020-anchor", "cheng2020-attn", "hific"]:
# for method in ["cheng2020-anchor", "cheng2020-attn", "hific"]:
for method in ["jpeg", "png", "webp", "bpg", "flif", "j2k"]:
# for method in ["PostFilter-jpeg", "PostFilter-webp"]:
    for factor in test_factor[method]:
        compressed_folder = os.path.join(compressed_root, f"labels{label_suffix}-{method}", f"compressed-{factor}")
        decompressed_folder = os.path.join(decompressed_root, f"labels{label_suffix}-{method}", f"decompressed-{factor}")
        if not os.path.exists(compressed_folder):
            os.makedirs(compressed_folder)
        if not os.path.exists(decompressed_folder):
            os.makedirs(decompressed_folder)
        mIoU_list = []
        files = sorted(os.listdir(gt_folder))
        files = [file for file in files if file.endswith(".png")]
        for image_file in tqdm.tqdm(files[:test_total]):
            ori_file = os.path.join(gt_folder, image_file)
            compressed_file = image_file.replace(".png", suffix[method])
            decompressed_file = image_file
            ssm_ori = cv2.imread(os.path.join(gt_folder, ori_file))[:, :, 0]
            if method not in ["cheng2020-anchor", "cheng2020-attn", "hific", "PostFilter-jpeg", "PostFilter-webp"]:
                ssm_compressed = compress_image(
                    image=ssm_ori,
                    method=method,
                    compress_factor=factor,
                    compressed_path=os.path.join(compressed_folder, compressed_file),
                    decompressed_path=os.path.join(decompressed_folder, decompressed_file),
                    save_compressed=True,
                    save_decompressed=True,
                    return_array=True
                )[:, :, 0]
            else:  # 三种深度压缩方法直接读取压缩后的文件
                ssm_compressed = cv2.imread(os.path.join(decompressed_folder, decompressed_file))[:, :, 0]
            if label_suffix == "(extended)":
                ssm_ori = ssm_ori // 7
                ssm_ori[ssm_ori == 36] = 19
                ssm_compressed = ssm_compressed // 7
                ssm_compressed[ssm_compressed == 36] = 19
            mIoU = get_mIoU(ssm_ori, ssm_compressed)
            mIoU_list.append(mIoU)
        bpp_list = []
        if method not in ["PostFilter-jpeg", "PostFilter-webp"]:
            for image_file in sorted(os.listdir(compressed_folder))[:test_total]:
                file = os.path.join(compressed_folder, image_file)
                size = os.path.getsize(file)
                bpp_list.append(size * 8 / 256 / 512)
        else:  # 两种方法没有存储编码后的文件而是存储日志
            for image_file in sorted(os.listdir(compressed_folder))[:test_total]:
                file = os.path.join(compressed_folder, image_file)
                with open(file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    if "compressed bpp" in line:
                        bpp = float(line.strip().split(" ")[-1])
                        bpp_list.append(bpp)
                        break
        tqdm.tqdm.write(f"{method}\t{factor}\t{sum(bpp_list)/test_total:.4f}\t{sum(mIoU_list)/test_total:.4f}")
