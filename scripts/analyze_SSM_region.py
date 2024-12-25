"""测试基于区域表征与多边形拟合的语义分割图压缩算法性能"""
import os
import cv2
import imageio
import tqdm
import sys
sys.path.append("/home/Users/dqy/Projects/SPIC/")
from tools.sketch_func import compress_ssm
from tools.base_utils import ade20k_color2label, ade20k_label2color, ade20k_semantic_orders
from tools.base_utils import get_ade20k_ssm_paths
from tools.base_utils import get_cityscapes_ssm_paths, cityscapes_semantic_orders

test_dataset = "cityscapes"
test_size = [512, 256]
test_factor = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
save_root = "/home/Users/dqy/Dataset/Cityscapes/SSM_compressed/val/labels-region"
print("method\tfactor\tbpp\tmIoU")
if test_dataset == "cityscapes":
    files = sorted(get_cityscapes_ssm_paths())
    class_count = 19
    test_total = 500
    ignore_label = 255
    semantic_priorities = cityscapes_semantic_orders
elif test_dataset == "ade20k":
    files = sorted(get_ade20k_ssm_paths())
    class_count = 150
    test_total = 2000
    ignore_label = 0
    semantic_priorities = ade20k_semantic_orders
else:
    raise ValueError(f"Unsupported dataset {test_dataset}")
for factor in test_factor:
    ssm_save_folder = os.path.join(save_root, f"decompressed-{factor}")
    os.makedirs(ssm_save_folder, exist_ok=True)
    bpp_list, mIoU_list = [], []
    for image_file in tqdm.tqdm(files[:test_total]):
        # if "frankfurt_000000_005543" not in image_file:
        #     continue  # 关注"frankfurt_000000_005543"的编码过程
        ssm_ori = imageio.v2.imread(image_file)
        ssm_ori = cv2.resize(ssm_ori, test_size, interpolation=cv2.INTER_NEAREST)
        bpp, mIoU, ssm_recovered = compress_ssm(ssm_ori, factor,
                                    ignore_label=ignore_label,
                                    class_count=class_count,
                                    semantic_priorities=None)
        image_name = os.path.basename(image_file)
        save_path = os.path.join(ssm_save_folder, image_name)
        # cv2.imwrite(save_path, ssm_recovered)
        bpp_list.append(bpp)
        mIoU_list.append(mIoU)
    tqdm.tqdm.write(f"Region\t{factor}\t{sum(bpp_list) / test_total:.4f}\t{sum(mIoU_list) / test_total:.4f}")
