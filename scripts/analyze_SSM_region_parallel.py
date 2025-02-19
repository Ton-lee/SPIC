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
import os
import cv2
import imageio.v2
import tqdm
import concurrent.futures

test_dataset = "ade20k"
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


# 假设你已经定义了以下参数:
# save_root, test_factor, files, test_total, test_size, ignore_label, class_count, compress_ssm

def process_image(image_file, factor, test_size, ignore_label, class_count, test_total, ssm_save_folder):
    # print(f"Processing: {image_file}")
    ssm_ori = imageio.v2.imread(image_file)
    ssm_ori = cv2.resize(ssm_ori, test_size, interpolation=cv2.INTER_NEAREST)
    bpp, mIoU, ssm_recovered = compress_ssm(ssm_ori, factor,
                                            ignore_label=ignore_label,
                                            class_count=class_count,
                                            semantic_priorities=None)

    image_name = os.path.basename(image_file)
    save_path = os.path.join(ssm_save_folder, image_name)
    # if ssm_recovered is not None:
    #     cv2.imwrite(save_path, ssm_recovered)  # If you want to save the image
    # print(f"Processed: {image_file}")
    return bpp, mIoU


def process_factor(factor, files, test_total, test_size, ignore_label, class_count, save_root):
    ssm_save_folder = os.path.join(save_root, f"decompressed-{factor}")
    os.makedirs(ssm_save_folder, exist_ok=True)
    bpp_list, mIoU_list = [], []

    with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
        # Submit tasks to the executor pool
        future_to_image = {
            executor.submit(process_image, image_file, factor, test_size, ignore_label, class_count, test_total,
                            ssm_save_folder): image_file
            for image_file in files[:test_total]
        }

        # Collect the results as they complete
        for future in concurrent.futures.as_completed(future_to_image):
            bpp, mIoU = future.result()
            bpp_list.append(bpp)
            mIoU_list.append(mIoU)

    bpp_list = [bpp for bpp in bpp_list if bpp != -1]
    mIoU_list = [mIoU for mIoU in mIoU_list if mIoU != -1]
    avg_bpp = sum(bpp_list) / len(bpp_list)
    avg_mIoU = sum(mIoU_list) / len(mIoU_list)
    tqdm.tqdm.write(f"Region\t{factor}\t{avg_bpp:.4f}\t{avg_mIoU:.4f}")


# Main loop to process all factors
for factor in test_factor:
    process_factor(factor, files, test_total, test_size, ignore_label, class_count, save_root)
