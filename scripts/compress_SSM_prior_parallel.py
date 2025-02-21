"""测试基于区域表征与多边形拟合的语义分割图压缩算法性能"""
import os
import cv2
import imageio
import tqdm
import sys
sys.path.append("/home/Users/dqy/Projects/SPIC/")
from tools.sketch_func import compress_ssm_prior
from tools.base_utils import ade20k_color2label, ade20k_label2color, ade20k_semantic_orders
from tools.base_utils import get_ade20k_ssm_paths
from tools.base_utils import get_cityscapes_ssm_paths, cityscapes_semantic_orders
import os
import cv2
import imageio.v2
import tqdm
import concurrent.futures
import numpy as np

test_dataset = "cityscapes"
test_phase = "train"
test_size = [512, 256]
test_factor = [1.0, 1.5, 2.0, 2.5, 3.0]
print("method\tfactor\tlevel_layout\tlevel_boundary\tbpp\tmIoU")
if test_dataset == "cityscapes":
    save_root = "/home/Users/dqy/Dataset/Cityscapes/ssm_compressed(prior)"
    files = sorted(get_cityscapes_ssm_paths(phase=test_phase))
    class_count = 19
    test_total = 500 if test_phase == "val" else 2975
    ignore_label = 255
    semantic_priorities = cityscapes_semantic_orders
    boxes_folder = f"/home/Users/dqy/Dataset/Cityscapes/format_yolo/labels/{test_phase}/"
elif test_dataset == "ade20k":
    save_root = "/home/Users/dqy/Dataset/ADE20k/ssm_compressed(prior)"
    files = sorted(get_ade20k_ssm_paths(phase=test_phase))
    class_count = 150
    test_total = 2000 if test_phase == "val" else 20210
    ignore_label = 0
    semantic_priorities = ade20k_semantic_orders
    boxes_folder = ""
else:
    raise ValueError(f"Unsupported dataset {test_dataset}")


# 假设你已经定义了以下参数:
# save_root, test_factor, files, test_total, test_size, ignore_label, class_count, compress_ssm

def process_image(image_file, factor, test_size, ignore_label, class_count, test_total, ssm_save_folder, layout_level=19, boundary_level=19):
    # print(f"Processing: {image_file}")
    image_name = os.path.basename(image_file)  # aachen_000000_000019_gtFine_boundingBox.txt
    boxes_file = os.path.join(boxes_folder, image_name.replace("labelTrainIds.png", "boundingBox.txt"))
    boxes = np.loadtxt(boxes_file)
    box_labels = boxes[:, 0].astype("int")
    ssm_ori = imageio.v2.imread(image_file)
    ssm_ori = cv2.resize(ssm_ori, test_size, interpolation=cv2.INTER_NEAREST)
    bpp, mIoU, ssm_recovered = compress_ssm_prior(ssm_ori, factor,
                                                  ignore_label=ignore_label,
                                                  class_count=class_count,
                                                  semantic_priorities=semantic_priorities,
                                                  layout_level=layout_level,
                                                  boundary_level=boundary_level,
                                                  box_labels=box_labels
                                                  )

    save_path = os.path.join(ssm_save_folder, image_name)
    if ssm_recovered is not None:
        cv2.imwrite(save_path, ssm_recovered)  # If you want to save the image
    # print(f"Processed: {image_file}")
    return bpp, mIoU


def process_factor(factor, files, test_total, test_size, ignore_label, class_count, save_root, layout_level, boundary_level):
    ssm_save_folder = os.path.join(save_root, f"decompressed-{factor}")
    os.makedirs(ssm_save_folder, exist_ok=True)
    bpp_list, mIoU_list = [], []

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        # Submit tasks to the executor pool
        future_to_image = {
            executor.submit(process_image, image_file, factor, test_size, ignore_label, class_count, test_total,
                            ssm_save_folder, layout_level, boundary_level): image_file
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
    for layout_level in range(5, 20):
        for boundary_level in range(0, layout_level + 1):
            process_factor(factor, files, test_total, test_size, ignore_label, class_count, save_root, layout_level, boundary_level)
