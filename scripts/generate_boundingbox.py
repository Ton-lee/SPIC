import json
import os
from pathlib import Path
import re
from tqdm import tqdm
import shutil


def convert_annotation(image_id, paths):
    global label_map

    def find_box(points):  # 该函数用来找出xmin, xmax, ymin ,ymax 即bbox包围框
        _x, _y = [float(pot[0]) for pot in points], [float(pot[1]) for pot in points]
        return min(_x), max(_x), min(_y), max(_y)

    def convert(size, bbox):  # 转为中心坐标
        # size: (原图宽， 原图长)
        center_x, center_y = (bbox[0] + bbox[1]) / 2.0 - 1, (bbox[2] + bbox[3]) / 2.0 - 1
        center_w, center_h = bbox[1] - bbox[0], bbox[3] - bbox[2]
        return center_x / size[0], center_y / size[1], center_w / size[0], center_h / size[1]

    final_label_path, final_output_path = paths
    label_json_url = os.path.join(final_label_path, f'{image_id}_gtFine_polygons.json')
    # 输出到 ：final_output_path / f'{image_id}_leftImg8bit.txt'

    load_dict = json.load(open(label_json_url, 'r'))  # 图像的实例
    output_cache = []
    for obj in load_dict['objects']:  # load_dict['objects'] -> 目标的几何框体
        obj_label = obj['label']  # 目标的类型
        if obj_label not in label_map:  # 直接跳过不存在于 19 类目标中的区域，与语义分割标注对齐
            continue

        x, y, w, h = convert((load_dict['imgWidth'], load_dict['imgHeight']), find_box(obj['polygon']))  # 归一化为中心点

        # yolo 标准格式：img.jpg -> img.txt
        # 内容的类别 归一化后的中心点x坐标 归一化后的中心点y坐标 归一化后的目标框宽度w 归一化后的目标框高度h
        output_cache.append(f'{label_map[obj_label]} {x} {y} {w} {h}\n')

    with open(os.path.join(final_output_path, f'{image_id}_gtFine_boundingBox.txt'), 'w') as label_f:  # 写出标签文件
        label_f.writelines(output_cache)


def mkdir(url):
    if not os.path.exists(url):
        os.makedirs(url)


if __name__ == '__main__':
    root_dir = "/home/Users/dqy/Dataset/Cityscapes/"
    save_dir = "/home/Users/dqy/Dataset/Cityscapes/format_yolo/"
    image_dir = root_dir + 'leftImg8bit'
    label_dir = root_dir + 'gtFine'
    image_output_root_dir = save_dir + 'images'
    label_output_root_dir = save_dir + 'labels'

    label_map = {
        "road": 0,
        "sidewalk": 1,
        "building": 2,
        "wall": 3,
        "fence": 4,
        "pole": 5,
        "traffic light": 6,
        "traffic sign": 7,
        "vegetation": 8,
        "terrain": 9,
        "sky": 10,
        "person": 11,
        "rider": 12,
        "car": 13,
        "truck": 14,
        "bus": 15,
        "train": 16,
        "motorcycle": 17,
        "bicycle": 18
    }  # 存放所有的类别标签  eg. {'car': 0, 'person': 1}
    for _t_ in tqdm(os.listdir(image_dir)):  # _t_ as ['train', 'test' 'val;]
        type_files = []  # 存放一类的所有文件，如训练集所有文件
        mkdir(os.path.join(image_output_root_dir, _t_)), mkdir(os.path.join(label_output_root_dir, _t_))
        for cities_name in os.listdir(os.path.join(image_dir, _t_)):
            _final_img_path = os.path.join(image_dir, _t_, cities_name)  # root_dir / leftImg8bit / test / berlin
            _final_label_path = os.path.join(label_dir, _t_, cities_name)  # root_dir / getfine / test / berlin

            # berlin_000000_000019_leftImg8bit.png -> berlin_000000_000019_gtFine_polygons.json
            image_ids = list(map(lambda s: re.sub(r'_leftImg8bit\.png', '', s), os.listdir(_final_img_path)))
            # print(names[:0])  -> berlin_000000_000019

            for img_id in image_ids:
                convert_annotation(img_id, [_final_label_path, os.path.join(label_output_root_dir, _t_)])  # 转化标签
                img_file = f'{img_id}_leftImg8bit.png'
                shutil.copy(os.path.join(_final_img_path, img_file), save_dir + f'images/{_t_}/{img_file}')  # 复制移动图片
                type_files.append(f'images/{_t_}/{img_id}_leftImg8bit.png\n')

        with open(os.path.join(save_dir, f'yolo_{_t_}.txt'), 'w') as f:  # 记录训练样本等的具体内容
            f.writelines(type_files)

    with open(os.path.join(label_output_root_dir, 'classes.txt'), 'w') as f:  # 写出类别对应
        for k, v in label_map.items():
            f.write(f'{k}\n')
    print([k for k in label_map.keys()], len([k for k in label_map.keys()]))