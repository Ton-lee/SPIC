import os
import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import cv2


def load_data(
    *,
    dataset_mode,
    data_dir,
    batch_size,
    image_size,
    num_classes=35,
    generated_semantic="",
    image_path="",
    coarse_path="",
    no_instance=True,
    class_cond=False,
    coarse_cond=True,
    deterministic=False,
    random_crop=True,
    random_flip=True,
    is_train=True,
    shifted=False,
    args=None
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if dataset_mode == 'cityscapes':
        if image_path=="":
            all_files = _list_image_files_recursively(os.path.join(data_dir, 'leftImg8bit', 'train' if is_train else 'val'))
        else:
            all_files = _list_image_files_recursively(image_path)
            
        if coarse_path=="":    
            coarse_all_files = None
        else:
            coarse_all_files = _list_image_files_recursively(coarse_path)
        ssm_path = os.path.join(data_dir, 'gtFine', 'train' if is_train else 'val')
        ori_ssm_files = ""
        if args.ssm_path != "":
            ssm_path = args.ssm_path
        if args.condition in ["ssm", "boundary", "sketch"]:
            labels_file = _list_image_files_recursively(ssm_path)
        elif args.condition in ["layout"]:
            labels_file = _list_all_files_recursively(ssm_path)
        elif args.condition in ["layout+boundary"]:
            print(ssm_path)
            labels_file = _list_all_files_recursively(ssm_path)
            ori_ssm_files = _list_image_files_recursively(args.other_folder)
        else:
            raise NotImplementedError(f"Not implemented condition: {args.condition}")
        # print(ssm_path, len(labels_file))
        other_semantics = None
        if args.condition == "ssm":
            if num_classes == 19:
                if generated_semantic == "":
                    classes = [x for x in labels_file if x.endswith('_labelTrainIds.png')]
                else:
                    label_generated =  _list_image_files_recursively(generated_semantic)
                    classes = [x for x in label_generated if x.endswith('_color.png')]
            else:
                classes = [x for x in labels_file if x.endswith('_labelIds.png')]
            if not no_instance:
                instances = [x for x in labels_file if x.endswith('_instanceIds.png')]
            else:
                instances = None
        elif args.condition == "boundary":
            classes = [x for x in labels_file if x.endswith('_instanceBoundary.png')]
            instances = None
        elif args.condition == "layout":
            classes = [x for x in labels_file if x.endswith('_layout.npy')]
            instances = None
        elif args.condition == "sketch":
            classes = [x for x in labels_file]
            instances = None
        elif args.condition == "layout+boundary":
            classes = [x for x in labels_file if x.endswith('_layout.npy')]
            instances = None
            other_semantics = [x for x in ori_ssm_files if x.endswith('_labelTrainIds.png')]
        else:
            raise NotImplementedError(f"Not implemented condition: {args.condition}")
    elif dataset_mode == 'ade20k':
        path_images_ = os.path.join(data_dir, 'train_img' if is_train else 'test_img')
        path_labels_ = os.path.join(data_dir, 'train_label' if is_train else 'test_label')
        all_files = _list_image_files_recursively(path_images_) #os.path.join(data_dir, 'images', 'training' if is_train else 'validation'))
        classes = _list_image_files_recursively(path_labels_) #os.path.join(data_dir, 'annotations', 'training' if is_train else 'validation'))
        instances = None
        other_semantics = None
        coarse_all_files = None
    elif dataset_mode == 'celeba':
        # The edge is computed by the instances.
        # However, the edge get from the labels and the instances are the same on CelebA.
        # You can take either as instance input
        all_files = _list_image_files_recursively(os.path.join(data_dir, 'train' if is_train else 'test', 'images'))
        classes = _list_image_files_recursively(os.path.join(data_dir, 'train' if is_train else 'test', 'labels'))
        instances = _list_image_files_recursively(os.path.join(data_dir, 'train' if is_train else 'test', 'labels'))
        other_semantics = None
        coarse_all_files = None
    else:
        raise NotImplementedError('{} not implemented'.format(dataset_mode))

    print("Len of Dataset:", len(all_files))

    dataset = ImageDataset(
        dataset_mode,
        image_size,
        all_files,
        coarse_all_files=coarse_all_files,
        classes=classes,
        instances=instances,
        num_classes=num_classes,
        generated_semantic=generated_semantic,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        is_train=is_train,
        class_cond=class_cond,
        coarse_cond=coarse_cond,
        args=args,
        other_semantics=other_semantics
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def _list_all_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy", "txt"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_all_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_mode,
        resolution,
        image_paths,
        coarse_all_files,
        classes=None,
        instances=None,
        num_classes=35,
        generated_semantic="",
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        is_train=True,
        class_cond=True,
        coarse_cond=True,
        args=None,
        other_semantics=None,
    ):
        super().__init__()
        self.args = args
        self.is_train = is_train
        self.dataset_mode = dataset_mode
        self.coarse_all_files = None if coarse_all_files is None else coarse_all_files[shard:][::num_shards]
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.generated_semantic = generated_semantic
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.local_instances = None if instances is None else instances[shard:][::num_shards]
        self.other_semantics = other_semantics
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.num_classes = num_classes
        self.class_cond = class_cond
        self.coarse_cond = coarse_cond
        self.shifted = args.shifted
        self.prefix = "shifted"

        if dataset_mode == 'cityscapes':
            self.semantic_priority = [11, 12, 18, 17, 13, 14, 15, 16, 6, 7, 4, 5, 8, 0, 1, 9, 2, 3, 10]
        else:
            self.semantic_priority = []

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # STEP 1: 索引图像
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        # STEP 2: 获取粗糙图像
        if self.coarse_all_files is not None:
            coarse_path = self.coarse_all_files[idx]
            with bf.BlobFile(coarse_path, "rb") as f:
                pil_coarse = Image.open(f)
                pil_coarse.load()
            pil_coarse = pil_coarse.convert("RGB")
            arr_coarse = np.array(pil_coarse)
        else:
            pil_coarse = None
            arr_coarse = None

        # STEP 3: 对视频处理的尝试（舍弃）
        # 为了方便处理，当 shifted 时，将上一张图像的缩小版作为当前的条件
        if self.shifted:
            coarse_path = self.local_images[idx - 1]
            folder_image = path.split("/")[-2]
            folder_coarse = coarse_path.split("/")[-2]
            if folder_image != folder_coarse:
                coarse_path = self.local_images[idx]  # 简单起见，如果遇到视频第一帧则直接用该帧进行生成
            # print(path, coarse_path)
            with bf.BlobFile(coarse_path, "rb") as f:
                pil_coarse = Image.open(f)
                pil_coarse.load()
            pil_coarse = pil_coarse.convert("RGB")
            arr_coarse = np.array(pil_coarse)
            if self.args.compression_type == 'down+bpg':
                temp_folder = "/home/Users/dqy/Projects/SPIC/temp/"
                timestamp = time.time()
                image = cv2.resize(arr_coarse, (self.args.large_size, self.args.small_size))
                cv2.imwrite(f"{temp_folder}compressed_bpg-{self.prefix}#{timestamp}.png", image)
                os.system(f"/home/Users/dqy/myLibs/libbpg-0.9.7/bin/bpgenc -c ycbcr -q  {int(self.args.compression_level)} -o {temp_folder}compressed_bpg-{self.prefix}#{timestamp}.bpg {temp_folder}compressed_bpg-{self.prefix}#{timestamp}.png")
                os.system(f"/home/Users/dqy/myLibs/libbpg-0.9.7/bin/bpgdec -o {temp_folder}compressed_bpg-{self.prefix}#{timestamp}.png {temp_folder}compressed_bpg-{self.prefix}#{timestamp}.bpg")
                decompressed_image = cv2.imread(f"{temp_folder}compressed_bpg-{self.prefix}#{timestamp}.png")
                os.remove(f"{temp_folder}compressed_bpg-{self.prefix}#{timestamp}.png")
                os.remove(f"{temp_folder}compressed_bpg-{self.prefix}#{timestamp}.bpg")
                decompressed_image = cv2.cvtColor(decompressed_image, cv2.COLOR_BGR2RGB)
                arr_coarse = decompressed_image
            else:
                arr_coarse = cv2.resize(arr_coarse, (self.args.large_size, self.args.small_size))

        out_dict = {}
        # print(len(self.local_classes))
        # STEP 4: 获取原始语义条件
        class_path = self.local_classes[idx]
        if self.other_semantics is None:
            other_semantic_path = ""
        else:
            other_semantic_path = self.other_semantics[idx]
        # 对于 layout，进行随机掩蔽：修改对应文件路径（OAR 嵌入的通道表示）
        if self.args.condition == 'layout' and self.num_classes == 16:
            if self.args.random_eliminate:
                # 随机选择一个数量，范围从 0 到 self.num_classes 之间
                num_to_eliminate = random.randint(0, self.num_classes)
                out_dict['eliminate_level'] = num_to_eliminate
                if num_to_eliminate > 0:
                    class_path = class_path.replace("L0", f"L{num_to_eliminate}")
                # print("replace layout file with: ", class_path)
            else:
                if self.args.eliminate_level > 0:
                    class_path = class_path.replace("L0", f"L{self.args.eliminate_level}")
        other_class = None
        if self.args.condition in ["ssm", "boundary", "sketch"]:
            with bf.BlobFile(class_path, "rb") as f:
                pil_class = Image.open(f)
                pil_class.load()
            if self.generated_semantic == "":
                pil_class = pil_class.convert("L")
            else:
                pil_class = pil_class.convert("P")
        elif self.args.condition in ["layout"]:
            layout = np.load(class_path)
            pil_class = layout
        elif self.args.condition in ["layout+boundary"]:
            layout = np.load(class_path)
            pil_class = layout
            with bf.BlobFile(other_semantic_path, "rb") as f:
                other_class = Image.open(f)
                other_class.load()
            if self.generated_semantic == "":
                other_class = other_class.convert("L")
            else:
                other_class = other_class.convert("P")
        else:
            raise NotImplementedError(f"Not implemented condition: {self.args.condition}")
        # STEP 5: 处理实例分割信息
        if self.local_instances is not None:
            instance_path = self.local_instances[idx] # DEBUG: from classes to instances, may affect CelebA
            with bf.BlobFile(instance_path, "rb") as f:
                pil_instance = Image.open(f)
                pil_instance.load()
            pil_instance = pil_instance.convert("L")
        else:
            pil_instance = None
        # STEP 6: 处理图像尺寸
        if self.dataset_mode == 'cityscapes':
            arr_image, arr_class, arr_instance, other_class = self.resize_arr([pil_image, pil_class, pil_instance, other_class], self.resolution)
        else:
            if self.is_train:
                if self.random_crop:
                    arr_image, arr_class, arr_instance = random_crop_arr([pil_image, pil_class, pil_instance], self.resolution)
                else:
                    arr_image, arr_class, arr_instance = center_crop_arr([pil_image, pil_class, pil_instance], self.resolution)
            else:
                arr_image, arr_class, arr_instance, other_class = self.resize_arr([pil_image, pil_class, pil_instance, other_class], self.resolution, keep_aspect=False)
        # STEP 7: 随机翻转
        if self.random_flip and random.random() < 0.5:
            arr_image = arr_image[:, ::-1].copy()
            arr_class = arr_class[:, ::-1].copy()
            arr_instance = arr_instance[:, ::-1].copy() if arr_instance is not None else None
            arr_coarse = arr_coarse[:, ::-1].copy() if arr_coarse is not None else None
            if other_class is not None:
                other_class = other_class[:, ::-1].copy()
        # STEP 8: 图像格式转换
        arr_image = arr_image.astype(np.float32) / 127.5 - 1
        arr_coarse = arr_coarse.astype(np.float32) / 127.5 - 1 if arr_coarse is not None else None
        arr_coarse = np.transpose(arr_coarse, [2, 0, 1]) if arr_coarse is not None else None

        out_dict['path'] = path
        out_dict['label_ori'] = arr_class.copy()
        # STEP 9: 语义标签映射处理
        if self.dataset_mode == 'ade20k':
            arr_class = arr_class - 1
            arr_class[arr_class == 255] = 150
        elif self.dataset_mode == 'coco':
            arr_class[arr_class == 255] = 182
        if self.num_classes == 19:
            arr_class[arr_class == 255] = 19
        if self.args.condition == 'layout+boundary':
            other_class[other_class == 255] = 19
        # STEP 10: 语义标签掩蔽
        # arr_class = np.ones_like(arr_class) * 19  # 测试将所有语义标签掩蔽时的性能
        eliminate_semantics = []
        # 对语义分割图，随机进行语义掩蔽
        if self.args.condition == 'ssm':
            if self.args.random_eliminate:
                # 随机选择一个数量，范围从 0 到 self.num_classes 之间
                num_to_eliminate = random.randint(0, self.num_classes)
                out_dict['eliminate_level'] = num_to_eliminate
            else:
                out_dict['eliminate_level'] = self.args.eliminate_level
            eliminate_semantics = self.semantic_priority[-out_dict['eliminate_level']:]  # 掩蔽不重要的几种语义
            for idx in eliminate_semantics:
                arr_class[arr_class == idx] = self.num_classes
            eliminate_semantics_array = np.ones((self.num_classes, 1))
            for idx in eliminate_semantics:
                eliminate_semantics_array[idx] = 0
        # 对 OAR layout，随机进行语义掩蔽（多通道语义分割图的表示）
        eliminate_semantics_array = None
        if self.args.condition == 'layout' and self.num_classes == 19:
            if self.args.random_eliminate:
                # 随机选择一个数量，范围从 0 到 self.num_classes 之间
                num_to_eliminate = random.randint(0, self.num_classes)
                out_dict['eliminate_level'] = num_to_eliminate
            else:
                out_dict['eliminate_level'] = self.args.eliminate_level
            if out_dict['eliminate_level'] > 0:
                eliminate_semantics = self.semantic_priority[-out_dict['eliminate_level']:]  # 掩蔽不重要的几种语义
            else:
                eliminate_semantics = []
            for idx in eliminate_semantics:
                arr_class[:, :, idx] = 0
            eliminate_semantics_array = np.ones((self.num_classes, 1))
            for idx in eliminate_semantics:
                eliminate_semantics_array[idx] = 0
        if self.args.condition == 'layout+boundary':
            if self.args.random_eliminate:
                # 随机选择一个数量，范围从 0 到 self.num_classes 之间
                num_to_eliminate_layout = random.randint(0, self.num_classes)
                num_to_eliminate_boundary = random.randint(num_to_eliminate_layout, self.num_classes)
                out_dict['eliminate_level'] = {
                    "layout": num_to_eliminate_layout,
                    "boundary": num_to_eliminate_boundary
                }
            else:
                out_dict['eliminate_level'] = self.args.eliminate_level
            full_sem = convert_to_full_sem(other_class, self.num_classes + 1)[:, :, :-1]
            # 将被掩蔽 boundary 的通道置为 layout 信息
            if out_dict['eliminate_level']["boundary"] > 0:
                eliminate_semantics_boundary = self.semantic_priority[-out_dict['eliminate_level']["boundary"]:]
            else:
                eliminate_semantics_boundary = []
            for idx in eliminate_semantics_boundary:
                full_sem[:, :, idx] = arr_class[:, :, idx]
            # 将被掩蔽 layout 的通道置为 0
            if out_dict['eliminate_level']["layout"] > 0:
                eliminate_semantics_layout = self.semantic_priority[-out_dict['eliminate_level']["layout"]:]
            else:
                eliminate_semantics_layout = []
            for idx in eliminate_semantics_layout:
                full_sem[:, :, idx] = 0
            arr_class = full_sem
            eliminate_semantics_array = np.ones((self.num_classes, 2))
            for idx in eliminate_semantics_layout:
                eliminate_semantics_array[idx, 0] = 0
            for idx in eliminate_semantics_boundary:
                eliminate_semantics_array[idx, 1] = 0
        if self.args.eliminate_channels_assist:  # 通过该元素的存在与否判断是否具有掩蔽语义信息辅助
            out_dict['eliminate_semantics'] = eliminate_semantics_array

        # STEP 11: 语义标签格式转换
        if self.args.condition in ["boundary", "sketch"]:
            arr_class = 1 - (arr_class / 255).astype("int")
        if self.args.condition in ["ssm", "boundary", "sketch"]:  # 图像形式的标签需要添加一个通道
            out_dict['label'] = arr_class[None,]
        else:
            out_dict['label'] = np.transpose(arr_class, [2, 0, 1])
        if not self.class_cond:
            out_dict['label'] *= 0
        if not self.coarse_cond:
            out_dict['coarse'] = np.zeros([3] + list(arr_image.shape[:2])).astype(np.float32)

        if arr_instance is not None:
            out_dict['instance'] = arr_instance[None,]

        if arr_coarse is not None:
            out_dict['coarse'] = arr_coarse

        # 清除无用项
        del out_dict['eliminate_level']
        return np.transpose(arr_image, [2, 0, 1]), out_dict#, np.transpose(arr_compressed, [2, 0, 1])


    def resize_arr(self, pil_list, image_size, keep_aspect=True):
        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        pil_image, pil_class, pil_instance, other_class = pil_list
        
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        if keep_aspect:
            scale = image_size / min(*pil_image.size)
            pil_image = pil_image.resize(
                tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
            )
        else:
            pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)
        
        
        if self.args.condition in ["ssm", "boundary", "sketch"]:
            pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
        if pil_instance is not None:
            pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)
        if other_class is not None:
            other_class = other_class.resize(pil_image.size, resample=Image.NEAREST)

        arr_image = np.array(pil_image)
        arr_class = np.array(pil_class)
        arr_instance = np.array(pil_instance) if pil_instance is not None else None
        arr_other = np.array(other_class) if other_class is not None else None
        return arr_image, arr_class, arr_instance, arr_other  # arr_image, arr_compressed, arr_class, arr_instance


def center_crop_arr(pil_list, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    crop_y = (arr_image.shape[0] - image_size) // 2
    crop_x = (arr_image.shape[1] - image_size) // 2
    return arr_image[crop_y : crop_y + image_size, crop_x : crop_x + image_size],\
           arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size],\
           arr_instance[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_instance is not None else None


def random_crop_arr(pil_list, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    crop_y = random.randrange(arr_image.shape[0] - image_size + 1)
    crop_x = random.randrange(arr_image.shape[1] - image_size + 1)
    return arr_image[crop_y : crop_y + image_size, crop_x : crop_x + image_size],\
           arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size],\
           arr_instance[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_instance is not None else None


def convert_to_full_sem(other_class, num_classes):
    H, W = other_class.shape
    full_sem = np.zeros((H, W, num_classes), dtype=np.uint8)
    full_sem[np.arange(H)[:, None], np.arange(W), other_class] = 1
    return full_sem
