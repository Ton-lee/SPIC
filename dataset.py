import os
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as F


class ImageBppDataset(Dataset):
    def __init__(self, root, excel_file, phase):
        """
        初始化数据集

        :param root: 图像所在的根目录
        :param excel_file: Excel文件路径
        :param phase: 选择数据集阶段，'train' 或 'val'
        """
        self.root = root
        self.phase = phase
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        # 读取Excel文件中的数据
        self.df = pd.read_excel(excel_file, sheet_name=phase)  # 读取工作表

        # 提取图像名称、QP和bpp列
        self.image_names = self.df['name'].values
        self.qp_values = self.df['QP'].values
        self.bpp_values = self.df['bpp'].values

    def __len__(self):
        """
        返回数据集的样本数
        """
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        根据索引获取样本（图像，QP，bpp）

        :param idx: 索引
        :return: 样本数据，包括图像（tensor）、QP（int）和bpp（float）
        """
        image_name = self.image_names[idx]
        qp = self.qp_values[idx]
        bpp = self.bpp_values[idx]

        # 读取图像
        image_path = os.path.join(self.root, self.phase, f"{image_name}.png")
        image = Image.open(image_path).convert('RGB')

        # 应用转换（如果有）
        if self.transform:
            image = self.transform(image)

        # 返回图像，QP和bpp
        return image, torch.tensor(qp, dtype=torch.int), torch.tensor(bpp, dtype=torch.float), image_name


class SSMBppDataset(Dataset):
    def __init__(self, image_root, ssm_root, excel_file, phase):
        """
        初始化数据集

        :param image_root: 图像所在的根目录
        :param excel_file: Excel文件路径
        :param phase: 选择数据集阶段，'train' 或 'val'
        """
        self.image_root = image_root
        self.ssm_root = ssm_root
        self.phase = phase
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 读取Excel文件中的数据
        # self.thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
        self.thresholds = [1.0, 1.5]
        self.categories = 19
        self.df = {}
        for threshold in self.thresholds:
            data = pd.read_excel(excel_file, sheet_name=f"{phase}#th{threshold:.1f}")  # 读取工作表
            self.df[threshold] = data

        # 提取图像名称、QP和bpp列
        self.image_names = {}
        self.layout_levels = {}
        self.boundary_levels = {}
        self.fit_thresholds = {}
        self.bpp_values = {}
        for threshold in self.thresholds:
            self.image_names[threshold] = self.df[threshold]['name'].values
            self.layout_levels[threshold] = self.df[threshold]['layout_level'].values
            self.boundary_levels[threshold] = self.df[threshold]['boundary_level'].values
            self.bpp_values[threshold] = self.df[threshold]['bpp'].values

    def __len__(self):
        """
        返回数据集的样本数
        """
        return len(self.image_names[1.0]) * len(self.thresholds)

    def __getitem__(self, idx):
        """
        根据索引获取样本（图像，QP，bpp）

        :param idx: 索引
        :return: 样本数据，包括图像（tensor）、QP（int）和bpp（float）
        """
        threshold_index = idx // len(self.image_names[1.0])
        threshold = self.thresholds[threshold_index]
        idx %= len(self.image_names[1.0])
        image_name = self.image_names[threshold][idx]
        layout_level = self.layout_levels[threshold][idx]
        boundary_level = self.boundary_levels[threshold][idx]
        bpp_value = self.bpp_values[threshold][idx]
        # 读取图像
        image_path = os.path.join(self.image_root, self.phase,
                                  image_name.replace("gtFine_labelTrainIds", "leftImg8bit"))
        image = Image.open(image_path).convert('RGB')
        ssm_path = os.path.join(self.ssm_root, self.phase, image_name)
        ssm = Image.open(ssm_path).convert('L')
        # 对 ssm 进行 one-hot 映射
        ssm_array = np.array(ssm)  # 转换为 numpy 数组
        one_hot_ssm = np.zeros((self.categories, *ssm_array.shape), dtype=np.float32)  # 创建 19 通道的空矩阵
        # 将 ssm 的值映射到 one-hot 编码
        for i in range(self.categories):  # ssm 值为 0 到 18
            one_hot_ssm[i] = (ssm_array == i).astype(np.float32)
        # 将 one-hot 编码转换为 tensor
        one_hot_ssm = torch.tensor(one_hot_ssm)
        # 应用转换（如果有）
        if self.transform:
            image = self.transform(image)
        # 以 0.5 的概率对图像和 ssm 进行左右翻转
        if random.random() < 0.5:
            image = F.hflip(image)
            one_hot_ssm = F.hflip(one_hot_ssm)
        # 以 0.5 的概率对图像和 ssm 进行上下翻转
        if random.random() < 0.5:
            image = F.vflip(image)
            one_hot_ssm = F.vflip(one_hot_ssm)

        # 返回图像，QP和bpp
        return (image,
                one_hot_ssm,
                torch.tensor(layout_level, dtype=torch.float),
                torch.tensor(boundary_level, dtype=torch.float),
                torch.tensor(threshold, dtype=torch.float),
                torch.tensor(bpp_value, dtype=torch.float),
                image_name)


def test_image_bpp_estimation():
    # 使用数据集
    root_dir = "/home/Users/dqy/Dataset/Cityscapes/leftImg8bit_merged(512x256)"
    excel_file = "/home/Users/dqy/Dataset/Cityscapes/indicators/bpp_image.xlsx"

    # 创建训练集数据集
    train_dataset = ImageBppDataset(root=root_dir, excel_file=excel_file, phase='train')

    # 创建验证集数据集
    val_dataset = ImageBppDataset(root=root_dir, excel_file=excel_file, phase='val')

    # 通过 DataLoader 加载数据
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 示例：获取一个训练批次的样本
    for images, qps, bpps in train_loader:
        print(images.shape)  # 批次中图像的形状
        print(qps.shape)  # 批次中QP的形状
        print(bpps.shape)  # 批次中bpp的形状
        break  # 只查看一个批次



if __name__ == '__main__':
    test_image_bpp_estimation()
