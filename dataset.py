import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms


class CityscapesDataset(Dataset):
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


def test_image_bpp_estimation():
    # 使用数据集
    root_dir = "/home/Users/dqy/Dataset/Cityscapes/leftImg8bit_merged(512x256)"
    excel_file = "/home/Users/dqy/Dataset/Cityscapes/indicators/bpp_image.xlsx"

    # 创建训练集数据集
    train_dataset = CityscapesDataset(root=root_dir, excel_file=excel_file, phase='train')

    # 创建验证集数据集
    val_dataset = CityscapesDataset(root=root_dir, excel_file=excel_file, phase='val')

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
