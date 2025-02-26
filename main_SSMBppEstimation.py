import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.networks import build_ssm_bpp_model
from dataset import SSMBppDataset
import os
import argparse
import pandas as pd
from tqdm import tqdm


# 训练函数
def train(model, train_loader, optimizer, criterion, device):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0

    # 使用 tqdm 显示训练过程的进度条
    for images, ssms, layout_levels, boundary_levels, fit_thresholds, bpps, _ in tqdm(train_loader,
                                                                                      desc="Training", unit="batch"):
        images, ssms, layout_levels, boundary_levels, fit_thresholds, bpps = (images.to(device),
                                                                              ssms.to(device),
                                                                              layout_levels.to(device),
                                                                              boundary_levels.to(device),
                                                                              fit_thresholds.to(device),
                                                                              bpps.to(device)
                                                                              )
        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images, ssms, layout_levels, boundary_levels, fit_thresholds)  # 假设模型输出为 bpp 的估计值

        # 计算损失
        loss = criterion(outputs.squeeze(), bpps)  # MSE损失，输出是bpp的估计值，bpp是目标值
        running_loss += loss.item()

        # 反向传播
        loss.backward()

        # 更新优化器
        optimizer.step()

    epoch_loss = running_loss / len(train_loader)
    print(f"Train Loss: {epoch_loss:.4f}")
    return epoch_loss


# 验证函数
def validate(model, val_loader, criterion, device, print_file=None):
    model.eval()  # 设置模型为验证模式
    running_loss = 0.0
    predictions = []

    # 使用 tqdm 显示验证过程的进度条
    with torch.no_grad():  # 不计算梯度
        for images, ssms, layout_levels, boundary_levels, fit_thresholds, bpps, names in tqdm(val_loader, desc="Validation", unit="batch"):
            images, ssms, layout_levels, boundary_levels, fit_thresholds, bpps = (images.to(device),
                                                                                  ssms.to(device),
                                                                                  layout_levels.to(device),
                                                                                  boundary_levels.to(device),
                                                                                  fit_thresholds.to(device),
                                                                                  bpps.to(device)
                                                                                  )

            # 前向传播
            outputs = model(images, ssms, layout_levels, boundary_levels, fit_thresholds)  # 假设模型输出为 bpp 的估计值

            # 计算损失
            loss = criterion(outputs.squeeze(), bpps)
            running_loss += loss.item()

            # 收集预测结果
            for i in range(len(images)):
                predictions.append({
                    'name': names[i],
                    'layout_level': layout_levels[i].item(),
                    'boundary_level': boundary_levels[i].item(),
                    'fit_threshold': fit_thresholds[i].item(),
                    'bpp_gt': bpps[i].item(),
                    'bpp_pred': outputs[i].item()
                })

    epoch_loss = running_loss / len(val_loader)
    print(f"Validation Loss: {epoch_loss:.4f}")

    # 如果指定了输出文件路径，保存预测结果
    if print_file:
        df = pd.DataFrame(predictions)
        with pd.ExcelWriter(print_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        print(f"Prediction results saved to {print_file}")

    return epoch_loss


# 主程序
def main():
    # 配置命令行参数
    parser = argparse.ArgumentParser(description="Training and validation of the model.")
    parser.add_argument('--phase', type=str, choices=['train', 'val', 'train+val'], default='train',
                        help="Phase to run: 'train', 'val', or 'train+val'")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="Directory where the model checkpoints will be saved.")
    parser.add_argument('--checkpoint', type=str, default='',
                        help="Path to a pre-trained model to load. Leave empty to not load a pre-trained model.")
    parser.add_argument('--print_file', type=str, default='',
                        help="File path to save predictions on validation. If empty, predictions will not be saved.")
    args = parser.parse_args()

    # 配置设备：使用 GPU 如果可用，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集初始化
    root_dir = "/home/Users/dqy/Dataset/Cityscapes/leftImg8bit_merged(512x256)"
    ssm_dir = "/home/Users/dqy/Dataset/Cityscapes/ssm_merged(512x256)"
    excel_file = "/home/Users/dqy/Dataset/Cityscapes/indicators/bpp_layout+boundary.xlsx"

    # 初始化训练和验证数据集
    train_dataset = SSMBppDataset(image_root=root_dir, ssm_root=ssm_dir, excel_file=excel_file, phase='train')
    val_dataset = SSMBppDataset(image_root=root_dir, ssm_root=ssm_dir, excel_file=excel_file, phase='val')

    # 使用 DataLoader 加载数据
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 初始化模型
    model = build_ssm_bpp_model()
    model.to(device)  # 将模型移动到 GPU 或 CPU

    # 如果指定了预训练模型路径，加载预训练模型
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint))
            print(f"Pre-trained model loaded from {args.checkpoint}")
        else:
            raise FileNotFoundError(f"Pre-trained model not found at {args.checkpoint}")

    # 设置超参数
    learning_rate = 1e-4
    epochs = 30

    # 设置优化器（Adam 优化器）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 设置损失函数（MSE损失）
    criterion = nn.MSELoss()

    # 定义保存模型的路径
    checkpoint_dir = args.save_dir
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)  # 确保保存目录存在
    # 定义保存预测结果的路径
    save_pred_dir = os.path.dirname(args.print_file)
    if save_pred_dir:
        os.makedirs(save_pred_dir, exist_ok=True)  # 确保保存目录存在

    # 训练和验证循环
    best_val_loss = float('inf')

    for epoch in range(epochs):

        # 训练模型
        if args.phase in ['train', 'train+val']:
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss = train(model, train_loader, optimizer, criterion, device)
            # 保存当前epoch模型
            model_save_path = os.path.join(checkpoint_dir, f"ImageBpp_EP{epoch + 1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

        # 验证模型
        if args.phase in ['val', 'train+val']:
            val_loss = validate(model, val_loader, criterion, device, args.print_file)

        # 保存最佳模型（根据验证损失）
        if args.phase in ['train+val'] and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

        if args.phase in ['train+val']:
            print(f"Best Validation Loss: {best_val_loss:.4f}")

        if args.phase in ["val"]:
            break


if __name__ == '__main__':
    main()
