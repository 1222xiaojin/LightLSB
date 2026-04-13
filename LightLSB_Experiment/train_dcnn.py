# train_dcnn.py  适配 112x112 图像
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import numpy as np
import cv2


class ImageDataset(Dataset):
    def __init__(self, blurry_dir, clear_dir, image_size=(112, 112)):  # 改这里！
        self.blurry_dir = blurry_dir
        self.clear_dir = clear_dir
        self.image_size = image_size
        self.filenames = [f for f in os.listdir(blurry_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 读取模糊图像
        blurry_path = os.path.join(self.blurry_dir, self.filenames[idx])
        blurry_img = cv2.imread(blurry_path)

        # 读取清晰图像
        clear_path = os.path.join(self.clear_dir, self.filenames[idx])
        clear_img = cv2.imread(clear_path)

        # 预处理
        blurry_img = cv2.resize(blurry_img, self.image_size)
        clear_img = cv2.resize(clear_img, self.image_size)

        # 转换为Tensor
        blurry_img = torch.from_numpy(blurry_img).float().permute(2, 0, 1) / 255.0
        clear_img = torch.from_numpy(clear_img).float().permute(2, 0, 1) / 255.0

        return blurry_img, clear_img


class DCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()
        # 移除所有下采样，保证输入输出尺寸一致 112x112
        self.encoder_decoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_decoder(x)
        return x


def train_dcnn(blurry_dir, clear_dir, model_save_path, num_epochs=50, batch_size=8, learning_rate=0.001):
    # 创建数据集和DataLoader
    dataset = ImageDataset(blurry_dir, clear_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建模型、损失函数和优化器
    model = DCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for blurry, clear in dataloader:
            # 前向传播
            outputs = model(blurry)
            loss = criterion(outputs, clear)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到 {model_save_path}")


def main():
    parser = argparse.ArgumentParser(description='训练DCNN模型')
    parser.add_argument('--blurry_dir', required=True, help='模糊图像训练目录')
    parser.add_argument('--clear_dir', required=True, help='清晰图像训练目录')
    parser.add_argument('--model_save_path', required=True, help='模型保存路径')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--learning_rate', type=str, default=0.001, help='学习率')
    args = parser.parse_args()

    train_dcnn(
        blurry_dir=args.blurry_dir,
        clear_dir=args.clear_dir,
        model_save_path=args.model_save_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=float(args.learning_rate)
    )


if __name__ == "__main__":
    main()