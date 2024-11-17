import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler  # 用于混合精度训练


def conv_block(in_channels, out_channels, kernel_size, stride, padding):
    """辅助函数：创建一个卷积块，包括卷积层、批归一化层和LeakyReLU激活"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True)
    )


class ResidualBlock(nn.Module):
    """残差块，用于特征提取，包含两个卷积块和残差连接"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            conv_block(channels, channels // 2, 1, 1, 0),  # 降低通道数
            conv_block(channels // 2, channels, 3, 1, 1)  # 恢复通道数
        )

    def forward(self, x):
        return x + self.block(x)


class CSPResidualBlock(nn.Module):
    """CSP残差块，通过部分通道分离和残差连接减少计算量和显存需求"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(CSPResidualBlock, self).__init__()
        mid_channels = out_channels // 2
        self.split_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride)
        self.residual_blocks = nn.Sequential(
            conv_block(mid_channels, mid_channels, 3, 1, 1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1)
        )
        self.transition_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride)
        self.merge_conv = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        transition = self.transition_conv(x)
        split = self.split_conv(x)
        split = self.residual_blocks(split)
        merged = torch.cat([split, transition], dim=1)
        out = self.merge_conv(merged)
        return out


class CSPDarknet53Classifier(nn.Module):
    """CSPDarknet53的分类器版本，支持输入分辨率调整和混合精度训练"""

    def __init__(self, num_classes=10, input_resolution=224, num_blocks=[2, 2, 8, 8, 4], width_scale=1.0):
        super(CSPDarknet53Classifier, self).__init__()
        self.input_resolution = input_resolution  # 输入分辨率可配置
        self.initial = nn.Sequential(
            conv_block(3, int(32 * width_scale), 3, 1, 1),  # 使用宽度缩放因子
            conv_block(int(32 * width_scale), int(64 * width_scale), 3, 2, 1)
        )
        self.residual_blocks = nn.ModuleList([
            self._make_csp_layer(int(64 * width_scale), int(128 * width_scale), num_blocks[0]),
            self._make_csp_layer(int(128 * width_scale), int(256 * width_scale), num_blocks[1]),
            self._make_csp_layer(int(256 * width_scale), int(512 * width_scale), num_blocks[2]),
            self._make_csp_layer(int(512 * width_scale), int(1024 * width_scale), num_blocks[3]),
            self._make_csp_layer(int(1024 * width_scale), int(2048 * width_scale), num_blocks[4])
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化用于分类
        self.fc = nn.Linear(int(2048 * width_scale), num_classes)  # 全连接层，输出分类结果

    def _make_csp_layer(self, in_channels, out_channels, num_blocks):
        layers = [CSPResidualBlock(in_channels, out_channels, stride=2)]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 根据输入分辨率调整输入
        x = F.interpolate(x, size=(self.input_resolution, self.input_resolution), mode="bilinear", align_corners=False)

        x = self.initial(x)
        for layer in self.residual_blocks:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 设置混合精度训练的示例代码
def train_model(model, dataloader, criterion, optimizer, device, epochs=10):
    scaler = GradScaler()  # 混合精度训练的缩放器
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 混合精度前向传播
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")


# 测试代码示例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSPDarknet53Classifier(num_classes=10, input_resolution=128, width_scale=0.75).to(device)  # 使用缩小的分辨率和宽度
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # 假设输入图像大小为224x224
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 应输出 (1, 10)，表示10个分类结果
