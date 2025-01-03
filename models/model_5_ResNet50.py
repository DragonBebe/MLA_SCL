# import torch
# import torch.nn as nn
# import torch.nn.init as init
#
# def initialize_weights(module):
#     """对卷积层和全连接层进行 He 初始化"""
#     if isinstance(module, nn.Conv2d):
#         init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
#         if module.bias is not None:
#             init.constant_(module.bias, 0)
#     elif isinstance(module, nn.Linear):
#         init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
#         if module.bias is not None:
#             init.constant_(module.bias, 0)
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#         # 初始化权重
#         self.apply(initialize_weights)
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
# class ResNet50(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet50, self).__init__()
#         self.in_channels = 64
#
#         # 初始卷积层和池化层
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         # 四个残差层
#         self.layer1 = self._make_layer(Bottleneck, 64, 3)
#         self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
#         self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
#         self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
#
#         # 全局平均池化和全连接层
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
#
#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channels != out_channels * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * block.expansion),
#             )
#
#         layers = [block(self.in_channels, out_channels, stride, downsample)]
#         self.in_channels = out_channels * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x
#
# # 示例：创建一个 ResNet50 实例并移到 CUDA（如果可用）
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = ResNet50().to(device)
#     print(f"CUDA is available: {torch.cuda.is_available()}")
#     print("Model architecture:\n", model)



"""SE Block：
在 layer3 和 layer4 中启用，增强深层通道注意力。
GELU 激活函数：
在 Bottleneck Block 中替换部分 ReLU，提升训练的稳定性和收敛效果。
Dropout：
在全连接层前加入 Dropout(p=0.5)，防止过拟合。
混合池化：
替代全局平均池化，结合最大池化和平均池化的优点，提升泛化能力。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation (SE) Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

# Bottleneck Block with optional SE Block and GELU activation
class BottleneckSE(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=False, use_gelu=False):
        super(BottleneckSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU() if use_gelu else self.relu
        self.downsample = downsample
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels * self.expansion)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.gelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.use_se:
            out = self.se(out)  # 加入 SE 模块

        out += identity
        return self.gelu(out)

# 混合池化模块
class MixedPooling(nn.Module):
    def __init__(self):
        super(MixedPooling, self).__init__()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        max_pool = F.adaptive_max_pool2d(x, (1, 1))
        return 0.5 * (avg_pool + max_pool)

# Enhanced ResNet50
class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # ReLU for shallow layers
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottleneck layers
        self.layer1 = self._make_layer(BottleneckSE, 64, 3, use_se=False, use_gelu=False)
        self.layer2 = self._make_layer(BottleneckSE, 128, 4, stride=2, use_se=False, use_gelu=False)
        self.layer3 = self._make_layer(BottleneckSE, 256, 6, stride=2, use_se=False, use_gelu=True)
        self.layer4 = self._make_layer(BottleneckSE, 512, 3, stride=2, use_se=True, use_gelu=True)

        # Average Pooling and Dropout
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mixed_pooling = MixedPooling()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512 * BottleneckSE.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, use_se=False, use_gelu=False):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample, use_se, use_gelu)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_se=use_se, use_gelu=use_gelu))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        x = self.mixed_pooling(x)  # Mixed pooling
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# 测试代码
if __name__ == "__main__":
    model = ResNet50(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print("Output shape:", output.shape)  # 预期 [2, 10]
