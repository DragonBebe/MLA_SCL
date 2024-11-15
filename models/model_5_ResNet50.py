import torch
import torch.nn as nn
import torch.nn.init as init

def initialize_weights(module):
"""对卷积层和全连接层进行 He 初始化"""
if isinstance(module, nn.Conv2d):
    init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    if module.bias is not None:
        init.constant_(module.bias, 0)
elif isinstance(module, nn.Linear):
    init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    if module.bias is not None:
        init.constant_(module.bias, 0)

class Bottleneck(nn.Module):
expansion = 4

def __init__(self, in_channels, out_channels, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample

    # 初始化权重
    self.apply(initialize_weights)

def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

class ResNet50(nn.Module):
def __init__(self, num_classes=10):
    super(ResNet50, self).__init__()
    self.in_channels = 64

    # 初始卷积层和池化层
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # 四个残差层
    self.layer1 = self._make_layer(Bottleneck, 64, 3)
    self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
    self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
    self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

    # 全局平均池化和全连接层
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

def _make_layer(self, block, out_channels, blocks, stride=1):
    downsample = None
    if stride != 1 or self.in_channels != out_channels * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * block.expansion),
        )

    layers = [block(self.in_channels, out_channels, stride, downsample)]
    self.in_channels = out_channels * block.expansion
    for _ in range(1, blocks):
        layers.append(block(self.in_channels, out_channels))

    return nn.Sequential(*layers)

def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x

# 示例：创建一个 ResNet50 实例并移到 CUDA（如果可用）
if __name__ == "__main__":
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50().to(device)
print(f"CUDA is available: {torch.cuda.is_available()}")
print("Model architecture:\n", model)
