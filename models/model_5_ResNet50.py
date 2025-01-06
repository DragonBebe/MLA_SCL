import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation (SE) Block: Enhances feature representation by adaptively recalibrating channel-wise feature responses.
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling for spatial dimensions.
        self.fc1 = nn.Linear(channels, channels // reduction)  # First fully connected layer for dimensionality reduction.
        self.fc2 = nn.Linear(channels // reduction, channels)  # Second fully connected layer to restore dimensions.
        self.sigmoid = nn.Sigmoid()  # Activation function to produce weights in the range [0, 1].

    def forward(self, x):
        b, c, _, _ = x.size()  # Extract batch and channel dimensions.
        y = self.global_pool(x).view(b, c)  # Apply global average pooling and flatten.
        y = F.relu(self.fc1(y))  # Pass through the first fully connected layer with ReLU activation.
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)  # Apply the second layer and reshape to match input dimensions.
        return x * y  # Multiply input features by the recalibrated weights.

# Bottleneck Block: A residual block with optional SE Block and GELU activation for flexibility and efficiency.
class BottleneckSE(nn.Module):
    expansion = 4  # Factor by which the number of output channels expands.

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=False, use_gelu=False):
        super(BottleneckSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1 convolution for channel reduction.
        self.bn1 = nn.BatchNorm2d(out_channels)  # Batch normalization after the first convolution.
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # 3x3 convolution.
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)  # 1x1 convolution for channel expansion.
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)  # Standard ReLU activation for intermediate layers.
        self.gelu = nn.GELU() if use_gelu else self.relu  # Optional GELU activation.
        self.downsample = downsample  # Optional downsampling to match dimensions.
        self.use_se = use_se  # Flag to include SE Block.
        if use_se:
            self.se = SEBlock(out_channels * self.expansion)

    def forward(self, x):
        identity = x  # Save the input for residual connection.
        if self.downsample is not None:
            identity = self.downsample(x)  # Adjust dimensions if necessary.

        out = self.gelu(self.bn1(self.conv1(x)))  # First convolution with GELU/ReLU activation.
        out = self.gelu(self.bn2(self.conv2(out)))  # Second convolution with GELU/ReLU activation.
        out = self.bn3(self.conv3(out))  # Third convolution without activation.

        if self.use_se:
            out = self.se(out)  # Apply SE Block if enabled.

        out += identity  # Add residual connection.
        return self.gelu(out)  # Final activation.

# Mixed Pooling Module: Combines average and max pooling for better feature extraction.
class MixedPooling(nn.Module):
    def __init__(self):
        super(MixedPooling, self).__init__()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling.
        max_pool = F.adaptive_max_pool2d(x, (1, 1))  # Global max pooling.
        return 0.5 * (avg_pool + max_pool)  # Combine average and max pooling results.

# Enhanced ResNet50: ResNet50 with Bottleneck blocks, optional SE Blocks, and mixed pooling.
class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.in_channels = 64  # Initial number of channels.

        # Initial convolutional layer and max pooling.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stacked Bottleneck layers with varying configurations.
        self.layer1 = self._make_layer(BottleneckSE, 64, 3, use_se=False, use_gelu=False)
        self.layer2 = self._make_layer(BottleneckSE, 128, 4, stride=2, use_se=False, use_gelu=False)
        self.layer3 = self._make_layer(BottleneckSE, 256, 6, stride=2, use_se=False, use_gelu=True)
        self.layer4 = self._make_layer(BottleneckSE, 512, 3, stride=2, use_se=True, use_gelu=True)

        # Mixed pooling, dropout, and fully connected layer for classification.
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
        x = self.relu(self.bn1(self.conv1(x)))  # Initial convolutional layer.
        x = self.maxpool(x)  # Max pooling.

        # Apply sequential layers.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.mixed_pooling(x)  # Mixed pooling.
        x = torch.flatten(x, 1)  # Flatten feature maps.
        x = self.dropout(x)  # Apply dropout.
        return self.fc(x)  # Final fully connected layer.

# Testing the enhanced ResNet50 implementation.
if __name__ == "__main__":
    model = ResNet50(num_classes=10)
    x = torch.randn(2, 3, 224, 224)  # Example input: batch size of 2, 3 color channels, 224x224 resolution.
    output = model(x)
    print("Output shape:", output.shape)  # Expected output shape: [2, 10].
