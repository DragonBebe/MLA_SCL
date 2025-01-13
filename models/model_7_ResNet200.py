import torch
import torch.nn as nn
import torch.nn.init as init

def initialize_weights(module):
    """Initialize weights using He initialization for Conv2d and Linear layers."""
    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet with three Conv2d layers."""
    expansion = 4  # Output channel expansion factor.

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # First 1x1 convolution for channel reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second 3x3 convolution for spatial processing
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Third 1x1 convolution for channel expansion
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)  # ReLU activation
        self.downsample = downsample  # Downsampling layer for residual connection if needed

        # Apply weight initialization
        self.apply(initialize_weights)

    def forward(self, x):
        identity = x  # Save input for the residual connection.

        # Forward through the bottleneck block
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Apply downsampling if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add residual connection
        out += identity
        out = self.relu(out)
        return out

class ResNet200(nn.Module):
    """ResNet-200 architecture for deep image feature extraction."""
    def __init__(self, num_classes=10):
        super(ResNet200, self).__init__()
        self.in_channels = 64  # Initial number of input channels.

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Initial max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four main ResNet blocks with increasing output dimensions
        self.layer1 = self._make_layer(Bottleneck, 64, 3)  # 64 output channels, 3 blocks
        self.layer2 = self._make_layer(Bottleneck, 128, 24, stride=2)  # 128 output channels, 24 blocks
        self.layer3 = self._make_layer(Bottleneck, 256, 36, stride=2)  # 256 output channels, 36 blocks
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)  # 512 output channels, 3 blocks

        # Adaptive average pooling for variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for classification
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """Construct a ResNet block with a specified number of Bottleneck layers."""
        downsample = None
        # Create a downsampling layer if output dimensions differ
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        # Create the layers for the block
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the ResNet-200 architecture."""
        x = self.relu(self.bn1(self.conv1(x)))  # Initial convolution
        x = self.maxpool(x)  # Max pooling

        # Pass through ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Final global pooling and fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Example usage: Instantiate and print the ResNet-200 model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet200().to(device)
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print("Model architecture:\n", model)
