import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Block for WideResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        A basic residual block used in WideResNet.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the convolutional layer.
            downsample (nn.Module): Downsampling layer to match dimensions of input and output.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # 3x3 convolution
        self.bn1 = nn.BatchNorm2d(out_channels)  # Batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)  # 3x3 convolution
        self.bn2 = nn.BatchNorm2d(out_channels)  # Batch normalization
        self.downsample = downsample  # Optional downsampling layer

    def forward(self, x):
        """
        Forward pass for the BasicBlock.

        Args:
            x: Input tensor.
        Returns:
            Output tensor after applying the block.
        """
        identity = x  # Save the input as residual connection
        out = F.relu(self.bn1(self.conv1(x)))  # First convolution + BN + ReLU
        out = self.bn2(self.conv2(out))  # Second convolution + BN
        if self.downsample is not None:
            identity = self.downsample(x)  # Apply downsampling if required
        out += identity  # Add residual connection
        return F.relu(out)  # Final ReLU activation

# WideResNet model definition
class WideResNet(nn.Module):
    def __init__(self, block, layers, width_factor, num_classes=10):
        """
        WideResNet Model.

        Args:
            block (nn.Module): The basic building block (e.g., BasicBlock).
            layers (list of int): Number of blocks in each layer.
            width_factor (int): Width multiplier to increase the number of channels.
            num_classes (int): Number of output classes.
        """
        super(WideResNet, self).__init__()
        self.in_channels = 16  # Initial number of channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)  # Initial 3x3 convolution
        self.layer1 = self._make_layer(block, 16 * width_factor, layers[0])  # First layer
        self.layer2 = self._make_layer(block, 32 * width_factor, layers[1], stride=2)  # Second layer with stride 2
        self.layer3 = self._make_layer(block, 64 * width_factor, layers[2], stride=2)  # Third layer with stride 2
        self.bn = nn.BatchNorm2d(64 * width_factor)  # Final batch normalization
        self.fc = nn.Linear(64 * width_factor, num_classes)  # Fully connected layer for classification

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Helper function to create a layer consisting of multiple blocks.

        Args:
            block (nn.Module): Block type (e.g., BasicBlock).
            out_channels (int): Number of output channels for the layer.
            blocks (int): Number of blocks in the layer.
            stride (int): Stride for the first block.
        Returns:
            nn.Sequential: A sequential container of blocks.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            # Create downsampling layer to match input and output dimensions
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),  # 1x1 convolution
                nn.BatchNorm2d(out_channels),  # Batch normalization
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]  # First block
        self.in_channels = out_channels  # Update in_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))  # Add additional blocks
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the WideResNet model.

        Args:
            x: Input tensor.
        Returns:
            Output tensor after passing through the network.
        """
        x = self.conv1(x)  # Initial convolution
        x = self.layer1(x)  # First layer
        x = self.layer2(x)  # Second layer
        x = self.layer3(x)  # Third layer
        x = F.relu(self.bn(x))  # Apply final batch normalization and ReLU
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)  # Fully connected layer
        return x

# Function to create a WideResNet-28-10 instance
def WideResNet_28_10(num_classes=10):
    """
    Constructs a WideResNet-28-10 model.

    Args:
        num_classes (int): Number of output classes.
    Returns:
        WideResNet: A WideResNet-28-10 model.
    """
    return WideResNet(BasicBlock, [4, 4, 4], width_factor=10, num_classes=num_classes)

# Example usage
if __name__ == "__main__":
    model = WideResNet_28_10()  # Create a WideResNet-28-10 model
    print(model)  # Print the model architecture
