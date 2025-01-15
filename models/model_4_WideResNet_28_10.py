# Differences and advantages of WideResNet compared to traditional ResNet, along with its inspirations:

# 1. **Width Expansion**:
#    - WideResNet introduces a `width_factor` that multiplies the number of channels in each layer.
#    - Instead of increasing depth, WideResNet broadens the network, allowing for better feature representation while reducing the risk of vanishing gradients.
#    - This addresses the inefficiency of very deep networks in ResNet, where adding more layers yields diminishing returns.

# 2. **Shallower Architecture**:
#    - WideResNet uses significantly fewer layers than traditional ResNet (e.g., WideResNet-28-10 has 28 layers compared to ResNet-50's 50 layers).
#    - Despite being shallower, it achieves comparable or better performance due to its increased width.

# 3. **Efficiency in Training**:
#    - By increasing width rather than depth, WideResNet reduces training time while maintaining high accuracy.
#    - Shallower networks also require fewer computations for forward and backward passes compared to deeper ResNets.

# 4. **Basic Block Simplification**:
#    - The BasicBlock in WideResNet is simpler, consisting of two 3x3 convolutions with batch normalization and ReLU activation.
#    - This contrasts with the bottleneck design in ResNet, which uses 1x1 convolutions for dimensionality reduction and restoration, adding additional complexity.

# 5. **Residual Connection and Downsampling**:
#    - WideResNet retains the residual connection mechanism, ensuring efficient gradient flow during training.
#    - Downsampling is handled through a 1x1 convolution in the first block of each layer when necessary, similar to ResNet.

# 6. **Global Average Pooling**:
#    - Like ResNet, WideResNet applies global average pooling before the fully connected layer.
#    - This reduces spatial dimensions to 1x1, ensuring the final feature map is compact and robust for classification tasks.

# 7. **Motivation from ResNet Limitations**:
#    - WideResNet addresses the shortcomings of very deep ResNets, such as increased training time and difficulty in optimizing very deep architectures.
#    - Research showed that increasing width instead of depth can provide a better trade-off between performance and computational cost.

# 8. **Wider Feature Representations**:
#    - Increasing width allows WideResNet to learn more diverse feature representations, particularly useful for complex datasets.
#    - This broadens the network's capacity to extract finer-grained details in images.

# Inspirations:
# - WideResNet is inspired by the success of residual networks in mitigating vanishing gradients but focuses on width expansion as a new avenue for improving performance.
# - The width-depth trade-off builds on findings from experiments with very deep ResNets, where increasing depth beyond a point yields diminishing accuracy improvements.

# Overall, WideResNet achieves high accuracy with fewer layers, shorter training times, and comparable or lower computational cost than traditional ResNet, making it an efficient alternative for various tasks.


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
