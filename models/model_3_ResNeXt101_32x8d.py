import torch
import torch.nn as nn

# Define the ResNeXt bottleneck block
class ResNeXtBottleneck(nn.Module):
    expansion = 4  # Expansion factor for the output channels

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=8, downsample=None):
        """
        ResNeXt Bottleneck Block:
        A residual block with grouped convolutions to increase cardinality.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels before expansion.
            stride (int): Stride for the convolution.
            cardinality (int): Number of groups for grouped convolutions.
            base_width (int): Base width for grouped convolutions.
            downsample (nn.Module): Optional downsampling layer for residual connection.
        """
        super(ResNeXtBottleneck, self).__init__()
        D = int(out_channels * (base_width / 64))  # Compute width of each group
        self.conv1 = nn.Conv2d(in_channels, D * cardinality, kernel_size=1, bias=False)  # 1x1 convolution
        self.bn1 = nn.BatchNorm2d(D * cardinality)
        self.conv2 = nn.Conv2d(D * cardinality, D * cardinality, kernel_size=3, stride=stride, padding=1,
                               groups=cardinality, bias=False)  # Grouped convolution
        self.bn2 = nn.BatchNorm2d(D * cardinality)
        self.conv3 = nn.Conv2d(D * cardinality, out_channels * self.expansion, kernel_size=1, bias=False)  # 1x1 convolution
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # Downsample layer for residual connection

    def forward(self, x):
        """
        Forward pass of the ResNeXt bottleneck block.

        Args:
            x: Input tensor.
        Returns:
            Output tensor after applying the block.
        """
        identity = x  # Save the input as the residual connection
        out = self.conv1(x)  # First 1x1 convolution
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # Grouped 3x3 convolution
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # Final 1x1 convolution
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # Apply downsampling if specified

        out += identity  # Add residual connection
        out = self.relu(out)  # Final ReLU activation
        return out

# Define the ResNeXt model
class ResNeXt(nn.Module):
    def __init__(self, block, layers, cardinality=32, base_width=8, num_classes=10):
        """
        ResNeXt Model:
        A deep neural network consisting of stacked ResNeXt bottleneck blocks.

        Args:
            block (nn.Module): The block type (e.g., ResNeXtBottleneck).
            layers (list of int): Number of blocks in each layer.
            cardinality (int): Number of groups for grouped convolutions.
            base_width (int): Base width for grouped convolutions.
            num_classes (int): Number of output classes.
        """
        super(ResNeXt, self).__init__()
        self.in_channels = 64  # Initial number of channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Initial 3x3 convolution
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, cardinality=cardinality, base_width=base_width)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, cardinality=cardinality, base_width=base_width)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, cardinality=cardinality, base_width=base_width)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, cardinality=cardinality, base_width=base_width)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # Fully connected layer for classification

    def _make_layer(self, block, out_channels, blocks, stride, cardinality, base_width):
        """
        Helper function to create a ResNeXt layer with multiple blocks.

        Args:
            block (nn.Module): Block type (e.g., ResNeXtBottleneck).
            out_channels (int): Number of output channels before expansion.
            blocks (int): Number of blocks in the layer.
            stride (int): Stride for the first block.
            cardinality (int): Number of groups for grouped convolutions.
            base_width (int): Base width for grouped convolutions.
        Returns:
            nn.Sequential: A sequential container of ResNeXt blocks.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, cardinality, base_width, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, cardinality=cardinality, base_width=base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNeXt model.

        Args:
            x: Input tensor.
        Returns:
            Output tensor after passing through the network.
        """
        x = self.conv1(x)  # Initial convolution
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)  # Layer 1
        x = self.layer2(x)  # Layer 2
        x = self.layer3(x)  # Layer 3
        x = self.layer4(x)  # Layer 4
        x = self.avgpool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten to vector
        x = self.fc(x)  # Fully connected layer
        return x

# Define a function to create the ResNeXt101_32x8d model
def ResNeXt101_32x8d(num_classes=10):
    """
    Constructs a ResNeXt-101 model with 32 groups and 8 base width.
    Args:
        num_classes (int): Number of output classes.
    Returns:
        ResNeXt: A ResNeXt-101 model.
    """
    return ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], cardinality=32, base_width=8, num_classes=num_classes)

# Test the ResNeXt model (for local execution)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability
    model = ResNeXt101_32x8d().to(device)  # Initialize the model
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print("Model architecture:\n", model)  # Print the model architecture
