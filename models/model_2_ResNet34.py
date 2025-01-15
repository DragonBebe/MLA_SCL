# The updated code introduces several changes and enhancements compared to the original version:

# 1. **Squeeze-and-Excitation (SE) Block**:
#    - The SEBlock module is added to recalibrate channel-wise feature responses by modeling interdependencies.
#    - Global average pooling is applied, followed by two fully connected layers with a reduction ratio.
#    - A sigmoid activation is used to generate channel-wise weights that rescale the input features.
#    - This helps improve the model's ability to focus on important channels, enhancing feature representation.

# 2. **Integration of SE Block in BasicBlock**:
#    - The `use_se` argument is added to optionally enable SE block usage.
#    - SE block is applied after the second convolution and batch normalization if enabled.
#    - This provides finer control over SE integration at the block level.

# 3. **Updates in ResNet Architecture**:
#    - The `_make_layer` function now includes an additional parameter `use_se` to propagate SE block usage.
#    - SE blocks are selectively applied in the third and fourth layers (higher-level feature maps).
#    - This selective application leverages SE blocks for more abstract and semantically rich features.

# 4. **Dropout Regularization**:
#    - A dropout layer is added before the fully connected layer in the ResNet class.
#    - Dropout with a 0.5 probability helps reduce overfitting by randomly deactivating neurons during training.

# 5. **Functional Enhancements**:
#    - The forward pass of both BasicBlock and ResNet incorporates the SE block and dropout layers.
#    - Modularity and parameterization are improved, making the architecture more adaptable for experiments.

# These updates enhance the model's ability to learn discriminative features, reduce overfitting, 
# and improve generalization, especially in tasks with complex patterns or limited data.
# The original implementation is preserved below

import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation (SE) Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Squeeze-and-Excitation (SE) Block:
        A module that recalibrates channel-wise feature responses by explicitly modeling 
        interdependencies between channels.
        
        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for channel-wise recalibration.
        """
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)  # First fully connected layer
        self.fc2 = nn.Linear(channels // reduction, channels)  # Second fully connected layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for recalibration

    def forward(self, x):
        """
        Forward pass of the SE block.
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
        Returns:
            Tensor with channel-wise recalibration applied.
        """
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # Global average pooling
        y = F.relu(self.fc1(y))  # ReLU activation after first fully connected layer
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)  # Reshape to match input dimensions
        return x * y  # Apply recalibration

# BasicBlock without Depthwise Convolution
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=False):
        """
        BasicBlock:
        A standard residual block consisting of two convolutional layers with optional 
        Squeeze-and-Excitation (SE) and a residual connection.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first convolution.
            downsample (nn.Module): Downsampling module for the residual connection.
            use_se (bool): Whether to include the SE block.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Batch normalization after the first convolution
        self.relu = nn.ReLU(inplace=True)  # ReLU activation
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # Batch normalization after the second convolution
        self.downsample = downsample  # Optional downsampling for the residual connection
        self.se = SEBlock(out_channels) if use_se else nn.Identity()  # Optional SE block

    def forward(self, x):
        """
        Forward pass of the BasicBlock.
        Args:
            x: Input tensor.
        Returns:
            Output tensor with residual connection applied.
        """
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # Apply downsampling if specified

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # Apply SE block if enabled

        out += identity  # Add the residual connection
        out = self.relu(out)  # Final ReLU activation
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        """
        ResNet:
        A ResNet-like architecture consisting of stacked residual blocks and a final classification head.

        Args:
            block (nn.Module): Residual block type (e.g., BasicBlock).
            layers (list of int): Number of blocks in each layer.
            num_classes (int): Number of output classes.
        """
        super(ResNet, self).__init__()
        self.in_channels = 64  # Initial number of channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Initial convolution
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization after initial convolution
        self.relu = nn.ReLU(inplace=True)  # ReLU activation
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, use_se=False)  # First residual layer
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_se=False)  # Second residual layer
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_se=True)  # Third residual layer with SE
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_se=True)  # Fourth residual layer with SE
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(512 * block.expansion, num_classes)  # Fully connected layer for classification
        )

    def _make_layer(self, block, out_channels, blocks, stride=1, use_se=False):
        """
        Helper function to create a residual layer with multiple blocks.
        Args:
            block (nn.Module): Residual block type.
            out_channels (int): Number of output channels.
            blocks (int): Number of blocks in the layer.
            stride (int): Stride for the first block.
            use_se (bool): Whether to include SE blocks.
        Returns:
            nn.Sequential: A sequential container of residual blocks.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample, use_se=use_se)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet.
        Args:
            x: Input tensor.
        Returns:
            Output tensor (logits).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)  # Pass through the fully connected layer
        return x

# Define ResNet-34
def ResNet34(num_classes=10):
    """
    Constructs a ResNet-34 model.
    Args:
        num_classes (int): Number of output classes.
    Returns:
        ResNet: A ResNet-34 model.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

# Test the improved ResNet-34
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34(num_classes=10).to(device)  # Initialize the model
    print(model)

# import torch
# import torch.nn as nn
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += identity
#         out = self.relu(out)
#         return out
#
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channels != out_channels * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * block.expansion),
#             )
#         layers = [block(self.in_channels, out_channels, stride, downsample)]
#         self.in_channels = out_channels * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x
#
# # Define a function to create different versions of ResNet
# def ResNet34(num_classes=10):
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
#
# # Example: Create a ResNet34 instance and move it to CUDA (if available)
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = ResNet34().to(device)
#     print(f"CUDA is available: {torch.cuda.is_available()}")
#     print("Model architecture:\n", model)

