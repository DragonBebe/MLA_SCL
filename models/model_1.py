import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, BatchNorm2d, Dropout

# Define a convolutional block with optional residual connection
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res=True):
        """
        A Convolutional Block with optional residual connection.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the convolution. Default is 1.
            res (bool): Whether to include a residual connection. Default is True.
        """
        super(ConvBlock, self).__init__()
        self.res = res  # Indicates whether to use residual connections
        
        # The main branch of the block
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut connection for residual
        self.shortcut = nn.Sequential()
        if res and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)  # Final activation function

    def forward(self, x):
        """
        Forward pass of the convolutional block.
        Args:
            x: Input tensor.
        Returns:
            Output tensor after applying the block.
        """
        out = self.left(x)  # Main branch
        if self.res:
            out += self.shortcut(x)  # Add the shortcut if residual is enabled
        out = self.relu(out)  # Apply activation
        return out

# Define the ResNet-like model
class ResModel(nn.Module):
    def __init__(self, res=True):
        """
        A ResNet-like model with 4 convolutional blocks and a fully connected classifier.
        Args:
            res (bool): Whether to use residual connections in ConvBlocks.
        """
        super(ResModel, self).__init__()
        # Define convolutional blocks
        self.block1 = ConvBlock(3, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        self.block4 = ConvBlock(256, 512)
        
        # Define the fully connected classifier
        self.classifier = nn.Sequential(
            Flatten(),               # Flatten the feature maps into a vector
            Dropout(0.4),            # Dropout for regularization
            Linear(2048, 256),       # Fully connected layer with 256 neurons
            Linear(256, 64),         # Fully connected layer with 64 neurons
            Linear(64, 10)           # Output layer with 10 neurons (e.g., for 10 classes)
        )
        
        self.relu = ReLU(inplace=True)  # Activation function
        self.maxpool = Sequential(MaxPool2d(kernel_size=2))  # Max pooling layer

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x: Input tensor.
        Returns:
            Output tensor after passing through the model.
        """
        # Pass input through each convolutional block and max-pooling
        out = self.block1(x)
        out = self.maxpool(out)
        
        out = self.block2(out)
        out = self.maxpool(out)
        
        out = self.block3(out)
        out = self.maxpool(out)
        
        out = self.block4(out)
        out = self.maxpool(out)
        
        # Pass the result through the classifier
        out = self.classifier(out)
        return out
