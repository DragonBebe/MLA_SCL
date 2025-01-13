import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, kernel_size, stride, padding):
    """Create a convolutional block with Conv2D, BatchNorm, and LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True)
    )


class ResidualBlock(nn.Module):
    """A standard Residual Block with two convolutional layers."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            conv_block(channels, channels // 2, 1, 1, 0),  # Reduce channel dimension
            conv_block(channels // 2, channels, 3, 1, 1)   # Restore channel dimension
        )

    def forward(self, x):
        return x + self.block(x)  # Add input (identity) for residual connection


class CSPResidualBlock(nn.Module):
    """CSP Residual Block for Cross-Stage Partial Connections."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(CSPResidualBlock, self).__init__()
        mid_channels = out_channels // 2
        self.split_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride)  # Split path
        self.residual_blocks = nn.Sequential(  # Residual path
            conv_block(mid_channels, mid_channels, 3, 1, 1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1)
        )
        self.transition_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride)  # Transition path
        self.merge_conv = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1, stride=1)  # Merge path

    def forward(self, x):
        transition = self.transition_conv(x)  # Transition path
        split = self.split_conv(x)  # Split path
        split = self.residual_blocks(split)  # Process residual path
        merged = torch.cat([split, transition], dim=1)  # Concatenate split and transition paths
        return self.merge_conv(merged)  # Final output


class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling for capturing multi-scale spatial features."""
    def __init__(self, in_channels, pool_sizes=(1, 5, 9, 13)):
        super(SpatialPyramidPooling, self).__init__()
        self.pools = nn.ModuleList([  # Create multi-scale pooling layers
            nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2)
            for size in pool_sizes
        ])
        self.conv = nn.Conv2d(in_channels * len(pool_sizes), in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        pooled_features = [pool(x) for pool in self.pools]  # Apply pooling at different scales
        concat_features = torch.cat(pooled_features, dim=1)  # Concatenate pooled features
        return self.conv(concat_features)  # Reduce concatenated channels


class CSPDarknet53(nn.Module):
    """CSPDarknet53 with enhanced SPP module for object detection."""
    def __init__(self, num_classes=10, input_resolution=224, num_blocks=[2, 2, 8, 8, 4], width_scale=1.0):
        super(CSPDarknet53, self).__init__()
        self.input_resolution = input_resolution  # Input image resolution
        self.initial = nn.Sequential(  # Initial layers for downsampling
            conv_block(3, int(32 * width_scale), 3, 1, 1),
            conv_block(int(32 * width_scale), int(64 * width_scale), 3, 2, 1)
        )
        self.residual_blocks = nn.ModuleList([  # CSP residual blocks at multiple scales
            self._make_csp_layer(int(64 * width_scale), int(128 * width_scale), num_blocks[0]),
            self._make_csp_layer(int(128 * width_scale), int(256 * width_scale), num_blocks[1]),
            self._make_csp_layer(int(256 * width_scale), int(512 * width_scale), num_blocks[2]),
            self._make_csp_layer(int(512 * width_scale), int(1024 * width_scale), num_blocks[3]),
            self._make_csp_layer(int(1024 * width_scale), int(2048 * width_scale), num_blocks[4])
        ])
        self.spp = SpatialPyramidPooling(in_channels=int(2048 * width_scale))  # Add SPP for spatial feature aggregation
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(int(2048 * width_scale), num_classes)  # Fully connected layer for classification

    def _make_csp_layer(self, in_channels, out_channels, num_blocks):
        """Create a CSP layer with a specified number of residual blocks."""
        layers = [CSPResidualBlock(in_channels, out_channels, stride=2)]  # Downsample in the first block
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels))  # Add standard residual blocks
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass for CSPDarknet53."""
        x = F.interpolate(x, size=(self.input_resolution, self.input_resolution), mode="bilinear", align_corners=False)
        x = self.initial(x)
        for layer in self.residual_blocks:  # Pass through CSP layers
            x = layer(x)
        x = self.spp(x)  # Apply spatial pyramid pooling
        x = self.avgpool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten for fully connected layer
        return self.fc(x)  # Classification output
