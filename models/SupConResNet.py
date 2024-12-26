import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConResNet(nn.Module):
    """
    A ResNet model extended with a projection head for contrastive learning.
    """
    def __init__(self, base_model, feature_dim=128, dim_in=None):
        """
        初始化 SupConResNet 模型。
        参数:
            base_model (nn.Module): 基础的 ResNet 模型（如 ResNet34, ResNet50）。
            feature_dim (int): 投影头的输出特征维度。
            dim_in (int): 投影头的输入特征维度。
                          如果为 None，则需要在使用 ResNet 变体时提供
        """
        super(SupConResNet, self).__init__()
        self.encoder = base_model  # Base ResNet backbone

        if dim_in is None:
            raise ValueError("The input feature dimension (dim_in) must be specified for the projection head.")

        print(f"Encoder output feature dimension: {dim_in}")  # Debugging line

        # Projection head (MLP)
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.GELU(),  # 使用 GELU 激活函数
            nn.Linear(dim_in, dim_in),
            nn.GELU(),  # 增加一层全连接层和 GELU 激活函数
            nn.Linear(dim_in, feature_dim)
        )
        # self.head = nn.Sequential(
        #     nn.Linear(dim_in, dim_in),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dim_in, feature_dim)
        # )

    def forward(self, x):
        """
        Forward pass for the SupConResNet.
        Args:
            x: Input data, can be a list for multi-view contrastive learning or a tensor for single view.
        Returns:
            Normalized features for contrastive loss.
        """
        if isinstance(x, list) and len(x) == 2:  # Multi-view input
            feat1 = self.encoder(x[0])
            feat2 = self.encoder(x[1])
            feat1 = F.normalize(self.head(feat1), dim=1)  # Normalize
            feat2 = F.normalize(self.head(feat2), dim=1)  # Normalize
            return [feat1, feat2]

        # Single view input
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)  # Normalize
        return feat

def determine_feature_dim(model):
    """
    Dynamically determine the output feature dimension of a model.
    """
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 32, 32)  # Dummy input for CIFAR-10
            features = model(dummy_input)
            return features.shape[1]
    finally:
        model.train()  # Always restore training mode, even if an exception occurs

def SupConResNetFactory(base_model_func, feature_dim=128):
    """
    Factory function to create SupConResNet models based on ResNet variants.
    """
    base_model = base_model_func()
    base_model.fc = nn.Identity()  # 移除分类头
    dim_in = determine_feature_dim(base_model)  # 动态确定特征维度
    return SupConResNet(base_model, feature_dim, dim_in)

def SupConResNetFactory_CSPDarknet53(base_model_func=None, feature_dim=128):
    """
    Factory function to create SupConResNet models based on various backbone models.
    """
    if base_model_func is None:
        raise ValueError("base_model_func must be provided.")

    # Initialize the backbone
    base_model = base_model_func()
    base_model.fc = nn.Identity()  # Remove classification head
    base_model.avgpool = nn.Identity()  # Remove final pooling

    # Dynamically determine feature dimensions
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32)  # For CIFAR-like input
        features = base_model(dummy_input)
        feature_dim_in = features.shape[1]

    return SupConResNet(base_model, feature_dim=feature_dim, dim_in=feature_dim_in)


