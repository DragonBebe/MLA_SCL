import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConResNet(nn.Module):
    """
    A ResNet model extended with a projection head for contrastive learning.
    """
    def __init__(self, base_model, feature_dim=128, dim_in=None):
        """
        Initialize the SupConResNet model.
        
        Args:
            base_model (nn.Module): The backbone ResNet model (e.g., ResNet34, ResNet50).
            feature_dim (int): Output feature dimension of the projection head.
            dim_in (int): Input feature dimension to the projection head. 
                          If None, this must be specified manually.
        """
        super(SupConResNet, self).__init__()
        self.encoder = base_model  # Backbone ResNet model

        # Ensure the input feature dimension is provided
        if dim_in is None:
            raise ValueError("The input feature dimension (dim_in) must be specified for the projection head.")

        print(f"Encoder output feature dimension: {dim_in}")  # Debugging log

        # Define the projection head (MLP)
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),  # First linear layer
            nn.GELU(),  # Activation function
            nn.Linear(dim_in, dim_in),  # Second linear layer
            nn.GELU(),  # Activation function
            nn.Linear(dim_in, feature_dim)  # Final linear layer to project to feature_dim
        )

    def forward(self, x):
        """
        Forward pass of the SupConResNet.
        
        Args:
            x: Input data. Can be a list (for multi-view contrastive learning) 
               or a tensor (for single-view).
        Returns:
            Normalized features for contrastive loss.
        """
        if isinstance(x, list) and len(x) == 2:  # Handle multi-view input
            feat1 = self.encoder(x[0])  # Encode first view
            feat2 = self.encoder(x[1])  # Encode second view
            feat1 = F.normalize(self.head(feat1), dim=1)  # Normalize features from projection head
            feat2 = F.normalize(self.head(feat2), dim=1)
            return [feat1, feat2]

        # Handle single-view input
        feat = self.encoder(x)  # Encode input
        feat = F.normalize(self.head(feat), dim=1)  # Normalize features
        return feat

def determine_feature_dim(model):
    """
    Dynamically determine the output feature dimension of a model.
    """
    try:
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Use a dummy input to determine the output feature dimension
            dummy_input = torch.randn(2, 3, 32, 32)  # Example: for CIFAR-10
            features = model(dummy_input)
            return features.shape[1]  # Return the feature dimension
    finally:
        model.train()  # Restore the model to training mode

def SupConResNetFactory(base_model_func, feature_dim=128):
    """
    Factory function to create a SupConResNet model with a ResNet variant.
    
    Args:
        base_model_func: Function to initialize a ResNet model (e.g., torchvision.models.resnet50).
        feature_dim (int): Output feature dimension of the projection head.
    Returns:
        SupConResNet instance.
    """
    base_model = base_model_func()  # Initialize the base model
    base_model.fc = nn.Identity()  # Remove the final classification layer
    dim_in = determine_feature_dim(base_model)  # Dynamically determine feature dimension
    return SupConResNet(base_model, feature_dim, dim_in)

# def SupConResNetFactory_CSPDarknet53(base_model_func=None, feature_dim=128):
#     """
#     Factory function to create a SupConResNet model based on a custom backbone model.
    
#     Args:
#         base_model_func: Function to initialize the backbone model.
#         feature_dim (int): Output feature dimension of the projection head.
#     Returns:
#         SupConResNet instance.
#     """
#     if base_model_func is None:
#         raise ValueError("base_model_func must be provided.")

#     # Initialize the backbone
#     base_model = base_model_func()
#     base_model.fc = nn.Identity()  # Remove classification head
#     base_model.avgpool = nn.Identity()  # Remove final pooling layer

#     # Dynamically determine the feature dimension
#     with torch.no_grad():
#         dummy_input = torch.randn(1, 3, 32, 32)  # Example input for CIFAR-like data
#         features = base_model(dummy_input)
#         feature_dim_in = features.shape[1]  # Feature dimension from the backbone

#     return SupConResNet(base_model, feature_dim=feature_dim, dim_in=feature_dim_in)
