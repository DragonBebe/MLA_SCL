import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from losses import SupConLoss_in, SupConLoss_out
from models import ResNet34, ResNet50, ResNet101, ResNet200, CSPDarknet53, SupConResNetFactory, SupConResNetFactory_CSPDarknet53
from data_augmentation import TwoCropTransform, get_base_transform
from torchvision import datasets
from torch.optim.optimizer import Optimizer

from tqdm import tqdm
import datetime

class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling (LARS) optimizer with improved robustness."""

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, eta=0.001, epsilon=1e-8, min_lr=1e-6):
        """
        Args:
            params: Model parameters.
            lr: Base learning rate.
            momentum: Momentum.
            weight_decay: Weight decay.
            eta: Scaling factor for controlling adaptive_lr.
            epsilon: A small constant for numerical stability to avoid division by zero.
            min_lr: Minimum value for adaptive_lr to avoid floating-point precision issues.
        """
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, epsilon=epsilon, min_lr=min_lr)
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # Add weight decay to the gradient
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Compute norms of parameters and gradients
                param_norm = torch.norm(p)
                grad_norm = torch.norm(grad)

                # Avoid division by zero if both parameter and gradient norms are zero
                if param_norm == 0 or grad_norm == 0:
                    # print(f"Warning: Zero norm detected in param or grad during LARS update.")
                    continue

                # Compute adaptive_lr
                adaptive_lr = group['eta'] * param_norm / (grad_norm + group['epsilon'])

                # Apply minimum learning rate limit
                adaptive_lr = max(adaptive_lr, group['min_lr'])

                # Update parameters
                p.add_(grad, alpha=-group['lr'] * adaptive_lr)

        return loss

# Dynamically normalize parameters and data augmentation
def set_loader(opt):
    """
    Dynamically load dataset and apply data augmentation and normalization based on configuration.
    Args:
        opt (dict): Configuration dictionary containing dataset name, path, input resolution, etc.
    Returns:
        DataLoader: Training data loader.
    """
    # Dynamically set data augmentation and normalization based on dataset name
    transform = TwoCropTransform(get_base_transform(opt['dataset_name'], opt['input_resolution']))

    # Dataset mapping
    dataset_dict = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'imagenet': datasets.ImageFolder
    }

    # Get corresponding dataset class
    dataset_class = dataset_dict.get(opt['dataset_name'])
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {opt['dataset_name']}")

    # Load dataset
    if opt['dataset_name'] in ['cifar10', 'cifar100']:
        train_dataset = dataset_class(root=opt['dataset'], train=True, download=True, transform=transform)
    elif opt['dataset_name'] == 'imagenet':
        train_dataset = dataset_class(root=opt['dataset'], transform=transform)

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt['batch_size'],
        shuffle=True,
        num_workers=opt.get('num_workers', 2),
        pin_memory=True,
        persistent_workers=opt.get('num_workers', 0) > 0
    )
    return train_loader


def set_model(opt):
    model_dict = {
        'ResNet34': lambda: ResNet34(num_classes=opt['num_classes']),
        'ResNet50': lambda: ResNet50(num_classes=opt['num_classes']),
        'ResNet101': lambda: ResNet101(num_classes=opt['num_classes']),
        'ResNet200': lambda: ResNet200(num_classes=opt['num_classes']),
        'CSPDarknet53': lambda: CSPDarknet53(num_classes=10, input_resolution=32, num_blocks=[1, 2, 8, 8, 4]),
    }
    base_model_func = model_dict.get(opt['model_type'])

    if base_model_func is None:
        raise ValueError(f"Unknown model type: {opt['model_type']}")

    if opt['model_type'] == 'CSPDarknet53':
        model = SupConResNetFactory_CSPDarknet53(
            base_model_func=base_model_func,
            feature_dim=opt['feature_dim'],
        )
    else:
        model = SupConResNetFactory(
            base_model_func=base_model_func,
            feature_dim=opt['feature_dim'],
        )

    device = torch.device(f"cuda:{opt['gpu']}" if torch.cuda.is_available() and opt['gpu'] is not None else "cpu")
    model = model.to(device)

    if opt['loss_type'] == 'supout':
        criterion = SupConLoss_out(temperature=opt['temp']).to(device)
    elif opt['loss_type'] == 'supin':
        criterion = SupConLoss_in().to(device)
    else:
        raise ValueError(f"Unknown loss type: {opt['loss_type']}")

    return model, criterion, device


def create_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Create Warmup + Cosine Annealing Learning Rate Scheduler

    The scheduler consists of two phases:
    1. **Warmup Phase**:
        - During the first `warmup_epochs`, the learning rate increases linearly from 0 to the initial learning rate.
        - Learning rate formula: `lr = base_lr * (epoch + 1) / warmup_epochs`.

    2. **Cosine Annealing Phase**:
        - From `warmup_epochs` to `total_epochs`, the learning rate gradually decreases according to the cosine annealing formula.
        - Formula: `lr = base_lr * 0.5 * (1 + cos(pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))`.

    Parameter Descriptions:
    - `warmup_epochs`: Number of epochs for linear increase of learning rate to prevent unstable training.
    - `total_epochs`: Total number of training epochs, affects the ending position of cosine annealing phase.

    Example:
    - Assuming `warmup_epochs=5`, `total_epochs=100`:
      - Epochs 0-4: Learning rate increases linearly to the initial value.
      - Epochs 5-99: Learning rate decreases gradually based on the cosine function.

    Notes:
    - If `warmup_epochs` is set too high, it may delay training convergence.
    - Adjusting `total_epochs` affects the curve shape of the cosine annealing phase and should match the training objective and task scale.

    Args:
        optimizer (Optimizer): Optimizer object.
        warmup_epochs (int): Number of epochs for the warmup phase.
        total_epochs (int): Total number of training epochs.

    Returns:
        LambdaLR: Custom learning rate scheduler.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos((epoch - warmup_epochs) / (total_epochs - warmup_epochs) * math.pi))

    return LambdaLR(optimizer, lr_lambda)


def save_best_model(model, opt, epoch, loss, save_root, best_loss, last_save_path):
    """
    Save the best performing model and delete the old best model.
    """
    if loss < best_loss:
        model_dir = os.path.join(save_root, opt['model_type'])
        os.makedirs(model_dir, exist_ok=True)

        # Generate save path for the new model
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(
            model_dir,
            f"{opt['model_type']}_{opt['dataset_name']}_feat{opt['feature_dim']}_batch{opt['batch_size']}_epoch{epoch}_loss{loss:.4f}_{timestamp}.pth"
        )

        # Save the new model
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": opt
        }, save_path)
        print(f"New best model saved to {save_path}")

        # Delete the old model (if it exists)
        if last_save_path and os.path.exists(last_save_path):
            os.remove(last_save_path)
            print(f"Deleted previous best model: {last_save_path}")

        return loss, save_path  # Update best loss and save path
    else:
        return best_loss, last_save_path


def train(train_loader, model, criterion, optimizer, opt, device, epoch=None):
    """
    Training function for contrastive learning pretraining.
    Supports saving the best performing model and deleting previous inferior models.
    """
    model.train()  # Set model to training mode
    running_loss = 0.0  # Initialize cumulative loss
    total_steps = len(train_loader)  # Total steps
    best_loss = opt.get("best_loss", float('inf'))  # Initialize best loss
    last_save_path = opt.get("last_save_path", None)  # Initialize save path

    train_bar = tqdm(enumerate(train_loader), total=total_steps, desc="Training", leave=False)
    for step, (inputs, labels) in train_bar:
        # Data preprocessing: concatenate two image augmentation results
        if isinstance(inputs, list) and len(inputs) == 2:
            inputs = torch.cat([inputs[0], inputs[1]], dim=0).to(device)
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Clear gradients
        features = model(inputs)  # Forward propagation

        # Split contrastive features and recombine
        f1, f2 = torch.split(features, features.size(0) // 2, dim=0)
        contrastive_features = torch.stack([f1, f2], dim=1)

        # Align labels and features size
        if contrastive_features.size(0) != labels.size(0):
            labels = labels[:contrastive_features.size(0)]

        # Compute loss
        loss = criterion(contrastive_features, labels)
        loss.backward()  # Backward propagation
        optimizer.step()  # Parameter update

        running_loss += loss.item()  # Accumulate loss
        train_bar.set_postfix(loss=loss.item())  # Update progress bar display

    epoch_loss = running_loss / len(train_loader)  # Compute average loss
    print(f"--- Summary for Epoch [{epoch + 1}] ---")
    print(f"    Average Loss: {epoch_loss:.4f}")

    # Update state in opt
    opt["best_loss"] = best_loss
    opt["last_save_path"] = last_save_path

    return epoch_loss
