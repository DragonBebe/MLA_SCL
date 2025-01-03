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
            params: 模型参数。
            lr: 基础学习率。
            momentum: 动量。
            weight_decay: 权重衰减。
            eta: 缩放系数，用于控制 adaptive_lr。
            epsilon: 用于数值稳定性的小常数，防止除零。
            min_lr: adaptive_lr 的最小值，避免浮点数精度问题。
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

                # 添加权重衰减到梯度
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # 计算参数和梯度的范数
                param_norm = torch.norm(p)
                grad_norm = torch.norm(grad)

                # 避免参数和梯度全为零导致的除零错误
                if param_norm == 0 or grad_norm == 0:
                    # print(f"Warning: Zero norm detected in param or grad during LARS update.")
                    continue

                # 计算 adaptive_lr
                adaptive_lr = group['eta'] * param_norm / (grad_norm + group['epsilon'])

                # 应用最小学习率下限
                adaptive_lr = max(adaptive_lr, group['min_lr'])

                # 更新参数
                p.add_(grad, alpha=-group['lr'] * adaptive_lr)

        return loss

# 动态标准化参数和数据增强
def set_loader(opt):
    """
    根据配置动态加载数据集并应用数据增强和标准化。
    Args:
        opt (dict): 包含数据集名称、路径、输入分辨率等的配置字典。
    Returns:
        DataLoader: 训练数据加载器。
    """
    # 根据数据集名称动态设置数据增强和标准化
    transform = TwoCropTransform(get_base_transform(opt['dataset_name'], opt['input_resolution']))

    # 数据集映射
    dataset_dict = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'imagenet': datasets.ImageFolder
    }

    # 获取对应数据集类
    dataset_class = dataset_dict.get(opt['dataset_name'])
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {opt['dataset_name']}")

    # 加载数据集
    if opt['dataset_name'] in ['cifar10', 'cifar100']:
        train_dataset = dataset_class(root=opt['dataset'], train=True, download=True, transform=transform)
    elif opt['dataset_name'] == 'imagenet':
        train_dataset = dataset_class(root=opt['dataset'], transform=transform)

    # 创建数据加载器
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
    创建 Warmup + 余弦退火学习率调度器

    调度器包含两个阶段：
    1. **Warmup 阶段**：
        - 在前 `warmup_epochs` 个 epoch 内，学习率从 0 增加到设定的初始学习率。
        - 学习率按线性增长，公式为：`lr = base_lr * (epoch + 1) / warmup_epochs`。

    2. **余弦退火阶段**：
        - 从 `warmup_epochs` 到 `total_epochs`，学习率按照余弦退火公式逐渐减少。
        - 公式为：`lr = base_lr * 0.5 * (1 + cos(pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))`。

    参数说明：
    - `warmup_epochs`: 学习率线性增长的阶段，用于避免学习率过快调整导致的不稳定训练。
    - `total_epochs`: 总训练 epoch 数，影响余弦退火阶段的结束位置。

    示例：
    - 假设 `warmup_epochs=5`, `total_epochs=100`:
      - 第 0-4 个 epoch：学习率线性从 0 增加到初始值。
      - 第 5-99 个 epoch：学习率按照余弦函数逐渐减小。

    注意：
    - 如果 `warmup_epochs` 设置过大，可能会延缓训练的收敛。
    - `total_epochs` 的调整会影响余弦退火阶段的曲线形状，应与训练目标和任务规模匹配。

    Args:
        optimizer (Optimizer): 优化器对象。
        warmup_epochs (int): Warmup 阶段的 epoch 数。
        total_epochs (int): 总训练的 epoch 数。

    Returns:
        LambdaLR: 自定义的学习率调度器。
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos((epoch - warmup_epochs) / (total_epochs - warmup_epochs) * math.pi))

    return LambdaLR(optimizer, lr_lambda)




def save_best_model(model, opt, epoch, loss, save_root, best_loss, last_save_path):
    """
    保存性能最佳的模型，并删除旧的最佳模型。
    """
    if loss < best_loss:
        model_dir = os.path.join(save_root, opt['model_type'])
        os.makedirs(model_dir, exist_ok=True)

        # 生成新模型的保存路径
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(
            model_dir,
            f"{opt['model_type']}_{opt['dataset_name']}_feat{opt['feature_dim']}_batch{opt['batch_size']}_epoch{epoch}_loss{loss:.4f}_{timestamp}.pth"
        )

        # 保存新模型
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": opt
        }, save_path)
        print(f"New best model saved to {save_path}")

        # 删除旧模型（如果存在）
        if last_save_path and os.path.exists(last_save_path):
            os.remove(last_save_path)
            print(f"Deleted previous best model: {last_save_path}")

        return loss, save_path  # 更新最佳损失和保存路径
    else:
        return best_loss, last_save_path


def train(train_loader, model, criterion, optimizer, opt, device, epoch=None):
    """
    对比学习预训练的训练函数。
    支持保存性能最佳的模型，并删除之前性能较差的模型。
    """
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 累积损失初始化
    total_steps = len(train_loader)  # 总步数
    best_loss = opt.get("best_loss", float('inf'))  # 初始化最佳损失
    last_save_path = opt.get("last_save_path", None)  # 初始化保存路径

    train_bar = tqdm(enumerate(train_loader), total=total_steps, desc="Training", leave=False)
    for step, (inputs, labels) in train_bar:
        # 数据预处理：拼接两种图像增强结果
        if isinstance(inputs, list) and len(inputs) == 2:
            inputs = torch.cat([inputs[0], inputs[1]], dim=0).to(device)
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 梯度清零
        features = model(inputs)  # 前向传播

        # 分割对比特征并重新组合
        f1, f2 = torch.split(features, features.size(0) // 2, dim=0)
        contrastive_features = torch.stack([f1, f2], dim=1)

        # 对齐标签和特征的尺寸
        if contrastive_features.size(0) != labels.size(0):
            labels = labels[:contrastive_features.size(0)]

        # 计算损失
        loss = criterion(contrastive_features, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        running_loss += loss.item()  # 累积损失
        train_bar.set_postfix(loss=loss.item())  # 更新进度条显示

    epoch_loss = running_loss / len(train_loader)  # 计算平均损失
    print(f"--- Summary for Epoch [{epoch + 1}] ---")
    print(f"    Average Loss: {epoch_loss:.4f}")

    # 更新 opt 中的状态
    opt["best_loss"] = best_loss
    opt["last_save_path"] = last_save_path

    return epoch_loss

