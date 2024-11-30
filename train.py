import os
import torch
import torch.nn as nn
from sympy import false
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from losses import SupConLoss_in, SupConLoss_out, CrossEntropyLoss

from models import ResModel, ResNet34, ResNeXt101_32x8d, WideResNet_28_10, ResNet50, ResNet101, ResNet200, \
    CSPDarknet53Classifier

from data_augmentation.data_augmentation_1 import TwoCropTransform, get_base_transform


from torchvision import datasets


def set_loader(opt):
    if opt['augmentation'] == 'basic':
        transform = TwoCropTransform(get_base_transform())
    # elif opt['augmentation'] == 'advanced':
    #     transform = TwoCropTransform(get_advanced_transform())
    else:
        raise ValueError(f"Unknown augmentation type: {opt['augmentation']}")

        # 根据数据集名称选择数据集
    if opt['dataset_name'] == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt['dataset'], train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10(root=opt['dataset'], train=False, download=True, transform=transform)
    elif opt['dataset_name'] == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt['dataset'], train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR100(root=opt['dataset'], train=False, download=True, transform=transform)
    # elif opt['dataset_name'] == 'imagenet':
    #     train_dataset = datasets.ImageNet(root=opt['dataset'], split='train', download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {opt['dataset_name']}")

    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=2)
    return train_loader, test_loader


def set_model(opt):
    model_dict = {
        'ResNet34': ResNet34(num_classes=opt['num_classes']),
        'ResNeXt101': ResNeXt101_32x8d(),
        'WideResNet': WideResNet_28_10(),

    }
    model = model_dict.get(opt['model_type'])
    if model is None:
        raise ValueError(f"Unknown model type: {opt['model_type']}")

    device = torch.device(f"cuda:{opt['gpu']}" if torch.cuda.is_available() and opt['gpu'] is not None else "cpu")
    model = model.to(device)

    # 根据 loss_type 参数选择损失函数
    if opt['loss_type'] == 'supout':
        criterion = SupConLoss_out(temperature=opt['temp']).to(device)
    elif opt['loss_type'] == 'cross_entropy':
        criterion = CrossEntropyLoss().to(device)
    elif opt['loss_type'] == 'supin':
        criterion = SupConLoss_in().to(device)

    else:
        raise ValueError(f"Unknown loss type: {opt['loss_type']}")

    return model, criterion, device


def adjust_learning_rate(optimizer, epoch, opt):
    if epoch in [4, 8, 12]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5


def calculate_accuracy(features, labels):
    """
    计算预测的准确率。

    Args:
        features (torch.Tensor): 模型的输出张量，通常是 logits。
        labels (torch.Tensor): 真实的标签。

    Returns:
        tuple: (正确预测的样本数量, 总样本数量)
    """
    _, predicted = features.max(1)  # 获取每个样本的预测类别
    correct = (predicted == labels).sum().item()  # 统计预测正确的样本数
    total = labels.size(0)  # 总样本数
    return correct, total





def train(train_loader, model, criterion, optimizer, opt, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (inputs, labels) in enumerate(train_loader):
        # 数据预处理
        if isinstance(inputs, list) and len(inputs) == 2:
            inputs = torch.cat([inputs[0], inputs[1]], dim=0).to(device)
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        features = model(inputs)

        # 对比损失特征处理
        f1, f2 = torch.split(features, features.size(0) // 2, dim=0)
        contrastive_features = torch.stack([f1, f2], dim=1)  # [batch_size // 2, 2, feature_dim]

        # 检查特征和标签是否匹配
        if contrastive_features.size(0) != labels.size(0):
            labels = labels[:contrastive_features.size(0)]  # 截取标签以匹配特征

        # 计算损失
        loss = criterion(contrastive_features, labels)
        loss.backward()
        optimizer.step()

        # 累加损失
        running_loss += loss.item()

        # 准确率计算（使用 f1）
        _, predicted = f1.max(1)
        correct += (predicted == labels[:f1.size(0)]).sum().item()  # 截取标签匹配特征大小
        total += labels.size(0) // 2  # 原始标签大小

        # 打印训练进度
        if (step + 1) % 100 == 0:

            print(f"Step [{step + 1}/{len(train_loader)}], Loss: {running_loss / (step + 1):.4f}")

    # 返回损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy








