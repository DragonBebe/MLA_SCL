import os
import torch
import torch.nn as nn
from sympy import false
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from losses.SupOut import SupConLoss_out

from models import ResModel, ResNet34, ResNeXt101_32x8d, WideResNet_28_10, ResNet50, ResNet101, ResNet200, CSPDarknet53Classifier

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
    elif opt['dataset_name'] == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt['dataset'], train=True, download=False, transform=transform)
    elif opt['dataset_name'] == 'imagenet':
        train_dataset = datasets.ImageNet(root=opt['dataset'], split='train', download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {opt['dataset_name']}")

    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=2)
    return train_loader

def set_model(opt):
    model_dict = {
        'resnet34': ResNet34(),
        'ResNeXt101': ResNeXt101_32x8d(),
        'WideResNet': WideResNet_28_10(),
        'resnet_HikVision': ResModel(),
    }
    model = model_dict.get(opt['model_type'])
    if model is None:
        raise ValueError(f"Unknown model type: {opt['model_type']}")

    device = torch.device(f"cuda:{opt['gpu']}" if torch.cuda.is_available() and opt['gpu'] is not None else "cpu")
    model = model.to(device)

    # 根据 loss_type 参数选择损失函数
    if opt['loss_type'] == 'supcon_in':
        criterion = SupConLoss_out(temperature=opt['temp']).to(device)
    elif opt['loss_type'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss().to(device)

    # elif opt['loss_type'] == 'supcon_out':
    #     criterion = SupConLoss_In().to(device)  # 假设 SupInLoss 是你未来的自定义损失函数

    else:
        raise ValueError(f"Unknown loss type: {opt['loss_type']}")

    return model, criterion, device

def adjust_learning_rate(optimizer, epoch, opt):
    if epoch in [4, 8, 12]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5


def train(train_loader, model, criterion, optimizer, opt, writer, device):
    model.train()
    for epoch in range(1, opt['epochs'] + 1):
        adjust_learning_rate(optimizer, epoch, opt)
        running_loss = 0.0

        for step, (inputs, labels) in enumerate(train_loader):
            # 使用传入的 device，将数据和标签转移到指定设备
            if isinstance(inputs, list) and len(inputs) == 2:
                inputs = torch.cat([inputs[0], inputs[1]], dim=0).to(device)
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            features = model(inputs)
            f1, f2 = torch.split(features, features.size(0) // 2, dim=0)
            features = torch.stack([f1, f2], dim=1)
            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch}/{opt["epochs"]}], Step [{step + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        writer.add_scalar('Epoch Loss', running_loss / len(train_loader), epoch)

        # 保存检查点
        if epoch % opt['save_freq'] == 0:
            save_path = os.path.join(opt['model_save_dir'], f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved at {save_path}")

    return running_loss / len(train_loader)


