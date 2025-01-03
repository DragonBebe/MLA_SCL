# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from models import ResNet34, ResNet50  # 确保路径正确
#
# import os
#
# # 加载 CIFAR-10 测试集
# def load_test_data(batch_size=32):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # 标准化
#     ])
#     testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
#     testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
#     return testloader
#
# # 测试模型性能
# def test_model(model_path, device):
#     # 加载模型
#     model = ResNet34(num_classes=100).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()  # 设置为评估模式
#
#     # 加载测试数据
#     testloader = load_test_data()
#
#     # 测试逻辑
#     correct = 0
#     total = 0
#     criterion = nn.CrossEntropyLoss()
#     running_loss = 0.0
#
#     with torch.no_grad():
#         for inputs, labels in testloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             running_loss += loss.item()
#
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total
#     avg_loss = running_loss / len(testloader)
#     print(f"Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")
#
# if __name__ == "__main__":
#     # 设置设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_path = "./saved_models/classification/scratch/ResNet34_cifar100_batch128_valAcc54.87_20250102-165250.pth"
#
#     # 测试模型
#     test_model(model_path, device)

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import ResNet34, ResNet50  # 确保路径正确
# import argparse

def load_test_data(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return testloader


def compute_top_k_accuracy(output, target, k=5):
    with torch.no_grad():
        max_k_preds = torch.topk(output, k, dim=1).indices  # 获取前 k 的预测索引
        correct = max_k_preds.eq(target.view(-1, 1).expand_as(max_k_preds))  # 检查是否包含正确标签
        return correct.any(dim=1).float().sum().item()  # 转换为布尔值后求和


def test_model(model_path, device):
    model = ResNet34(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    testloader = load_test_data()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Top-1 准确率
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()

            # Top-5 准确率
            correct_top5 += compute_top_k_accuracy(outputs, labels, k=5)

            total += labels.size(0)

    accuracy_top1 = 100 * correct_top1 / total
    accuracy_top5 = 100 * correct_top5 / total
    avg_loss = running_loss / len(testloader)

    print(f"Test Loss: {avg_loss:.4f}, Top-1 Accuracy: {accuracy_top1:.2f}%, Top-5 Accuracy: {accuracy_top5:.2f}%")


if __name__ == "__main__":
    # 添加命令行参数解析
    # parser = argparse.ArgumentParser(description="Test a trained model on CIFAR-10")
    # parser.add_argument("--modir", type=str, required=True, help="Path to the saved model file")
    # args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./saved_models/classification/scratch/ResNet34_cifar10_batch32_valAcc86.98_20241217-013556.pth"
    # model_path = args.modir
    test_model(model_path, device)

    # 当前最佳：
    #
    # cifar10
    # ResNet34_cifar10_batch32_valAcc86.98_20241217-013556，Test Loss: 0.4979, Top-1 Accuracy: 89.94%, Top-5 Accuracy: 99.61%， 新的resnet34，传统MLP


