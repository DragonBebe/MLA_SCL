import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import ResNet34, ResNet50,ResNet101 # 确保路径正确
import argparse

def load_test_data(batch_size=32,data_type = "Cifar10"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    # 根据 data_type 动态选择test_set
    if data_type == "Cifar10":
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif data_type == "Cifar100":
        testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported data type: {data_type}. Choose from Cifar10, Cifar100.")

    # testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return testloader


def compute_top_k_accuracy(output, target, k=5):
    with torch.no_grad():
        max_k_preds = torch.topk(output, k, dim=1).indices  # 获取前 k 的预测索引
        correct = max_k_preds.eq(target.view(-1, 1).expand_as(max_k_preds))  # 检查是否包含正确标签
        return correct.any(dim=1).float().sum().item()  # 转换为布尔值后求和


def test_model(model_path, device,model_type,data_type):

    # 根据 data_type 动态选择类别数量
    if data_type == "Cifar10":
        num_classes = 10
    elif data_type == "Cifar100":
        num_classes = 100
    else:
        raise ValueError(f"Unsupported data type: {data_type}. Choose from Cifar10, Cifar100.")

    # 根据 model_type 动态选择模型
    if model_type == "ResNet34":
        model = ResNet34(num_classes=num_classes).to(device)
    elif model_type == "ResNet50":
        model = ResNet50(num_classes=num_classes).to(device)
    elif model_type == "ResNet101":
        model = ResNet101(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from ResNet34, ResNet50, ResNet101.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    testloader = load_test_data(data_type=data_type)

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


def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Test a trained model on CIFAR-10")
    parser.add_argument("--modir", type=str, required=True, help="Path to the saved model file")
    parser.add_argument("--model", type=str, default="ResNet34",help="Model type: ResNet34, ResNet50, or ResNet101")
    parser.add_argument("--data", type=str, default="Cifar10", help="Data type: Cifar10, or Cifar100")
    args = parser.parse_args()

    model_path = args.modir
    model_type = args.model
    data_type = args.data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = "./saved_models/classification/scratch/ResNet34_cifar10_batch256_valAcc85.96_20250103-012057.pth"

    test_model(model_path, device,model_type,data_type)

if __name__ == "__main__":
    main()

    # 指令示例
    #python test_scratch_classifier.py --model ResNet34 --data Cifar10 --modir ./saved_models/classification/scratch/ResNet34_cifar10_batch256_valAcc85.96_20250103-012057.pth


