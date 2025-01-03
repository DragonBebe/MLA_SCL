import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import ResNet34, SupConResNetFactory,ResNet50,ResNet101
import argparse

# 线性分类器
class LinearClassifier(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():  # 冻结 Backbone
            features = self.backbone.encoder(x)
        out = self.fc(features)
        return out

# 加载测试集
def load_test_data(batch_size=32,data_type = "Cifar10"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # 根据 data_type 动态选择类别数量
    if data_type == "Cifar10":
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif data_type == "Cifar100":
        testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported data type: {data_type}. Choose from Cifar10, Cifar100.")

    # testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return testloader

# 计算 Top-k 准确率
def compute_top_k_accuracy(output, target, k=5):
    with torch.no_grad():
        max_k_preds = torch.topk(output, k, dim=1).indices  # 获取前 k 的预测索引
        correct = max_k_preds.eq(target.view(-1, 1).expand_as(max_k_preds))  # 检查是否包含正确标签
        return correct.any(dim=1).float().sum().item()  # 转换为布尔值后求和

# 测试模型性能
def test_model(model_path, device,model_type,data_type):
    print("Loading model...")

    # 构建模型
    # base_model_func = lambda: ResNet34()  # Backbone

    # 根据 model_type 动态选择模型
    if model_type == "ResNet34":
        base_model_func = lambda: ResNet34()
    elif model_type == "ResNet50":
        base_model_func = lambda: ResNet50()
    elif model_type == "ResNet101":
        base_model_func = lambda: ResNet101()
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from ResNet34, ResNet50, ResNet101.")

    # 根据 data_type 动态选择类别数量
    if data_type == "Cifar10":
        num_classes = 10
    elif data_type == "Cifar100":
        num_classes = 10
    else:
        raise ValueError(f"Unsupported data type: {data_type}. Choose from Cifar10, Cifar100.")

    supcon_model = SupConResNetFactory(base_model_func, feature_dim=128).to(device)  # 投影头的特征维度是 128

    # 动态确定 Backbone 的输出特征维度
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32).to(device)  # CIFAR-10 的输入尺寸
        feature_dim = supcon_model.encoder(dummy_input).size(1)  # 确定 Backbone 的输出特征维度
    print(f"Feature dimension dynamically determined: {feature_dim}")


    model = LinearClassifier(supcon_model, feature_dim, num_classes).to(device)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
    model.fc.load_state_dict(checkpoint["classifier_state_dict"])
    print("Model loaded successfully.")

    # 加载测试集
    testloader = load_test_data(data_type=data_type)

    # 测试逻辑
    model.eval()
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

# 主函数
def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Test a trained model on CIFAR-10")
    parser.add_argument("--modir", type=str, required=True, help="Path to the saved model file")
    parser.add_argument("--model", type=str, default="ResNet34",help="Model type: ResNet34, ResNet50, or ResNet101")
    parser.add_argument("--data", type=str, default="Cifar10", help="Data type: Cifar10 or Cifar100")
    args = parser.parse_args()

    model_path = args.modir
    model_type = args.model
    data_type = args.data

    # model_path = "./saved_models/classification/pretrained/ResNet34_cifar10_batch256_valAcc65.51_20250102-230510.pth"  # 修改为你的权重文件路径

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model(model_path, device,model_type,data_type)

if __name__ == "__main__":
    main()


    # 指令示例
    #python test_pretrained_classifier_top1_5.py --model ResNet34 --data Cifar10 --modir ./saved_models/classification/pretrained/ResNet34_cifar10_batch256_valAcc65.51_20250102-230510.pth
