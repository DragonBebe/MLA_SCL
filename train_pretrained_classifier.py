# 问题：学习率调整逻辑，是否还能优化

import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch_optimizer import Lookahead
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from models import ResNet34, ResNet50, ResNet101, ResNet200, SupConResNetFactory
from data_augmentation import cutmix_data, cutmix_criterion, mixup_data, mixup_criterion
import os
from tqdm import tqdm  # 用于显示进度条
import datetime

from torch.utils.tensorboard import SummaryWriter # 用于加载tensorboard
import subprocess  # 用于调用外部脚本 可以每x个eporch调用一次test
import re

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def save_best_model(backbone, classifier, save_path, last_save_path):
    if save_path == last_save_path:
        print(f"Saving new model to {save_path}, skipping deletion of identical path.")
    else:
        if last_save_path and os.path.exists(last_save_path):
            os.remove(last_save_path)
            print(f"Deleted previous model: {last_save_path}")

    # 保存 Backbone 和分类头的权重
    torch.save({
        "backbone_state_dict": backbone.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
    }, save_path)
    print(f"New best model saved to {save_path}")
    return save_path


def train_classifier(train_loader, val_loader, model, classifier, optimizer, scheduler, criterion, device, epochs=10,
                     save_dir="./saved_models", model_type="ResNet50", batch_size=64, use_pretrained=True,
                     dataset_name="cifar10",test_script_path="test_pretrained_classifier_top1_5.py"):
    if use_pretrained:
        model.eval()
    else:
        model.train()
    classifier.train()

    best_accuracy = 0.0
    last_save_path = None
    ensure_dir_exists(save_dir)

    # 初始化 TensorBoard
    log_dir = os.path.join(save_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir=log_dir)

    try:
        for epoch in range(epochs):
            print(f"Epoch [{epoch + 1}/{epochs}]")

            running_loss = 0.0
            correct = 0
            total = 0
            batch_losses = []
            batch_accuracies = []

            model.train()
            train_bar = tqdm(train_loader, desc="Training", leave=False)
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                # 随机选择 CutMix 或 Mixup
                if np.random.rand() < 0.00:  # 50% 概率使用 CutMix # 0.2
                    inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, alpha=1.0)
                    outputs = classifier(model.encoder(inputs))
                    loss = cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)
                elif np.random.rand() < 0.0: # 0.4
                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)
                    outputs = classifier(model.encoder(inputs))
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    outputs = classifier(model.encoder(inputs))
                    loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                running_loss += loss.item()
                batch_losses.append(loss.item())
                batch_accuracies.append((predicted == labels).float().mean().item())

                train_bar.set_postfix(loss=loss.item(), acc=batch_accuracies[-1] * 100)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct / total
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy * 100:.2f}%")
            print(f"  Batch Loss: min={min(batch_losses):.4f}, max={max(batch_losses):.4f}, mean={epoch_loss:.4f}")
            print(
                f"  Batch Accuracy: min={min(batch_accuracies) * 100:.2f}%, max={max(batch_accuracies) * 100:.2f}%, mean={epoch_accuracy * 100:.2f}%")


            # 记录训练损失和准确率到 TensorBoard
            writer.add_scalar("Train/Loss", epoch_loss, epoch)
            writer.add_scalar("Train/Accuracy", epoch_accuracy * 100, epoch)

            classifier.eval()
            val_correct = 0
            val_total = 0
            val_running_loss = 0.0
            val_bar = tqdm(val_loader, desc="Validating", leave=False)
            with torch.no_grad():
                for val_inputs, val_labels in val_bar:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_features = model.encoder(val_inputs)
                    val_outputs = classifier(val_features)
                    val_loss = criterion(val_outputs, val_labels)

                    _, val_predicted = val_outputs.max(1)
                    val_correct += (val_predicted == val_labels).sum().item()
                    val_total += val_labels.size(0)
                    val_running_loss += val_loss.item()

            val_loss = val_running_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")


            # 记录验证损失和准确率到 TensorBoard
            writer.add_scalar("Validation/Loss", val_loss, epoch)
            writer.add_scalar("Validation/Accuracy", val_accuracy * 100, epoch)

            # 这里用于保存当前的test

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(save_dir,
                                         f"{model_type}_{dataset_name}_batch{batch_size}_valAcc{val_accuracy * 100:.2f}_{timestamp}.pth")
                last_save_path = save_best_model(model, classifier, save_path, last_save_path)

            scheduler.step()
            """
            # 每 3 个 epoch 调用测试脚本
            if (epoch + 1) % 2 == 0:
                print("\nCalling test script...")
                try:
                    subprocess.run(["python", test_script_path, "--modir", last_save_path, "--model", model_type],
                                   check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while running the test script: {e}")
            """

            # 每 3 个 epoch 调用测试脚本
            if (epoch + 1) % 1 == 0:
                print("\nCalling test script...")
                try:
                    result = subprocess.run(
                        ["python", test_script_path, "--modir", last_save_path, "--model", model_type],
                        check=True, capture_output=True, text=True
                    )
                    # 从测试脚本输出中提取 Top-1 和 Top-5 准确率
                    output = result.stdout
                    top1_match = re.search(r"Top-1 Accuracy: (\d+\.\d+)%", output)
                    top5_match = re.search(r"Top-5 Accuracy: (\d+\.\d+)%", output)

                    if top1_match and top5_match:
                        top1_accuracy = float(top1_match.group(1))
                        top5_accuracy = float(top5_match.group(1))

                        # 记录到 TensorBoard
                        writer.add_scalar("Test/Top-1 Accuracy", top1_accuracy, epoch)
                        writer.add_scalar("Test/Top-5 Accuracy", top5_accuracy, epoch)
                        print(f"Test results added to TensorBoard: Top-1: {top1_accuracy}%, Top-5: {top5_accuracy}%")
                    else:
                        print("Failed to extract accuracies from test script output.")
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while running the test script: {e}")
        print(f"Training complete. Best model saved with validation accuracy: {best_accuracy * 100:.2f}%")
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train a classification head on top of a frozen feature extractor")
    parser.add_argument("--model_type", type=str, default="ResNet50",
                        help="Model type (ResNet50, ResNet34, ResNet101, ResNet200)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dataset_name", type=str, default="cifar10",
                        help="Dataset name (cifar10, cifar100, imagenet)")
    parser.add_argument("--dataset", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to pre-trained SupConResNet (required if --use_pretrained)")
    parser.add_argument("--save_dir", type=str, default="./saved_models/classification/pretrained",
                        help="Directory to save the best classifier")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use (default: 0)")
    # parser.add_argument("--use_pretrained", action="store_true", help="Use a pre-trained SupConResNet (default: False)")
    # parser.add_argument("--no_pretrained", dest="use_pretrained", action="store_false", help="Do not use pre-trained weights")
    parser.set_defaults(use_pretrained=True)

    args = parser.parse_args()

    if args.use_pretrained and not args.pretrained_model:
        parser.error("--pretrained_model is required when --use_pretrained is True.")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.dataset_name == "cifar10":

        transform = transforms.Compose([
            # transforms.RandomResizedCrop(32),
            # transforms.RandomHorizontalFlip(),

            ###########################################################################
            AutoAugment(AutoAugmentPolicy.CIFAR10),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
            transforms.RandomGrayscale(p=0.1),  # 随机灰度
            transforms.RandomRotation(10),  # 随机旋转
            ###########################################################################

            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.05),  # 随机高斯模糊
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2032, 0.1994, 0.2010)),
        ])

        dataset = datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=transform)
        num_classes = 10
    elif args.dataset_name == "cifar100":
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(32),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR100(root=args.dataset, train=True, download=True, transform=transform)
        num_classes = 100
    elif args.dataset_name == "imagenet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(root=os.path.join(args.dataset, "train"), transform=transform)
        num_classes = 1000
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    # 关闭tensorflow的oneDNN 优化，从而减少可能的冲突
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model_dict = {
        "ResNet34": lambda: ResNet34(),
        "ResNet50": lambda: ResNet50(),
        "ResNet101": lambda: ResNet101(),
        "ResNet200": lambda: ResNet200(),
    }
    base_model_func = model_dict.get(args.model_type)
    if base_model_func is None:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model = SupConResNetFactory(base_model_func, feature_dim=128)

    if args.use_pretrained:
        checkpoint = torch.load(args.pretrained_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded pre-trained model weights.")
    else:
        print("Using randomly initialized ResNet.")
    model = model.to(device)

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32 if args.dataset_name != "imagenet" else 224,
                                  32 if args.dataset_name != "imagenet" else 224).to(device)
        feature_dim = model.encoder(dummy_input).size(1)

    classifier = nn.Linear(feature_dim, num_classes).to(device)

    base_optimizer = optim.SGD(
        list(classifier.parameters()) + (list(model.parameters()) if not args.use_pretrained else []),
        lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    #     # 使用 AdamW 作为基础优化器
    #     base_optimizer = optim.AdamW(
    #         model.parameters(),
    #         lr=args.learning_rate,  # 学习率，与 SGD 的默认值可能不同，建议适当减小
    #         betas=(0.9, 0.999),  # 默认 AdamW 参数
    #         eps=1e-8,  # 防止数值不稳定
    #         weight_decay=1e-3  # 权重衰减
    #     )

    #     # 包装 Lookahead 优化器
    #     optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer,
    #         T_max=args.epochs  # Cosine退火周期与总训练 epoch 对应
    #     )

    criterion = nn.CrossEntropyLoss()

    print("Training started...")
    train_classifier(train_loader, val_loader, model, classifier, optimizer, scheduler, criterion,
                     device, epochs=args.epochs, save_dir=args.save_dir, model_type=args.model_type,
                     batch_size=args.batch_size, use_pretrained=args.use_pretrained, dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()

# python train_pretrained_classifier.py --model_type ResNet34 --batch_size 128 --epochs 10 --learning_rate 0.005 --dataset_name cifar10  --pretrained_model ./saved_models/pretraining/ResNet34/ResNet34_cifar10_feat128_batch256_epoch696_loss4.7631_20241217-143332.pth


# python train_pretrained_classifier.py --model_type ResNet34 --pretrained_model ./saved_models/pretraining/ResNet34/ResNet34_cifar10_feat128_supout_epoch241_batch32.pth --batch_size 32 --epochs 20 --learning_rate 0.001 --dataset_name cifar10

# python main.py --no_pretrained --model_type ResNet34 --dataset_name cifar10 --batch_size 32 --epochs 20 --learning_rate 0.001 --save_dir ./saved_models/classification/non_pretrained
