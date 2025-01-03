import argparse
import torch
import torch.nn as nn
from tqdm import tqdm  # 用于显示进度条
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from models import ResNet34, ResNet50, ResNet101, ResNet200
import os
import datetime

from torch.utils.tensorboard import SummaryWriter # 用于加载tensorboard

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def save_best_model(model, save_path, last_save_path):
    if save_path == last_save_path:
        print(f"Saving new model to {save_path}, skipping deletion of identical path.")
    else:
        if last_save_path and os.path.exists(last_save_path):
            os.remove(last_save_path)
            print(f"Deleted previous model: {last_save_path}")

    torch.save(model.state_dict(), save_path)
    print(f"New best model saved to {save_path}")
    return save_path


def train_from_scratch(train_loader, val_loader, model, optimizer, scheduler, criterion, device, dataset_name, epochs=10,
                       save_dir="./saved_models", model_type="ResNet50", batch_size=64):
    model.train()
    best_accuracy = 0.0
    last_save_path = None
    ensure_dir_exists(save_dir)

    # 初始化 TensorBoard
    log_dir = os.path.join(save_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir=log_dir)

    try:
        for epoch in range(epochs):
            print(f"Epoch [{epoch + 1}/{epochs}]")

            # Training loop
            running_loss = 0.0
            correct = 0
            total = 0
            batch_losses = []  # 保存每个 batch 的损失
            batch_accuracies = []  # 保存每个 batch 的准确率

            model.train()
            train_bar = tqdm(train_loader, desc="Training", leave=False)
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                running_loss += loss.item()

                # 记录 batch 的损失和准确率
                batch_losses.append(loss.item())
                batch_accuracies.append((predicted == labels).float().mean().item())

                # 更新进度条
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

            # Validation loop
            model.eval()
            val_correct = 0
            val_total = 0
            val_running_loss = 0.0
            val_bar = tqdm(val_loader, desc="Validating", leave=False)
            with torch.no_grad():
                for val_inputs, val_labels in val_bar:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = model(val_inputs)
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

            # Save the best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(save_dir,
                                         f"{model_type}_{dataset_name}_batch{batch_size}_valAcc{val_accuracy * 100:.2f}_{timestamp}.pth")
                last_save_path = save_best_model(model, save_path, last_save_path)

            # Update learning rate
            scheduler.step()

        print(f"Training complete. Best model saved with validation accuracy: {best_accuracy * 100:.2f}%")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        writer.close()

def main():
    parser = argparse.ArgumentParser(description="Train a ResNet model from scratch for classification")
    parser.add_argument("--model_type", type=str, default="ResNet50", help="Model type (ResNet50, ResNet34, ResNet101, ResNet200)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--dataset_name", type=str, default="cifar10", help="Dataset name (cifar10, cifar100, imagenet)")
    parser.add_argument("--dataset", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--save_dir", type=str, default="./saved_models/classification/scratch", help="Directory to save the best model")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use (default: 0)")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 关闭tensorflow的oneDNN 优化，从而减少可能的冲突
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Dataset loading
    if args.dataset_name == "cifar10":
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(32),
            # transforms.RandomHorizontalFlip(),
            AutoAugment(AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=transform)
        num_classes = 10
    elif args.dataset_name == "cifar100":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
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

    # Split dataset
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model_dict = {
        "ResNet34": lambda: ResNet34(num_classes=num_classes),
        "ResNet50": lambda: ResNet50(num_classes=num_classes),
        "ResNet101": lambda: ResNet101(num_classes=num_classes),
        "ResNet200": lambda: ResNet200(num_classes=num_classes),
    }
    base_model_func = model_dict.get(args.model_type)
    if base_model_func is None:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model = base_model_func().to(device)

    # Optimizer and scheduler
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,  # 学习率，与 SGD 的默认值可能不同，建议适当减小
        betas=(0.9, 0.999),  # 默认 AdamW 参数
        eps=1e-8,  # 防止数值不稳定
        weight_decay=5e-4  # 权重衰减
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs  # Cosine退火周期与总训练 epoch 对应
    )



    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Train from scratch
    print("Training started...")
    train_from_scratch(train_loader, val_loader, model, optimizer, scheduler, criterion, device, dataset_name=args.dataset_name,
                       epochs=args.epochs, save_dir=args.save_dir, model_type=args.model_type, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

# python train_scratch_classifier.py --model_type ResNet34 --batch_size 16 --epochs 20 --learning_rate 0.005 --dataset_name cifar10
