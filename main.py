import os
import torch
import config  # 配置模块
from train import train, set_loader, set_model

def main():
    # 从配置文件获取配置
    opt = config.get_config()

    # 创建日志目录和模型保存目录
    log_dir = opt['log_dir']
    model_save_dir = opt['model_save_dir']

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # 创建数据加载器和模型
    train_loader, test_loader = set_loader(opt)
    model, criterion, device = set_model(opt)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt['learning_rate'],
        momentum=0.9,
        weight_decay=1e-4,
    )

    # 训练循环
    for epoch in range(opt['epochs']):
        print(f"Epoch [{epoch + 1}/{opt['epochs']}]")
        epoch_loss, epoch_accuracy = train(train_loader, model, criterion, optimizer, opt, device)
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")

        # 保存检查点
        if (epoch + 1) % opt['save_freq'] == 0 or (epoch + 1) == opt['epochs']:
            checkpoint_path = os.path.join(opt['model_save_dir'], f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training complete.")

if __name__ == "__main__":
    main()
