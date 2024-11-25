import os
import torch
import config  # 假设你已有的配置模块
from train import train, set_loader, set_model

def main():
    # 从配置文件获取配置
    opt = config.get_config()

    # 确保日志目录和模型保存目录
    log_dir = opt['log_dir']
    model_save_dir = opt['model_save_dir']

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    elif not os.path.isdir(log_dir):
        raise ValueError(f"路径 '{log_dir}' 已存在，但不是目录。请删除或修改路径配置。")

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    elif not os.path.isdir(model_save_dir):
        raise ValueError(f"路径 '{model_save_dir}' 已存在，但不是目录。请删除或修改路径配置。")

    print(f"Log directory: {log_dir}")
    print(f"Model save directory: {model_save_dir}")

    # 加载数据和模型
    train_loader = set_loader(opt)
    model, criterion, device = set_model(opt)

    # 设置优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt['learning_rate'],
        momentum=0.9,
        weight_decay=1e-4,
    )

    # 开始训练并保存日志
    for epoch in range(opt['epochs']):
        epoch_loss, epoch_accuracy = train(train_loader, model, criterion, optimizer, opt, device)

        # 保存每个 epoch 的日志
        log_data = {
            "epoch": epoch,
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "learning_rate": optimizer.param_groups[0]['lr'],
        }
        torch.save(log_data, os.path.join(log_dir, f"log_epoch_{epoch}.pth"))
        print(f"Epoch {epoch}: Loss={epoch_loss}, Accuracy={epoch_accuracy}")

        # 保存模型检查点
        if (epoch + 1) % opt['save_freq'] == 0 or (epoch + 1) == opt['epochs']:
            save_file = os.path.join(model_save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_file)
            print(f"Checkpoint saved to {save_file}")

    print("Training complete.")

if __name__ == "__main__":
    main()




