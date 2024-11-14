import os
import torch
import config  # 导入配置文件
from train import train, set_loader, set_model
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def main():
    # 从配置文件获取配置
    opt = config.get_config()

    # 创建日志和模型保存目录
    if not os.path.isdir(opt['log_dir']):
        os.makedirs(opt['log_dir'])
    if not os.path.isdir(opt['model_save_dir']):
        os.makedirs(opt['model_save_dir'])

    # 设置日志记录器
    writer = SummaryWriter(log_dir=opt['log_dir'])

    # 加载数据和模型
    train_loader = set_loader(opt)
    model, criterion, device = set_model(opt)  # 返回 device

    # 设置优化器
    optimizer = optim.SGD(model.parameters(), lr=opt['learning_rate'], momentum=0.9, weight_decay=1e-4)

    # 开始训练
    epoch_loss = train(train_loader, model, criterion, optimizer, opt, writer, device)

    # 保存模型
    save_file = os.path.join(opt['model_save_dir'], 'last.pth')
    torch.save(model.state_dict(), save_file)
    print(f'Final model saved to {save_file}')
    writer.close()


if __name__ == '__main__':
    main()
