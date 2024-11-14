import argparse


def parse_option():
    parser = argparse.ArgumentParser('Supervised Contrastive Learning with Config and CLI')
    # 批量大小
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    # 学习率
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    # 训练的epoch数
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    # 温度参数，用于对比损失中的温度缩放
    parser.add_argument('--temp', type=float, default=0.07, help='Temperature for contrastive loss')
    # 检查点保存频率，每隔多少个epoch保存一次模型
    parser.add_argument('--save_freq', type=int, default=5, help='Save frequency for checkpoints')
    # 日志保存目录，用于保存TensorBoard日志
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    # 模型保存目录，用于保存训练的模型检查点
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints', help='Directory to save models')
    # GPU ID，指定使用的GPU设备ID；如果为None则使用CPU
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    # 数据集路径，指定数据集的存储路径
    parser.add_argument('--dataset', type=str, default='./data', help='Dataset to use')
    # 损失函数类型，选择使用的损失函数
    parser.add_argument('--loss_type', type=str, default='supcon',
                        help='Loss type (e.g., cross_entropy, supcon, supin)')
    # 数据集名称，用于指定选择的数据集（cifar10、cifar100、imagenet等）
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        help='Dataset name (e.g., cifar10, cifar100, imagenet)')
    # 模型类型，指定要使用的模型结构（resnet34、ResNeXt101、WideResNet等）
    parser.add_argument('--model_type', type=str, default='resnet34',
                        help='Model type (e.g., resnet34, ResNeXt101, WideResNet)')
    # 数据增强方法，选择数据增强模式
    parser.add_argument('--augmentation', type=str, default='basic',
                        help='Data augmentation method (e.g., basic, advanced)')

    # class_name参数通常用于指定数据集中具体的类标签，尤其是在多分类任务中需要对特定类别进行操作时会用到。
    # parser.add_argument('--class_name', type=str, default='default_class', help='Class name for dataset')
    # action_type参数可以用来指定代码的执行模式（如train、test、inference等），这样可以在一个脚本中实现不同的功能
    # parser.add_argument('--action_type', type=str, default='norm-train', help='Action type (e.g., norm-train)')

    args = parser.parse_args()

    # 直接将 argparse 的解析结果转换为字典形式
    opt = vars(args)

    return opt


def get_config():
    return parse_option()
