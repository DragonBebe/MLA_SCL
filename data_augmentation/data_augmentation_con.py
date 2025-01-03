from torchvision import transforms


class TwoCropTransform:
    """Create two crops of the same image for contrastive learning."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


# 定义标准化参数的映射
DATASET_STATS = {
    'cifar10': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    },
    'cifar100': {
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761)
    },
    'imagenet': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225)
    }
}


# def get_base_transform(input_resolution=32):
#     """
#     定义基本的数据增强组合，支持动态输入分辨率。
#     Args:
#         input_resolution (int): 输入图像的分辨率（默认32）。
#     """
#     return transforms.Compose([
#         # transforms.RandomResizedCrop(input_resolution, scale=(0.2, 1.0)),  # 动态分辨率
#         transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
#         # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # 颜色抖动
#         transforms.RandomGrayscale(p=0.2),           # 随机灰度
#         # transforms.RandomRotation(10),  # 随机旋转
#         # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),  # 随机高斯模糊
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])


def get_base_transform(dataset_name, input_resolution=32):
    """
    根据数据集动态设置标准化参数和其他数据增强方法。
    Args:
        dataset_name (str): 数据集名称 (e.g., 'cifar10', 'cifar100', 'imagenet')。
        input_resolution (int): 输入图像分辨率。
    """
    stats = DATASET_STATS.get(dataset_name, DATASET_STATS['cifar10'])  # 默认使用 CIFAR-10 参数
    return transforms.Compose([
        # transforms.RandomResizedCrop(input_resolution, scale=(0.2, 1.0)),  # 动态分辨率
        transforms.RandomResizedCrop(size=input_resolution, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
        # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # 颜色抖动
        transforms.RandomGrayscale(p=0.2),  # 随机灰度
        # transforms.RandomRotation(10),  # 随机旋转
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),  # 随机高斯模糊
        transforms.ToTensor(),
        transforms.Normalize(mean=stats['mean'], std=stats['std'])  # 动态标准化参数
    ])
