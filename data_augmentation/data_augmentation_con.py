from torchvision import transforms


class TwoCropTransform:
    """Create two crops of the same image for contrastive learning."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


# Define normalization parameters for different datasets
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
#     Define basic data augmentation pipeline, supporting dynamic input resolution.
#     Args:
#         input_resolution (int): Input image resolution (default 32).
#     """
#     return transforms.Compose([
#         # transforms.RandomResizedCrop(input_resolution, scale=(0.2, 1.0)),  # Dynamic resolution
#         transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
#         # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Color jittering
#         transforms.RandomGrayscale(p=0.2),           # Random grayscale
#         # transforms.RandomRotation(10),  # Random rotation
#         # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),  # Random Gaussian blur
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])


def get_base_transform(dataset_name, input_resolution=32):
    """
    Dynamically set normalization parameters and other data augmentation methods based on the dataset.
    Args:
        dataset_name (str): Dataset name (e.g., 'cifar10', 'cifar100', 'imagenet').
        input_resolution (int): Input image resolution.
    """
    stats = DATASET_STATS.get(dataset_name, DATASET_STATS['cifar10'])  # Default to CIFAR-10 parameters
    return transforms.Compose([
        # transforms.RandomResizedCrop(input_resolution, scale=(0.2, 1.0)),  # Dynamic resolution
        transforms.RandomResizedCrop(size=input_resolution, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
        # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Color jittering
        transforms.RandomGrayscale(p=0.2),  # Random grayscale
        # transforms.RandomRotation(10),  # Random rotation
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),  # Random Gaussian blur
        transforms.ToTensor(),
        transforms.Normalize(mean=stats['mean'], std=stats['std'])  # Dynamic normalization parameters
    ])
