import argparse
import os

def parse_option():
    parser = argparse.ArgumentParser('Supervised Contrastive Learning with Config and CLI')
    # Batch size
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    # Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    # Number of training epochs
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    # Temperature parameter, used for scaling in contrastive loss
    parser.add_argument('--temp', type=float, default=0.07, help='Temperature for contrastive loss')
    # # Checkpoint save frequency, save the model every certain number of epochs
    # parser.add_argument('--save_freq', type=int, default=5, help='Save frequency for checkpoints')

    # Log directory, used to save training logs (e.g., loss, accuracy per epoch)
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save training logs')
    # # Model save directory, used to save trained model checkpoints
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')

    # GPU ID, specifies the GPU device ID to use; if None, CPU will be used
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    # Dataset path, specifies the storage path of the dataset
    parser.add_argument('--dataset', type=str, default='./data', help='Dataset to use')
    # Type of loss function, selects the loss function to use
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                        help='Loss type (e.g., cross_entropy, supcon, supin)')
    # Dataset name, specifies the dataset to use (e.g., cifar10, cifar100, imagenet)
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        help='Dataset name (e.g., cifar10, cifar100, imagenet)')
    # Model type, specifies the model architecture to use (e.g., resnet34, ResNeXt101, WideResNet)
    parser.add_argument('--model_type', type=str, default='resnet34',
                        help='Model type (e.g., resnet34, ResNeXt101, WideResNet)')
    # # Data augmentation method, selects the data augmentation mode
    # parser.add_argument('--augmentation', type=str, default='basic',
    #                     help='Data augmentation method (e.g., basic, advanced)')
    # Input resolution
    parser.add_argument('--input_resolution', type=int, default=32, help='Input image resolution')
    # Feature dimension for the projection head
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension for the projection head')

    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading (default: 2)")

    # The class_name parameter is typically used to specify a specific class label in the dataset, especially when performing operations on a specific category in multi-class tasks.
    # parser.add_argument('--class_name', type=str, default='default_class', help='Class name for dataset')
    # The action_type parameter can specify the execution mode of the code (e.g., train, test, inference), allowing for different functionalities within one script.
    # parser.add_argument('--action_type', type=str, default='norm-train', help='Action type (e.g., norm-train)')

    args = parser.parse_args()

    # Dynamically set the number of classes
    if args.dataset_name == 'cifar10':
        args.num_classes = 10
    elif args.dataset_name == 'cifar100':
        args.num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    # Directly convert the argparse results into a dictionary
    opt = vars(args)

    return opt


def get_config():
    return parse_option()

if __name__ == "__main__":
    opt = get_config()
    print(opt)
