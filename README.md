
## 1 代码结构
- Contrastive_Learning/：包含与对比学习相关的代码。
   - init.py：该文件是简单的初始化文件，将 Contrastive_Learning 文件夹标记为一个 Python 包。通过该文件，项目中其他模块可以导入 Contrastive_Learning 文件夹中的函数、类或配置。
   
   - config_con.py：该文件是监督式对比学习的配置文件，包含对比学习训练过程中的参数设置。
   
   - train_con.py：该文件实现了监督式对比学习的训练过程。

- data_augmentation/：包含数据增强的相关代码，用于在训练过程中对输入数据进行预处理。
   - CutMix.py：实现CutMix数据增强。
   
   - MixUp.py：实现MixUp数据增强。
   
   - data_augmentation_con.py：实现监督对比学习的数据增强逻辑。
 
   - init.py：该文件是简单的初始化文件

- losses/：定义了损失函数相关代码，supin和supout。
   - SupIn.py：实现SupIn损失函数。
   
   - SupOut.py：实现SupOut损失函数。
   
   - CrossEntropy.py：实现CrossEntropy损失函数。

- models/：存放用于测试的模型代码，包括不同架构的神经网络模型定义，以及MLP的实现。
   - ResNet34.py：实现ResNet34网络结构。
   
   - ResNet50.py：实现ResNet50网络结构。
     
   - 其他神经网络架构
   
   - SupConResNet.py：实现MLP多层感知器。
   - init.py：该文件是简单的初始化文件

- my_logs/：用于存储训练过程中的日志信息，便于跟踪和分析模型的训练情况。

- saved_models/：用于保存训练后的模型，便于后续加载和评估。
    - classification/：该目录保存了用于分类任务的模型权重
      - pretrain/：该目录保存了，经过对比学习预训练的分类任务模型
      - scratch/：该目录保存了，从头开始，未经过预训练的分类任务模型
    - pretraining/：该目录保存了经过对比监督学习的预训练权重
      - ResNet34/：该目录保存了，使用 ResNet34 进行监督式对比学习训练后的权重
      - ResNet101/：该目录保存了，使用 ResNet101 进行监督式对比学习训练后的权重
      - 以及更多的主干网络进行监督式对比学习训练后的权重
  
- environment.yml：列出了项目所需的Python库及其版本，便于环境的搭建和依赖管理。

- main_con.py：对比学习的主程序入口，负责解析config_con.py命令行参数，并调用相应的训练和测试函数。

- test_pretrained_classifier.py：用于测试经过预训练的分类器性能的代码，评估模型在测试集上的表现。

- test_scratch_classifier.py：用于测试直接训练的分类器性能的代码，评估模型在测试集上的表现。
  
- train_pretrained_classifier.py：用于训练预训练分类器的代码，加载预训练模型并进行微调。

- train_scratch_classifier.py：用于从头开始训练分类器的代码，初始化模型并进行训练，用于和训练预训练分类器的性能做对比。

- utils.py：包含辅助函数，如数据加载、模型保存和日志记录等功能（用不上）。


## 2 参数解释以及示例运行指令

### 2.1 监督式对比学习的训练

#### 2.1.1 参数说明  

- `--batch_size`：设置批量大小（默认值：32）。示例：`--batch_size 64`。  
- `--learning_rate`：设置学习率（默认值：0.001）。示例：`--learning_rate 0.01`。  
- `--epochs`：设置训练的 epoch 数（默认值：15）。示例：`--epochs 25`。  
- `--temp`：设置对比损失的温度参数（默认值：0.07）。示例：`--temp 0.1`。  
- `--log_dir`：设置训练日志的保存目录（默认值：`./logs`）。示例：`--log_dir ./my_logs`。  
- `--model_save_dir`：设置模型检查点的保存目录（默认值：`./checkpoints`）。示例：`--model_save_dir ./my_checkpoints`。  
- `--gpu`：指定使用的 GPU 设备 ID（默认值：0）。示例：`--gpu 0`。  
- `--dataset`：设置数据集路径（默认值：`./data`）。示例：`--dataset ./datasets`。  
- `--loss_type`：选择损失函数类型（默认值：`cross_entropy`）。示例：`--loss_type supcon`。  
- `--dataset_name`：设置数据集名称，支持 `cifar10`、`cifar100` 等（默认值：`cifar10`）。示例：`--dataset_name cifar10`。  
- `--model_type`：设置模型结构，支持 `resnet34`、`ResNeXt101` 等（默认值：`resnet34`）。示例：`--model_type ResNeXt101`。  
- `--input_resolution`：设置输入图像的分辨率（默认值：32）。示例：`--input_resolution 32`，如果还了更大的数据集，如imagenet，需要调整为更大的值。  
- `--feature_dim`：设置投影头特征尺寸（默认值：128）。示例：`--feature_dim 256`。  
- `--num_workers`：设置数据加载的线程数（默认值：2）。示例：`--num_workers 2`。  

#### 2.1.2 示例运行指令  
完整的示例运行指令，参数可以按需求修改：
```bash
python main_con.py --batch_size 32 --learning_rate 0.01 --epochs 2 --temp 0.1 --log_dir ./my_logs --model_save_dir ./saved_models/pretraining --gpu 0 --dataset ./data --dataset_name cifar10 --model_type ResNet34 --loss_type supout --input_resolution 32 --feature_dim 128 --num_workers 2
```


### 2.2 使用对比学习预训练的权重，进行分类训练

#### 2.2.1 参数说明 

- `--model_type`：设置模型类型，支持 `ResNet34`、`ResNet50`、`ResNet101`、`ResNet200`（默认值：`ResNet50`）。  
- `--batch_size`：设置批量大小（默认值：64）。  
- `--epochs`：设置训练的 epoch 数（默认值：10）。  
- `--learning_rate`：设置学习率（默认值：0.001）。  
- `--dataset_name`：设置数据集名称，支持 `cifar10`、`cifar100`、`imagenet`（默认值：`cifar10`）。  
- `--dataset`：设置数据集路径（默认值：`./data`）。  
- `--pretrained_model`：设置预训练模型路径，需与 `--use_pretrained` 一起使用（默认值：`None`）。  
- `--save_dir`：设置模型保存目录（默认值：`./saved_models/classification/pretrain`）。  
- `--gpu`：指定使用的 GPU 设备 ID（默认值：0）。

#### 2.2.2 示例运行指令 

```bash
python train_pretrained_classifier.py --model_type ResNet34 --pretrained_model ./saved_models/pretraining/ResNet34/ResNet34_cifar10_feat128_supout_epoch241_batch32.pth --save_dir ./saved_models/classification/pretrained --batch_size 32 --epochs 3 --learning_rate 0.001 --dataset_name cifar10 --dataset ./data --gpu 0
```

### 2.3 从头开始的分类器训练
#### 2.3.1 参数说明 
- `--model_type`：设置模型类型，支持 `ResNet50`、`ResNet34`、`ResNet101`、`ResNet200`（默认值：`ResNet50`）。  
- `--batch_size`：设置批量大小（默认值：64）。  
- `--epochs`：设置训练的 epoch 数（默认值：10）。  
- `--learning_rate`：设置学习率（默认值：0.1）。  
- `--dataset_name`：设置数据集名称，支持 `cifar10`、`cifar100`、`imagenet`（默认值：`cifar10`）。  
- `--dataset`：设置数据集路径（默认值：`./data`）。  
- `--save_dir`：设置保存最佳模型的目录（默认值：`./saved_models/classification/scratch`）。  
- `--gpu`：指定使用的 GPU 设备 ID（默认值：0）。

#### 2.3.2 示例运行指令
```bash
python train_scratch_classifier.py --model_type ResNet34 --batch_size 32 --epochs 3 --learning_rate 0.1 --dataset_name cifar10 --dataset ./data --save_dir ./saved_models/classification/scratch --gpu 0
```
## 3 训练过程和逻辑

在该项目中，监督式对比学习实际上是一种预训练思路，它提前为数据集进行了聚类。

所以在训练过程中，需要先运行main_con，进行预训练。再运行train_pretrained_classifier，对预训练的权重进行线性分类训练。

注意：
- 两次训练必须使用同样的主干网络（如ResNet34），以及同样的数据集（如cifar10）！！！
- train_pretrained_classifier时需要注意调用对应的预训练权重。
  
此外，train_scratch_classifier.py 是直接对数据集进行传统的分类训练，未经过预训练。用于和经过对比学习的结果进行性能对比。

在训练中，会自动保存性能最佳的模型。

服务器挂起的相关指令：

tmux new -s

tmux ls

tmux attach -t 

tmux kill-session -t 

tmux kill-server

  

