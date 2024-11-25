# MLA_SCL
This project is part of an Advanced Machine Learning (MLA), focused on reproducing the paper Supervised Contrastive Learning.



## 参数解释

以下是项目中各个命令行参数的解释及其用途，您可以根据项目需求灵活配置这些参数：

* `--batch_size`：设置批量大小（默认值：32）。示例： `--batch_size 8` ，表示每次迭代使用8个样本进行训练。
* `--learning_rate`：设置学习率（默认值：0.001），控制优化器的步长。较小的学习率会使模型收敛更慢但可能获得更好的结果。示例：`--learning_rate 0.01`。
* `--epochs`：设置训练的 epoch 数（默认值：15），即数据集的完整迭代次数。示例：`--epochs 25` 表示训练25个完整的 epoch。
* `--temp`：设置对比损失中的温度参数（默认值：0.07），用于缩放对比损失。示例：`--temp 0.1`。
* `--save_freq`：设置模型检查点的保存频率（默认值：5），即每隔多少个 epoch 保存一次模型。示例：`--save_freq 3` 表示每3个 epoch 保存一次。
* `--log_dir`：设置 TensorBoard 日志保存目录（默认值：`./logs`）。所有的训练日志会保存在该目录下。示例：`--log_dir ./my_logs`。
* `--model_save_dir`：设置模型检查点保存目录（默认值：`./checkpoints`）。训练过程中每个保存的模型文件会保存在该目录下。示例：`--model_save_dir ./my_checkpoints`。
* `--gpu`：指定使用的 GPU 设备 ID（默认值：0）。设置为 `None` 时则使用 CPU。示例：`--gpu 0` 表示使用第一个 GPU。
* `--dataset`：指定数据集的存储路径（默认值：`./data`）。示例：`--dataset ./data`。
* `--dataset_name`：设置数据集名称，支持 `cifar10`、`cifar100`、`imagenet`（默认值：`cifar10`）。示例：`--dataset_name cifar10`。
* `--model_type`：选择用于训练的模型类型（默认值：`resnet34`），支持 `resnet34`、`ResNeXt101`、`WideResNet` 等。示例：`--model_type ResNeXt101`。
* `--loss_type`：选择损失函数类型（默认值：`supcon`），可选项包括 `cross_entropy`、`supcon`、`supin`。示例：`--loss_type supcon` 表示使用 `SupConLoss_out` 损失函数。
* `--augmentation`：设置数据增强方式（默认值：`basic`），支持 `basic` 和 `advanced`。示例：`--augmentation advanced`。

### 示例运行指令

```bash
python main.py --batch_size 8 --learning_rate 0.001 --epochs 5 --temp 0.1 --save_freq 3 --log_dir ./my_logs --model_save_dir ./my_checkpoints --gpu 0 --dataset ./data --dataset_name cifar10 --model_type ResNet34 --loss_type supcon --augmentation basic
```

以上命令将在 CIFAR-10 数据集上以 resnet34 模型进行训练，使用 SupConLoss_out 损失函数，训练 5 个 epoch，批量大小为 8，学习率为 0.001，且每隔 3 个 epoch 保存一次模型。

根据项目需求，可以随时调整命令行中的参数配置以实现灵活的实验设置。

