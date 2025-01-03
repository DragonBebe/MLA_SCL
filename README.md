# Supervised Contrastive Learning (SCL)

This repository provides a reproduction and implementation of the **Supervised Contrastive Learning** framework, as introduced in the [original paper](https://arxiv.org/abs/2004.11362). The project focuses on training neural networks with supervised contrastive loss and evaluating their performance on benchmark datasets.

![image](https://github.com/DragonBebe/MLA_SCL/blob/main/image/scl.png)

---

## Features

- Implementation of **Supervised Contrastive Loss** (`supin` and `supout` variations).
- Support for multiple ResNet backbones: **ResNet-34**, **ResNet-50**, **ResNet-101**, and **ResNet-200**.
- Pretraining with supervised contrastive loss for improved feature representation.
- Fine-tuning and training classifiers from scratch for comparative evaluations.
- Data augmentation strategies including **CutMix**, **MixUp**, and **AutoAugment**.
- Configurable training settings to adapt to different tasks and datasets.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/DragonBebe/MLA_SCL.git
cd MLA_SCL
```
## Set Up the Environment

1. Create the conda environment using the provided `environment.yml` file:

    ```bash
    conda env create --file environment.yml
    ```

2. Activate the created environment:

    ```bash
    conda activate SCL
    ```
---
## Code Architecture

The project structure is organized as follows:

```plaintext
Supervised-Contrastive-Learning/
├── Contrastive_Learning/
│   ├── __init__.py                 # Marks the directory as a Python package
│   ├── config_con.py               # Configuration file for supervised contrastive learning
│   ├── train_con.py                # Main training script for contrastive learning
├── data_augmentation/
│   ├── __init__.py                 # Marks the directory as a Python package
│   ├── CutMix.py                   # Implementation of CutMix augmentation
│   ├── MixUp.py                    # Implementation of MixUp augmentation
│   ├── data_augmentation_con.py    # Augmentation pipeline for contrastive learning
├── losses/
│   ├── __init__.py                 # Marks the directory as a Python package
│   ├── SupIn.py                    # Implementation of SupIn loss
│   ├── SupOut.py                   # Implementation of SupOut loss
│   ├── CrossEntropy.py             # Implementation of CrossEntropy loss
├── models/
│   ├── __init__.py                 # Marks the directory as a Python package
│   ├── ResNet34.py                 # Implementation of ResNet-34 backbone
│   ├── ResNet50.py                 # Implementation of ResNet-50 backbone
│   ├── ResNet101.py                # Implementation of ResNet-101 backbone
│   ├── ResNet200.py                # Implementation of ResNet-200 backbone
│   ├── Projectionhead.py           # Implementation of the projection head
├── saved_models/                   # Directory for saving pretrained models and weights
│   ├── classification/             # Contains weights for classification tasks
│   │   ├── pretrain/               # Pretrained classification models
│   │   └── scratch/                # Models trained from scratch
│   ├── pretraining/                # Pretrained weights for contrastive learning
├── my_logs/                        # Stores training logs
├── main_con.py                     # Entry point for contrastive learning pretraining
├── train_pretrained_classifier.py  # Fine-tuning pretrained models
├── train_scratch_classifier.py     # Training classifiers from scratch
├── test_pretrained_classifier.py   # Evaluating pretrained models
├── test_scratch_classifier.py      # Evaluating classifiers trained from scratch
└── environment.yml                 # Python dependencies for setting up the environment
```
---
## Training and Evaluation

### Pretraining with Supervised Contrastive Loss

To pretrain the model using supervised contrastive loss, use the following command, parameters can be modified as needed:

```bash
python main_con.py --batch_size 32 --learning_rate 0.5 --epochs 700 --temp 0.1 --log_dir ./my_logs --model_save_dir ./saved_models/pretraining --gpu 0 --dataset ./data --dataset_name cifar10 --model_type ResNet34 --loss_type supout --input_resolution 32 --feature_dim 128 --num_workers 2
```
### Fine-tuning Pretrained Models

To fine-tune the pretrained model for classification, run the following command, parameters can be modified as needed:
```bash
python train_pretrained_classifier.py --model_type ResNet34 --pretrained_model ./saved_models/pretraining/ResNet34/ResNet34_cifar10_feat128_supout_epoch241_batch32.pth --save_dir ./saved_models/classification/pretrained --batch_size 32 --epochs 3 --learning_rate 0.001 --dataset_name cifar10 --dataset ./data --gpu 0
```
### Training Classifiers from Scratch
To train a classifier from scratch without pretraining, use the following command, parameters can be modified as needed:
```bash
python train_scratch_classifier.py --model_type ResNet34 --batch_size 32 --epochs 3 --learning_rate 0.1 --dataset_name cifar10 --dataset ./data --save_dir ./saved_models/classification/scratch --gpu 0
```
---
## Training Workflow

In this project, **Supervised Contrastive Learning** is implemented as a pretraining strategy that effectively clusters data representations before classification. The training process is divided into three distinct phases:

### 1. Pretraining with Supervised Contrastive Loss

The first step is to pretrain the model using supervised contrastive loss. This step clusters the feature representations, preparing them for downstream classification tasks. Use the `main_con.py` script to perform this pretraining step. The pretrained weights will be saved automatically.

### 2. Linear Classification Training

After pretraining, the next step is to fine-tune the pretrained weights for linear classification. Use the `train_pretrained_classifier.py` script to load the pretrained weights and perform the classification task. 

**Important Notes:**
- Both training steps must use the same backbone network (e.g., ResNet-34) and dataset (e.g., CIFAR-10) for consistency.
- Ensure that the correct pretrained weights are loaded during the fine-tuning step.

### 3. Training Classifiers from Scratch

For comparison, the `train_scratch_classifier.py` script trains a classifier from scratch on the dataset without any pretraining. This serves as a baseline to evaluate the performance improvement introduced by the supervised contrastive learning strategy.

### Model Saving

During training, the scripts automatically save the model weights with the best performance (e.g., highest accuracy). These saved weights can be used for further evaluations or deployment.

By structuring the training process this way, the project ensures:
1. Efficient feature extraction through pretraining.
2. Robust evaluation of the performance benefits of supervised contrastive learning.
3. Direct comparison between pretrained and non-pretrained approaches.
---
## Results
之后补充

---
## Contact

For any inquiries, feel free to reach out:

**Zhuoxuan Cao**  
Email: [caozhuoxuan@gmail.com](mailto:caozhuoxuan@gmail.com)


  

