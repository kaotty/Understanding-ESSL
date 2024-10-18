# Understanding the Role of Equivariance in Self-supervised Learning
This repository includes a PyTorch implementation of the NeurIPS 2024 paper [Understanding the Role of Equivariance in Self-supervised Learning]() authored by [Yifei Wang*](https://yifeiwang77.com/), Kaiwen Hu*, [Sharut Gupta](https://www.mit.edu/~sharut/), [Ziyu Ye](https://hazelye-bot.github.io/), [Yisen Wang](https://yisenwang.github.io/), and [Stefanie Jegelka](https://people.csail.mit.edu/stefje/).

## Abstract
Contrastive learning has been a leading paradigm for self-supervised learning, but it is widely observed that it comes at the price of sacrificing useful features (e.g., colors) by being invariant to data augmentations. Given this limitation, there has been a surge of interest in equivariant self-supervised learning (E-SSL) that learns features to be augmentation-aware. However, even for the simplest rotation prediction method, there is a lack of rigorous understanding of why, when, and how E-SSL learns useful features for downstream tasks. To bridge this gap between practice and theory, we establish an information-theoretic perspective to understand the generalization ability of E-SSL. In particular, we identify a critical explaining-away effect in E-SSL that creates a synergy between the equivariant and classification tasks. This synergy effect encourages models to extract class-relevant features to improve its equivariant prediction, which, in turn, benefits downstream tasks requiring semantic features. Based on this perspective, we theoretically analyze the influence of data transformations and reveal several principles for practical designs of E-SSL. Our theory not only aligns well with existing E-SSL methods but also sheds light on new directions by exploring the benefits of model equivariance. We believe that a theoretically grounded understanding on the role of equivariance would inspire more principled and advanced designs in this field.

## Instructions
All experiments are conducted with a single NVIDIA RTX 3090 GPU. We mainly conduct the following experiments on CIFAR-10 and CIFAR-100.

### Different Equivariant Pretraining Tasks
In this experiment, we conduct equivariant pretraining tasks based on seven different types of transformations. In order to maintain fairness and avoid cross-interactions, we only apply random crops to the raw images before we move on to these tasks.

In order to conduct the experiments, you can enter the ESSL folder and run the following command.
```bash
python equivariant_tasks.py method=four_fold_rotation
```
You may select the method from `{horizontal_flips, vertical_flips, four_fold_rotation, color_inversions, grayscale, jigsaws, four_fold_blurs}`.
You may also set method as `none` to run the baseline.

### How Class Information Affects Equivariant Pretraining Tasks
In this experiment, our goal is to figure out how class information affects rotation prediction. Figure~\ref{fig:model of experiment B.2} demonstrates the outline of the model we use to conduct this experiment. We apply random crops with size 32 and horizontal flips with probability 0.5 to the raw images.

In order to conduct the experiments, you can enter the ESSL folder and run the following commands respectively.
```bash
python verification.py method=normal
python verification.py method=add
python verification.py method=eliminate
```
![Rotation Accuracy on CIFAR-10]("C:\Users\kaott\OneDrive\桌面\NeurIPS2024\ESSL\Rotation-cifar10.png" "Rotation Accuracy on CIFAR-10")
![Rotation Accuracy on CIFAR-100]("C:\Users\kaott\OneDrive\桌面\NeurIPS2024\ESSL\Rotation-cifar100.png""Rotation Accuracy on CIFAR-100")

### The Study of Model Equivariance
In order to compare the performance of Resnet and EqResnet, we use rotation prediction as our pretraining task and obtain the linear probing results. We apply various augmentations to the raw images, such as no augmentation, a combination of random crops with size 32 and horizontal flips, and SimCLR augmentations with an output of 32x32. To be more specific, a SimCLR augmentation refers to a sequence of transformations, including a random resized crop with size 32 and scale 0.2-1.0, horizontal flip with probability 0.5, color jitter with probability 0.8, and finally grayscale with probability 0.2.

In order to conduct the experiments, you can enter the Equivariant Network folder and run the following command.
```bash
python train.py --model resnet18 --dataset cifar10 --train_aug sup --head mlp
```
You may select the dataset from `{cifar10, cifar100}`, the training augmentation from `{none, sup, simclr}`, the projection head from `{mlp, linear}`.

## Citing this work

## Acknowledgement
Our code partly follows the official implementation of Efficient Equivariant Network (https://github.com/LingshenHe/Efficient-Equivariant-Network).
