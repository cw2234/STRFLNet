"""
留一法，14个被试用于练，1个被试用于测，用第一个session的
"""

import numpy as np

import torch
import random

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import LeaveOneOut

from matplotlib import pyplot as plt
import os
import re


from model_nineTGtrans import GODE

from train import train, test

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")


class Data(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


def set_seed(seed: int = 42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_train_test(X_train: torch.Tensor, X_test: torch.Tensor):
    # (sample, times, channel)
    # 每个通道分别算自己的平均
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True)
    # 归一化 z-score
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm, mean, std


def get_subject_list(dataset_dir: str) -> list[str]:
    """获取所有被试的名称

    Returns:
        list[str]: 所有被试的名称
    """
    sub_dirs = []
    for item in os.listdir(dataset_dir):
        # 匹配 sub-数字 格式的目录
        if re.match(r"sub-\d+", item) and os.path.isdir(
            os.path.join(dataset_dir, item)
        ):
            sub_dirs.append(item)

    # 按数字顺序排序 (sub-01, sub-02, ...)
    sub_dirs.sort(key=lambda x: int(x.split("-")[-1]))

    return sub_dirs


def load_all_data():
    data_dir = "./first_session_data_npy"
    sub_dirs = get_subject_list(data_dir)
    data_list = []
    label_list = []
    for sub_name in sub_dirs:
        sub_dir = os.path.join(data_dir, sub_name)
        print(f"读取 {sub_name} 的数据")
        data = np.load(os.path.join(sub_dir, f"{sub_name}_data.npy"))
        label = np.load(os.path.join(sub_dir, f"{sub_name}_label.npy"))
        # 标签是-1，0，1，这里+1后是0，1，2
        label += 1
        print(f"{sub_name} 的数据形状为 {data.shape}")
        data_list.append(data)
        label_list.append(label)

    return data_list, label_list


def main():

    set_seed(42)
    data_list, label_list = load_all_data()

    num_classes = 3

    # 将模型和数据移到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 训练模型
    num_epochs = 50
    batch_size = 32
    mean_test_acc1 = []
    mean_test_loss1 = []
    std_test_acc1 = []
    i = 0
    milestones = [10]

    # 留一法
    loo = LeaveOneOut()
    print("model begin")

    for fold, (train_index, test_index) in enumerate(
        loo.split(np.arange(len(data_list)))
    ):
        model = GODE().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        train_losses = []
        test_losses = []
        test_accuracies = []
        X_train = np.concatenate([data_list[i] for i in train_index], axis=0)
        X_test = np.concatenate([data_list[i] for i in test_index], axis=0)
        y_train = np.concatenate([label_list[i] for i in train_index], axis=0)
        y_test = np.concatenate([label_list[i] for i in test_index], axis=0)

        # 转换为张量
        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

        # 归一化
        X_train_norm, X_test_norm, mean, std = normalize_train_test(X_train, X_test)

        # 查看每个情感的样本数量
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Fold {fold} train labels: {dict(zip(unique, counts))}")
        unique, counts = np.unique(y_test, return_counts=True)
        print(f"Fold {fold} test labels: {dict(zip(unique, counts))}")

        train_dataset = TensorDataset(X_train_norm, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(X_test_norm, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        i = i + 1
        for epoch in range(num_epochs):
            current_lr = optimizer.param_groups[0]["lr"]

            train_acc, train_loss = train(train_loader, model, criterion, optimizer)
            train_losses.append(train_loss)
            # 模型评估
            test_acc, test_loss = test(test_loader, model, criterion)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            print(
                "Epoch [{}/{}], 第 {} 组, Learning Rate:{}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}% ".format(
                    epoch + 1,
                    num_epochs,
                    i,
                    current_lr,
                    train_loss,
                    train_acc,
                    test_loss,
                    test_acc,
                )
            )
            scheduler.step()
        mean_test_acc1.append(test_accuracies[-1])
        mean_test_loss1.append(test_losses[-1])
        std_test_acc1.append(test_accuracies[-1])
    print("Mean Test Accuracy: {:.2f}%".format(np.mean(mean_test_acc1)))
    print("Mean Test Loss: {:.4f}".format(np.mean(mean_test_loss1)))
    print("Std Test Accuracy: {:.2f}%".format(np.std(std_test_acc1)))

    colors1 = ["r"]
    colors2 = ["purple", "orange", "c"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        color=colors1[0],
        label=" Train Loss",
    )

    ax.set_title("Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    plt.savefig("train_loss_XTGODE_cross.png")
    plt.close()

    fig1, ax = plt.subplots(figsize=(8, 8))

    ax.plot(
        range(1, len(test_losses) + 1),
        test_losses,
        color=colors1[0],
        label=" Test Loss",
    )

    ax.set_title("Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    plt.savefig("test_los_XTGODE_transformer_cross.png")
    plt.close()

    # Plotting test accuracies
    fig2, ax = plt.subplots(figsize=(8, 8))

    ax.plot(
        range(1, len(test_accuracies) + 1),
        test_accuracies,
        color=colors1[0],
        label=" Accuracy",
    )

    ax.set_title("Test Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right")
    plt.savefig("test_accuracy_XTGODE_transformer_cross.png")
    plt.close()


if __name__ == "__main__":
    main()
