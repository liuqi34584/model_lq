import torch
from torch import nn
import torch.nn.functional as F

def calculate_class_accuracy(labels, pred_choice, num_classes=8):

    # 初始化每个类别的总数和正确预测数
    total_per_class = torch.zeros(num_classes, dtype=torch.float32)
    correct_per_class = torch.zeros(num_classes, dtype=torch.float32)


    # 统计每个类别的总数
    for class_idx in range(num_classes):
        total_per_class[class_idx] = (labels == class_idx).sum().item()

    # 统计每个类别预测正确的数量
    correct_mask = (labels == pred_choice)
    for class_idx in range(num_classes):
        class_correct = ((labels == class_idx) & correct_mask).sum().item()
        correct_per_class[class_idx] = class_correct

    # 计算每个类别的准确率
    accuracy_per_class = correct_per_class / total_per_class

    # 处理可能出现的除零情况
    accuracy_per_class[torch.isnan(accuracy_per_class)] = 0

    return accuracy_per_class


if __name__ == '__main__':
    # 假设这是你的真实标签和预测结果
    labels = torch.randint(0, 8, (192000,))
    pred_choice = torch.randint(0, 8, (192000,))

    accuracy_per_class = calculate_class_accuracy(labels, pred_choice, num_classes=8)

    # 输出每个类别的准确率
    for class_idx in range(8):
        print(f"Class {class_idx} accuracy: {accuracy_per_class[class_idx]:.4f}")
    