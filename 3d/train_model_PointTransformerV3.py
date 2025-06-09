import torch
from torch import nn
import os
import time
import numpy as np
from datetime import datetime
from multiprocessing import freeze_support  # 导入 freeze_support
import torch.nn.functional as F
from model_pointnet.calculate import calculate_class_accuracy

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

output_path = './output/PointTransformerV3_output/PointTransformerV3_0609.pth'
output_log_path = "./output/PointTransformerV3_output/0121.txt"

def main():
    # 加载显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.cuda.is_available()
    print("正在使用：{}".format(device))
    print("显卡名称:", torch.cuda.get_device_name())

    import dataset.data_loader as data_loader
    train_loader = data_loader.get_loader_segment(batch_size=2, mode='train', dataset='PVT3')
    valid_loader = data_loader.get_loader_segment(batch_size=2, mode='valid', dataset='PVT3')
    test_loader = data_loader.get_loader_segment(batch_size=2, mode='test',   dataset='PVT3')

    import model_PointTransformerV3.model as models  # 确保模型路径正确
    import torch.optim as optim

    num_classes = 5
    model = models.PointTransformerV3(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    criterion = nn.CrossEntropyLoss().to(device)  # 使用 CrossEntropyLoss

    model.to(device)

    best_acc = 0
    for epoch in range(100):
        train_loss, valid_loss = 0, 0
        train_acc, valid_acc = 0, 0

        # -------------------------------------------------------------------------训练阶段
        model.train()
        total_train_pred = []
        total_train_labels = []
        for i, (points, labels, colors) in enumerate(train_loader):
            optimizer.zero_grad()
            points, labels, colors = points.float().to(device), labels.long().to(device), colors.float().to(device)

            # print(points.shape, labels.shape)  # [B, N, 3] [B, N]
            points = points.transpose(2, 1)  # [B, 3, N]
            colors = colors.transpose(2, 1)  # [B, 3, N]
            pred = model(points, colors)

            # print(pred.shape, labels.shape)  # [B, numclass, N] [B, N]
            # pred = pred.view(-1, num_classes)

            pred = pred.reshape(-1, num_classes)
            labels = labels.view(-1, 1)[:, 0]
            loss = criterion(pred.float(), labels.long())

            train_loss += float(loss.item() / len(train_loader))

            pred_choice = pred.argmax(1)
            total_train_pred.extend(pred_choice.cpu().tolist())
            total_train_labels.extend(labels.cpu().tolist())

            loss.backward()
            optimizer.step()

        scheduler.step()

        # 计算训练集平均准确率
        total_train_pred = torch.tensor(total_train_pred)
        total_train_labels = torch.tensor(total_train_labels)
        train_acc = (total_train_pred == total_train_labels).float().mean().item()

        # -------------------------------------------------------------------------验证阶段
        model.eval()
        with torch.no_grad():
            for i, (points, labels, colors) in enumerate(valid_loader):
                points, labels, colors = points.float().to(device), labels.long().to(device), colors.float().to(device)
                points = points.transpose(2, 1)  # [B, 3, N]
                colors = colors.transpose(2, 1)  # [B, 3, N]
                pred = model(points, colors)

                # pred = pred.view(-1, num_classes)
                pred = pred.reshape(-1, num_classes)
                labels = labels.view(-1, 1)[:, 0]
                loss = criterion(pred.float(), labels.long())
                valid_loss += float(loss.item() / len(valid_loader))

        # -------------------------------------------------------------------------测试阶段
        model.eval()
        total_test_pred = []
        total_test_labels = []
        with torch.no_grad():
            for i, (points, labels, colors) in enumerate(test_loader):
                points, labels, colors = points.float().to(device), labels.long().to(device), colors.float().to(device)
                points = points.transpose(2, 1)  # [B, 3, N]
                colors = colors.transpose(2, 1)  # [B, 3, N]
                pred = model(points, colors)

                # pred = pred.view(-1, num_classes)
                pred = pred.reshape(-1, num_classes)
                labels = labels.view(-1, 1)[:, 0]
                pred_choice = pred.data.max(1)[1]
                total_test_pred.extend(pred_choice.cpu().tolist())
                total_test_labels.extend(labels.cpu().tolist())

        total_test_pred = torch.tensor(total_test_pred)
        total_test_labels = torch.tensor(total_test_labels)

        acc_per_class = calculate_class_accuracy(total_test_labels, total_test_pred, num_classes=num_classes)
        test_acc = (total_test_pred == total_test_labels).float().mean().item()

        # -------------------------------------------------------------------------输出阶段
        test_acc_str = " ".join([f"{i}:{acc:.4f}" for i, acc in enumerate(acc_per_class)])
        records_str = f"\nEpoch:{epoch + 1:<2d} {datetime.now().strftime('%H:%M:%S')}|loss_t:{train_loss:.8f} loss_v:{valid_loss:.8f}|train_acc:{train_acc:.5f}|{test_acc_str} |test_acc:{test_acc:.5f}"

        with open(output_log_path, 'a') as file:
            file.write(records_str)

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), output_path)

        print(records_str[1:])


if __name__ == '__main__':
    freeze_support()  # 如果程序不需要打包成可执行文件，可以省略这一行
    main()
