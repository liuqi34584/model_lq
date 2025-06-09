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

output_path = './output/pointnet_output/point_0418.pth'
output_log_path = "./output/pointnet_output/0418.txt"

def main():
    # 加载显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.cuda.is_available()
    print("正在使用：{}".format(device))
    print("显卡名称:", torch.cuda.get_device_name())

    import dataset.data_loader as data_loader
    train_loader = data_loader.get_loader_segment(batch_size=16, mode='train', dataset='HH')
    valid_loader = data_loader.get_loader_segment(batch_size=16, mode='valid', dataset='HH')
    test_loader = data_loader.get_loader_segment(batch_size=16, mode='test',   dataset='HH')

    import model_pointnet.models as models  
    import torch.optim as optim

    num_classes = 6
    model = models.PointNetDenseCls(k=num_classes,feature_transform=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]).to(device))  # 权重设置为1.0和1.0，表示两个类别的权重相同
    criterion = nn.CrossEntropyLoss().to(device)

    # state_dict = torch.load(output_path)
    # model.load_state_dict(state_dict , strict=False)
    model.to(device)

    best_acc = 0
    for epoch in range(100):
        train_loss, valid_loss = 0, 0
        train_acc, valid_acc = 0, 0

        #-------------------------------------------------------------------------训练阶段
        model.train()
        total_train_pred = []
        total_train_labels = []
        for i, (points, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            points, labels = points.float().to(device), labels.float().to(device)
            points = points.transpose(2, 1)  # [B, N, 3]-->[B, 3, N]

            pred, trans, trans_feat = model(points)
            # print(pred.shape, labels.shape)  # [B, L, num_classes] [B, L]      

            pred = pred.view(-1, num_classes)
            labels = labels.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred.float(), labels.long())
            # loss = criterion(pred, labels.long())

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
        #-------------------------------------------------------------------------验证阶段
        model.eval()
        with torch.no_grad():
            for i, (points, labels) in enumerate(valid_loader):
                points, labels = points.float().to(device), labels.float().to(device)
                points = points.transpose(2, 1)
                pred, trans, trans_feat = model(points)
                pred = pred.view(-1, num_classes)
                labels = labels.view(-1, 1)[:, 0]
                loss = F.nll_loss(pred.float(), labels.long())
                valid_loss += float(loss.item() / len(valid_loader))

        #-------------------------------------------------------------------------测试阶段
        model.eval()
        total_test_pred = []
        total_test_labels = []
        with torch.no_grad():
            for i, (points, labels) in enumerate(test_loader):
                points, labels = points.float().to(device), labels.float().to(device)
                points = points.transpose(2, 1)
                pred, trans, trans_feat = model(points)
                pred = pred.view(-1, num_classes)
                labels = labels.view(-1, 1)[:, 0]
                pred_choice = pred.data.max(1)[1]

                # print("labels.shape:", labels.shape, "pred_choice.shape:", pred_choice.shape)
                # print("labels中：1的数量 ->", (labels == 1).sum().item(), "，0的数量 ->", (labels == 0).sum().item())
                # print("pred中  ：1的数量 ->", (pred_choice == 1).sum().item(), "，0的数量 ->", (pred_choice == 0).sum().item())

                total_test_pred.extend(pred_choice.cpu().tolist())
                total_test_labels.extend(labels.cpu().tolist())

        total_test_pred = torch.tensor(total_test_pred)
        total_test_labels = torch.tensor(total_test_labels)
        acc_per_class = calculate_class_accuracy(total_test_labels, total_test_pred, num_classes=num_classes)
        test_acc = (total_test_pred == total_test_labels).float().mean().item()

        #-------------------------------------------------------------------------输出阶段
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