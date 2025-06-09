'''
导入相关包
'''
import torch
from torch import nn
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

output_path = './output/transformer_gpt/AF1.pth'
output_log_path = "./output/transformer_gpt/0121.txt"

# '''
# 加载显卡
# '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.cuda.is_available()
print("正在使用：{}".format(device))
print("显卡名称:", torch.cuda.get_device_name())

# '''
# 加载数据  dataset='CPSC2025'   dataset='MIT_BIH_AF'
# '''
import dataset.data_loader as data_loader
train_loader = data_loader.get_loader_segment(batch_size=1024, mode='train', dataset='CPSC2025')
valid_loader = data_loader.get_loader_segment(batch_size=1024, mode='valid', dataset='CPSC2025')
test_loader  = data_loader.get_loader_segment(batch_size=1024, mode='test' , dataset='CPSC2025')

# '''
# 模型搭建
# '''
import model_transformer_gpt.Models as models
import model_transformer_gpt.Optim as Optimizer
import torch.optim as optim

model = models.Transformer()
optimizer = Optimizer.ScheduledOptim(optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), lr_mul=8, d_model=512, n_warmup_steps=2000)
criterion = nn.CrossEntropyLoss()
# state_dict = torch.load(output_path)
# model.load_state_dict(state_dict , strict=False)
model.to(device)

best_acc = 0

for epoch in range(100):

    train_loss, valid_loss = 0, 0
    train_acc, valid_acc = 0, 0
    test_acc, test_sensitivity, test_specificity = 0, 0, 0

    #-------------------------------------------------------------------------训练阶段
    model.train()
    for i, (signal_data, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        enc_inputs, labels = signal_data.to(device), labels.to(device)

        output = model(enc_inputs)  # torch.Size([8192, 1, 2])
        loss = criterion(output.view(-1, 2), labels.long())
        train_loss += float(loss.item()/(labels.size(0)*len(train_loader)))

        output = torch.argmax(output, dim=2)  # torch.Size([1024, 1])
        y_true, y_pred = labels.cpu().numpy(), output.squeeze().cpu().numpy()
        # (1024,) (1024,)

        tp = np.sum((y_true == 1) & (y_pred == 1))  # TP
        fp = np.sum((y_true == 0) & (y_pred == 1))  # FP
        fn = np.sum((y_true == 1) & (y_pred == 0))  # FN
        tn = np.sum((y_true == 0) & (y_pred == 0))  # TN

        accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-7)
        train_acc += accuracy/len(train_loader)

        loss.backward()
        optimizer.step_and_update_lr()

    #-------------------------------------------------------------------------验证阶段
    model.eval()
    with torch.no_grad():
        for i, (signal_data, labels) in enumerate(valid_loader):

            enc_inputs, labels = signal_data.to(device), labels.to(device)
            output = model(enc_inputs)  # torch.Size([8192, 1, 2])
            loss = criterion(output.view(-1, 2), labels.long())
            valid_loss += float(loss.item()/(labels.size(0)*len(valid_loader)))
            output = torch.argmax(output, dim=2)

            y_true, y_pred = labels.cpu().numpy(), output.squeeze().cpu().numpy()
            tp = np.sum((y_true == 1) & (y_pred == 1))  # TP
            fp = np.sum((y_true == 0) & (y_pred == 1))  # FP
            fn = np.sum((y_true == 1) & (y_pred == 0))  # FN
            tn = np.sum((y_true == 0) & (y_pred == 0))  # TN

            accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-7)
            sensitivity = tp / (tp + fn + 1e-7)
            specificity = tn / (tn + fp + 1e-7)

            valid_acc += accuracy/len(valid_loader)

    #-------------------------------------------------------------------------测试阶段
    model.eval()  # test_loader
    with torch.no_grad():
        for i, (signal_data, labels) in enumerate(test_loader):
 
            enc_inputs, labels = signal_data.to(device), labels.to(device)
            output = model(enc_inputs)
            output = torch.argmax(output, dim=2)

            y_true, y_pred = labels.cpu().numpy(), output.squeeze().cpu().numpy()
            tp = np.sum((y_true == 1) & (y_pred == 1))  # TP
            fp = np.sum((y_true == 0) & (y_pred == 1))  # FP
            fn = np.sum((y_true == 1) & (y_pred == 0))  # FN
            tn = np.sum((y_true == 0) & (y_pred == 0))  # TN

            accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-7)
            sensitivity = tp / (tp + fn + 1e-7)
            specificity = tn / (tn + fp + 1e-7)

            test_acc += accuracy/len(test_loader)
            test_sensitivity += sensitivity/len(test_loader)
            test_specificity += specificity/len(test_loader)


    #-------------------------------------------------------------------------输出阶段
    records_str = "\nEpoch:{:<2d} {} |loss train:{:.10f}  loss valid:{:.10f} |train_acc:{:.5f}  v_acc:{:.5f} |t_acc:{:.5f} t_sen:{:.5f} t_spc:{:.5f}".format(
        epoch+1, datetime.now().strftime("%H:%M:%S"), train_loss, valid_loss, train_acc, valid_acc, test_acc, test_sensitivity, test_specificity)

    with open(output_log_path, 'a') as file:
        file.write(records_str)

    if test_acc >= best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), output_path)

    print(records_str[1:])


# pytorch 1.10  python 3.8