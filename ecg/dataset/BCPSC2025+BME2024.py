import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import random

# 根据将信号重采样长度
# 传入：被重采样的信号，想要重采样的长度
# 返回值：被重采样的信号
# 使用举例：signal3 = resample_signal_length(signal2, 800)
# 作者：刘琦
def resample_signal_length(ori_signal, length):
    import numpy
    from scipy.interpolate import interp1d

    # 创建线性插值函数
    f = interp1d(numpy.arange(0, len(ori_signal)), ori_signal, kind='linear', fill_value="extrapolate")

    # 使用插值函数进行重采样
    new_signal = f(numpy.linspace(0, len(ori_signal), length))

    return new_signal[0:length]

#==============================================================加载BME2024数据集================================
BME2024_AF_data =h5py.File('other_file/BME2024/Training/Training/train_af.mat')
print(BME2024_AF_data.keys())
BME2024_AF_data=BME2024_AF_data['train_af']  # (4000, 20000) 采样率为 400 Hz，信号时间长度 10 s，
BME2024_AF_data = np.array(BME2024_AF_data)
BME2024_AF_data = BME2024_AF_data.T # (20000, 4000)

BME2024_Normal_Data =h5py.File('other_file/BME2024/Training/Training/train_nor.mat')
print(BME2024_Normal_Data.keys())
BME2024_Normal_Data = BME2024_Normal_Data['train_naf']  # (4000, 20000) 采样率为 400 Hz，信号时间长度 10 s，
BME2024_Normal_Data = np.array(BME2024_Normal_Data)
BME2024_Normal_Data = BME2024_Normal_Data.T # (20000, 4000)
print("BME2024_房颤数据形状:",   BME2024_AF_data.shape)
print("BME2024_非房颤数据形状:", BME2024_Normal_Data.shape)


#==============================================================加载CPSC2025数据集================================
CPSC2025_train_X=h5py.File('other_file/CPSC2025/traindata.mat')
print(CPSC2025_train_X.keys())
CPSC2025_train_X=CPSC2025_train_X['traindata']  # (4000, 20000) 采样率为 400 Hz，信号时间长度 10 s，
CPSC2025_train_X = np.array(CPSC2025_train_X)
CPSC2025_train_X = CPSC2025_train_X.T # (20000, 4000)

# 分割数据
CPSC2025_atrial_fibrillation_data = CPSC2025_train_X[:500]
CPSC2025_non_atrial_fibrillation_data = CPSC2025_train_X[500:1000]
CPSC2025_unlabeled_data = CPSC2025_train_X[1000:]

print("CPSC2025_房颤数据形状:",   CPSC2025_atrial_fibrillation_data.shape)
print("CPSC2025_非房颤数据形状:", CPSC2025_non_atrial_fibrillation_data.shape)
print("CPSC2025_无标签数据形状:", CPSC2025_unlabeled_data.shape)


#==============================================================划分CPSC2025数据集================================

CPSC2025_dataset = []
CPSC2025_label = []
signal_length = 768

# 遍历CPSC2025房颤数据信号
for i, signal in enumerate(CPSC2025_atrial_fibrillation_data):

    # 这里可以添加对信号的具体处理逻辑
    # 将长度为4000的信号重采样为长度为512的信号
    one_data = []
    one_data.append(list(resample_signal_length(signal, signal_length)))

    one_data = np.array(one_data).T
    scaler.fit(one_data)
    one_data = scaler.transform(one_data).T

    CPSC2025_dataset.append(one_data)
    CPSC2025_label.append([1])  # 房颤指定为1


# 遍历CPSC2025非房颤数据信号
for i, signal in enumerate(CPSC2025_non_atrial_fibrillation_data):

    # 这里可以添加对信号的具体处理逻辑
    # 将长度为4000的信号重采样为长度为512的信号
    one_data = []
    one_data.append(list(resample_signal_length(signal, signal_length)))

    one_data = np.array(one_data).T
    scaler.fit(one_data)
    one_data = scaler.transform(one_data).T

    CPSC2025_dataset.append(one_data)
    CPSC2025_label.append([0])  # 非房颤指定为0

# 遍历无标签数据信号
print("开始遍历无标签数据信号：先不处理")
# for i, signal in enumerate(unlabeled_data):
#     print(f"第 {i + 1} 条无标签信号：信号长度为 {len(signal)}")
    # 这里可以添加对信号的具体处理逻辑

# 在这里分训练集和测试集
shuffle_index = np.random.permutation(len(CPSC2025_dataset))  # 生成0-(X-1)的随机索引数组
random.shuffle(shuffle_index)  # 再打乱一次
train_dataset, train_label, test_dataset, test_label= [], [], [], []
for i, index in enumerate(shuffle_index):
    if i < int(0.90*len(CPSC2025_dataset)):  # 0.95
        train_dataset.append(CPSC2025_dataset[index])
        train_label.append(CPSC2025_label[index])
    else:
        test_dataset.append(CPSC2025_dataset[index])
        test_label.append(CPSC2025_label[index])

print("CPSC2025采集划分完成 train数量:{} test数量:{}".format(len(train_dataset), len(test_dataset)))

#==============================================================将BME2024数据集加到训练集中================================


# 遍历BME2024房颤数据信号
for i, signal in enumerate(BME2024_AF_data):

    # 这里可以添加对信号的具体处理逻辑
    # 将长度为4000的信号重采样为长度为512的信号
    one_data = []
    one_data.append(list(resample_signal_length(signal, signal_length)))

    one_data = np.array(one_data).T
    scaler.fit(one_data)
    one_data = scaler.transform(one_data).T

    train_dataset.append(one_data)
    train_label.append([1])  # 房颤指定为1

# 遍历BME2024非房颤数据信号
for i, signal in enumerate(BME2024_AF_data):

    # 这里可以添加对信号的具体处理逻辑
    # 将长度为4000的信号重采样为长度为512的信号
    one_data = []
    one_data.append(list(resample_signal_length(signal, signal_length)))

    one_data = np.array(one_data).T
    scaler.fit(one_data)
    one_data = scaler.transform(one_data).T

    train_dataset.append(one_data)
    train_label.append([0])  # 非房颤指定为0

print("BME2024补充到训练集  train数量:{} test数量:{}".format(len(train_dataset), len(test_dataset)))

# 在这里分训练集和验证集
shuffle_index = np.random.permutation(len(train_dataset))  # 生成0-(X-1)的随机索引数组
random.shuffle(shuffle_index)  # 再打乱一次
train_ratio_num = int(0.90*len(shuffle_index))
valid_ratio_num = int(0.10*len(shuffle_index))
train_X, train_Y = [], []
valid_X, valid_Y = [], []
for i, index in enumerate(shuffle_index):
    X = train_dataset[index]
    Y = train_label[index]
    if i < train_ratio_num:
        train_X.append(X)
        train_Y.append(Y)
    else:
        valid_X.append(X)
        valid_Y.append(Y)


test_X, test_Y, test_P = [], [], []
for i in range(len(test_dataset)):
    X = test_dataset[i]
    Y = test_label[i]

    test_X.append(X)
    test_Y.append(Y)


train_X = np.array(train_X)
train_Y = np.array(train_Y)

valid_X = np.array(valid_X)
valid_Y = np.array(valid_Y)

test_X = np.array(test_X)
test_Y = np.array(test_Y)


train_X_data_dict = {}
train_X_data_dict["train_X"] = train_X
train_Y_data_dict = {}
train_Y_data_dict["train_Y"] = train_Y

valid_X_data_dict = {}
valid_X_data_dict["valid_X"] = valid_X
valid_Y_data_dict = {}
valid_Y_data_dict["valid_Y"] = valid_Y

test_X_data_dict = {}
test_X_data_dict["test_X"] = test_X
test_Y_data_dict = {}
test_Y_data_dict["test_Y"] = test_Y


name = "CPSC2025_BME2024_768"  
# name = "mian404_psvt_ecg"  

from scipy.io import savemat
import os

if not os.path.exists("./dataset/"+ name):
    os.makedirs("./dataset/"+ name)


savemat("./dataset/"+ name + "/train_X.mat", train_X_data_dict)
savemat("./dataset/"+ name + "/train_Y.mat", train_Y_data_dict)

savemat("./dataset/"+ name + "/valid_X.mat", valid_X_data_dict)
savemat("./dataset/"+ name + "/valid_Y.mat", valid_Y_data_dict)

savemat("./dataset/"+ name + "/test_X.mat",  test_X_data_dict)
savemat("./dataset/"+ name + "/test_Y.mat",  test_Y_data_dict)


# 打印读取的数据形状，验证保存是否正确
print("读取的训练集数据形状------train:", train_X.shape)
print("读取的训练集标签形状------train:", train_Y.shape)
print("读取的验证集数据形状------valid:", valid_X.shape)
print("读取的验证集标签形状------valid:", valid_Y.shape)
print("读取的测试集数据形状------test:", test_X.shape)
print("读取的测试集标签形状------test:", test_Y.shape)


