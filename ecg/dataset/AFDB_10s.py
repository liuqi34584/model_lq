'''
导入相关包
'''
import wfdb
import MIT_BIH_AF_function as MIT_BIH_AF
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import pywt
import random
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# '00735', '03665' 病人没有 data 数据, 所以不放入列表中

patient_ids = { 
    '04015': "ECG0",
    '04043': "ECG0",
    '04048': "ECG0",
    '04126': "ECG0",
    '04746': "ECG1",
    '04908': "ECG1",
    '04936': "ECG0",
    '05091': "ECG0",
    '05121': "ECG0",#
    '05261': "ECG0",
    '06426': "ECG1",#
    '06453': "ECG0",
    '06995': "ECG0",#
    '07162': "ECG0",
    '07859': "ECG0",
    '07879': "ECG0",
    '07910': "ECG0",
    '08215': "ECG0",
    '08219': "ECG1",#
    '08378': "ECG0",
    '08405': "ECG0",
    '08434': "ECG0",
    '08455': "ECG0"
}

# patient_ids = ['04015', '04043', '04048', '04126', '04746']
# patient_ids = {'04015': "ECG0"}

# 错误数据标注段,这是数据集标注错误的片段
err_patient_ids = {
    '05121': [[6323006, 6331256], [3738754, 3839000], [5722506, 5729506], [4950255, 4958755], [2222502,2225502], [5461250, 5470750]],
    '06426': [[1860252, 1883002], [1884250, 1943252]],
    '04015': [[1245000, 1246750], [1092750, 1095250], [1263000, 1270000], [1396500, 1398750]],
    '04043': [[3406500, 3409750], [3139500, 3145500]],
    '06995': [[6160250, 6164500], [5902500, 5907000], [6247500, 6251750]],
}

dataset = []
label = []
pos = []  # 这个是给测试集准备的，测试要查看预测失败的片段

aug_dataset = []
aug_label = []
aug_pos = []
for patient_id in patient_ids:

    mit_bih_af_path = 'C:/mycode/dataset/mit-bih-atrial-fibrillation-database-1.0.0/files/' + patient_id

    # 读取患者文件
    signal = wfdb.rdrecord(mit_bih_af_path, physical=True)
    signal_annotation = wfdb.rdann(mit_bih_af_path, "atr")
    r_peak_annotation = wfdb.rdann(mit_bih_af_path, "qrs")

    ECG_rpeaks = r_peak_annotation.sample
    ann_aux_note = signal_annotation.aux_note
    ann_sample = signal_annotation.sample
    ECG0 = signal.p_signal[:, 0]
    ECG1 = signal.p_signal[:, 1]

    # 建立信号伴随列表
    ECG_ann = np.array(MIT_BIH_AF.AFDB_create_mate_ann(len(ECG0), ann_sample, ann_aux_note))

    r_peaks_position = MIT_BIH_AF.find_nR_peaks(5, 0, len(ECG0), ECG0, ECG_rpeaks)
    # for (s, e) in r_peaks_position:
    # for i in range(0, len(r_peaks_position), 5):
        # s,e = r_peaks_position[i]
        # e = s + 2500

    for i in range(0, r_peaks_position[-1][1], 2000):
        s,e = i, i + 2500

        # 首先如果 s,e 匹配在错误区间就跳出该区间不记录
        matched = False
        if patient_id in err_patient_ids:
            for interval in err_patient_ids[patient_id]:
                if interval[0] <= s <= interval[1] or interval[0] <= e <= interval[1]:
                    matched = True
                    # print(patient_id, interval, "错误段落")
                    break
        if matched:
            continue

        signal0, signal1 = ECG0[s:e], ECG1[s:e]
        if e > ECG_rpeaks[-1]:  # 说明超过最后标注位置了
            continue

        if len(signal0)<30:  # 说明遇到不合适的标注信息了
            continue

        # r_peaks_5s = MIT_BIH_AF.find_nR_peaks(1, s, s + 1250, ECG0, ECG_rpeaks)
        # if len(r_peaks_5s) > 7:  # 说明R峰太密集了
        #     continue

        if patient_ids[patient_id] == 'ECG0':
            signal = MIT_BIH_AF.scipy_denoise(ECG0[s:e])
            # if MIT_BIH_AF.R_detect(signal, signal_freq=250) == False:
            #     continue

        if patient_ids[patient_id] == 'ECG1':
            signal = MIT_BIH_AF.scipy_denoise(ECG1[s:e])
            # if MIT_BIH_AF.R_detect(signal, signal_freq=250) == False:
            #     continue

        one_data = []
        one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[                     :int(len(signal)/10)], 512)))
        one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)/10)  :int(len(signal)*2/10)], 512)))
        one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*2/10):int(len(signal)*3/10)], 512)))
        one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*3/10):int(len(signal)*4/10)], 512)))
        one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*4/10):int(len(signal)*5/10)], 512)))
        one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*5/10):int(len(signal)*6/10)], 512)))
        one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*6/10):int(len(signal)*7/10)], 512)))
        one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*7/10):int(len(signal)*8/10)], 512)))
        one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*8/10):int(len(signal)*9/10)], 512)))
        one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*9/10):                     ], 512)))

        # one_data = []
        # one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[                    :int(len(signal)/5)  ], 512)))
        # one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)/5)  :int(len(signal)*2/5)], 512)))
        # one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*2/5):int(len(signal)*3/5)], 512)))
        # one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*3/5):int(len(signal)*4/5)], 512)))
        # one_data.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*4/5):]                    , 512)))

        one_data = np.array(one_data).T
        scaler.fit(one_data)
        one_data = scaler.transform(one_data).T

        # one_data2 = []
        # signal = -signal
        # one_data2.append(list(MIT_BIH_AF.resample_signal_length(signal[                    :int(len(signal)/5)  ], 512)))
        # one_data2.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)/5)  :int(len(signal)*2/5)], 512)))
        # one_data2.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*2/5):int(len(signal)*3/5)], 512)))
        # one_data2.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*3/5):int(len(signal)*4/5)], 512)))
        # one_data2.append(list(MIT_BIH_AF.resample_signal_length(signal[int(len(signal)*4/5):]                    , 512)))

        # one_data2 = np.array(one_data2).T
        # scaler.fit(one_data2)
        # one_data2 = scaler.transform(one_data2).T

        if MIT_BIH_AF.find_signal_label(s, e, ECG_ann) == 1:  # 房颤
            dataset.append(one_data)
            label.append([1])
            pos.append([int(patient_id), s, e])

            # aug_dataset.append(one_data2)
            # aug_label.append([1])
            # aug_pos.append([int(patient_id), s, e])
        else:  # 正常
            dataset.append(one_data)
            label.append([0])
            pos.append([int(patient_id), s, e])
        
            # aug_dataset.append(one_data2)
            # aug_label.append([0])
            # aug_pos.append([int(patient_id), s, e])

    print("{}采集完成  {}".format(patient_id, len(dataset)))



# 在这里分训练集和测试集
shuffle_index = np.random.permutation(len(dataset))  # 生成0-(X-1)的随机索引数组
random.shuffle(shuffle_index)  # 再打乱一次
train_dataset, train_label, test_dataset, test_label, test_pos = [], [], [], [], []
for i, index in enumerate(shuffle_index):
    if i < int(0.95*len(dataset)):
        train_dataset.append(dataset[index])
        train_label.append(label[index])
    else:
        test_dataset.append(dataset[index])
        test_label.append(label[index])

# 在这里分训练集和验证集
shuffle_index = np.random.permutation(len(train_dataset))  # 生成0-(X-1)的随机索引数组
random.shuffle(shuffle_index)  # 再打乱一次
train_ratio_num = int(0.95*len(shuffle_index))
valid_ratio_num = int(0.05*len(shuffle_index))
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


name = "afdb_10s_768"  
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
