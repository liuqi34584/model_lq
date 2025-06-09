from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
import open3d as o3d

class H_Dataset(data.Dataset):
    def __init__(self, mode='train'):


        self.root_dir = r'C:\code\dataset_show\dataset\H_txt_pcd_segmentation_100'
        # self.root_dir = r"C:\code\dataset_show\dataset\H_txt_pcd_segmentation_360"
       
        # self.root_dir = r'C:\code\dataset_show\dataset\H_txt_pcd_segmentation_100_test'
        self.mode = mode
        self.npoints = 10000  # 根据数据集最大点数设定
        self.datapath = []

        # 根据split选择对应的文件夹
        split_dir = os.path.join(self.root_dir, mode)
        if os.path.exists(split_dir):
            # 收集所有PCD文件路径
            for fname in os.listdir(split_dir):
                if fname.endswith('.pcd'):
                    self.datapath.append(os.path.join(split_dir, fname))
        else:
            raise ValueError(f"Split directory {split_dir} does not exist.")

        print(mode, "数据集大小:", len(self.datapath))

    def __getitem__(self, index):
        # 读取并解析PCD文件
        file_path = self.datapath[index]
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # 定位数据起始位置
        data_start = lines.index('DATA ascii') + 1

        # 解析点云数据
        points = []
        labels = []
        for line in lines[data_start:]:
            if not line:
                continue
            parts = line.split()
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            labels.append(int(parts[3]))

        # 转换为numpy数组
        point_set = np.array(points, dtype=np.float32)
        seg = np.array(labels, dtype=np.int64)

        if self.mode == 'train':  # 数据增强
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            point_set = np.dot(point_set, rotation_matrix)

        # 重采样核心逻辑
        if len(seg) >= self.npoints:
            # 随机选择不重复的索引
            choice = np.random.choice(len(seg), self.npoints, replace=False)
        else:
            # 允许重复采样
            choice = np.random.choice(len(seg), self.npoints, replace=True)

        point_set = point_set[choice]
        seg = seg[choice]

        # # 标准化处理
        # mean = np.mean(point_set, axis=0)
        # std = np.std(point_set, axis=0)
        # point_set_standardized = (point_set - mean) / std

        # 归一化到 [0, 1] 范围
        min_val = np.min(point_set, axis=0)
        max_val = np.max(point_set, axis=0)
        point_set_normalized = (point_set - min_val) / (max_val - min_val)

        return point_set, seg

    def __len__(self):
        return len(self.datapath)

class Pointnet_2_Dataset(data.Dataset):
    def __init__(self, mode='train'):

        # self.root_dir = r"C:\code\dataset_show\dataset\gong3_dataset"
        # self.root_dir = r"C:\code\dataset_AnShanBaiLai\AnShanBaiNai_humian_dataset"
        self.root_dir = r"C:\code\dataset_hx\hx_dataset"

        self.mode = mode
        self.npoints = 5000  # 根据数据集最大点数设定，模型输入仍然是1024
        self.datapath = []

        # 根据split选择对应的文件夹
        split_dir = os.path.join(self.root_dir, mode)
        if os.path.exists(split_dir):
            # 收集所有PCD文件路径
            for fname in os.listdir(split_dir):
                if fname.endswith('.pcd'):
                    self.datapath.append(os.path.join(split_dir, fname))
        else:
            raise ValueError(f"Split directory {split_dir} does not exist.")

        print(mode, "数据集大小:", len(self.datapath))

    def __getitem__(self, index):
        # 读取并解析PCD文件
        file_path = self.datapath[index]
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # 定位数据起始位置
        data_start = lines.index('DATA ascii') + 1

        # 解析点云数据
        points = []
        labels = []
        colors = []
        for line in lines[data_start:]:
            if not line:
                continue
            parts = line.split()
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            labels.append(int(parts[3]))
            colors.append([float(parts[7]), float(parts[8]), float(parts[9])])

        # 转换为numpy数组
        point_set = np.array(points, dtype=np.float32)
        seg = np.array(labels, dtype=np.int64)
        colors_set = np.array(colors, dtype=np.int64)

        # if self.mode == 'train':  # 数据增强
        #     theta = np.random.uniform(0, 2 * np.pi)
        #     rotation_matrix = np.array([
        #         [np.cos(theta), -np.sin(theta), 0],
        #         [np.sin(theta), np.cos(theta), 0],
        #         [0, 0, 1]
        #     ])
        #     point_set = np.dot(point_set, rotation_matrix)

        # 重采样核心逻辑
        if len(seg) >= self.npoints:
            # 随机选择不重复的索引
            choice = np.random.choice(len(seg), self.npoints, replace=False)
        else:
            # 允许重复采样
            choice = np.random.choice(len(seg), self.npoints, replace=True)

        point_set = point_set[choice]
        seg = seg[choice]
        colors_set = colors_set[choice]
        # print(point_set.shape, seg.shape, colors_set.shape)

        return point_set, seg, colors_set

    def __len__(self):
        return len(self.datapath)


class PointTransformerV3_Dataset(data.Dataset):
    def __init__(self, mode='train'):

        # self.root_dir = r"C:\code\dataset_show\dataset\gong3_dataset"
        self.root_dir = r"C:\code\dataset_AnShanBaiLai\AnShanBaiNai_humian_dataset"
        # self.root_dir = r"C:\code\dataset_hx\hx_dataset"

        self.mode = mode
        self.npoints = 80000  # 根据数据集最大点数设定
        self.datapath = []

        # 根据split选择对应的文件夹
        split_dir = os.path.join(self.root_dir, mode)
        if os.path.exists(split_dir):
            # 收集所有PCD文件路径
            for fname in os.listdir(split_dir):
                if fname.endswith('.pcd'):
                    self.datapath.append(os.path.join(split_dir, fname))
        else:
            raise ValueError(f"Split directory {split_dir} does not exist.")

        print(mode, "数据集大小:", len(self.datapath))

    def __getitem__(self, index):
        # 读取并解析PCD文件
        file_path = self.datapath[index]
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # 定位数据起始位置
        data_start = lines.index('DATA ascii') + 1

        # 解析点云数据
        points = []
        labels = []
        colors = []
        for line in lines[data_start:]:
            if not line:
                continue
            parts = line.split()
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            labels.append(int(parts[3]))
            colors.append([float(parts[7]), float(parts[8]), float(parts[9])])

        # 转换为numpy数组
        point_set = np.array(points, dtype=np.float32)
        seg = np.array(labels, dtype=np.int64)
        colors_set = np.array(colors, dtype=np.int64)

        # 重采样核心逻辑
        if len(seg) >= self.npoints:
            # 随机选择不重复的索引
            choice = np.random.choice(len(seg), self.npoints, replace=False)
        else:
            # 允许重复采样
            choice = np.random.choice(len(seg), self.npoints, replace=True)

        point_set = point_set[choice]
        seg = seg[choice]
        colors_set = colors_set[choice]
        # print(point_set.shape, seg.shape, colors_set.shape)

        # if self.mode == 'train':  # 数据增强
        #     theta = np.random.uniform(0, 2 * np.pi)
        #     rotation_matrix = np.array([
        #         [np.cos(theta), -np.sin(theta), 0],
        #         [np.sin(theta), np.cos(theta), 0],
        #         [0, 0, 1]
        #     ])
        #     point_set = np.dot(point_set, rotation_matrix)


        return point_set, seg, colors_set

    def __len__(self):
        return len(self.datapath)


def get_loader_segment(batch_size, mode='train', dataset='HH'):
    if dataset == 'HH':
        dataset = H_Dataset(mode=mode)
    elif dataset == 'H++':
        dataset = Pointnet_2_Dataset(mode=mode)
    elif dataset == 'PVT3':
        dataset = PointTransformerV3_Dataset(mode=mode)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        # collate_fn=PointTransformerV3_Dataset.point_transformer_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return data_loader