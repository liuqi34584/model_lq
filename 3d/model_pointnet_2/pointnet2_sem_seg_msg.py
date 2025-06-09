import torch 
import torch.nn as nn
import torch.nn.functional as F
from model_pointnet_2.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation

class Pointnet2_Large(nn.Module):
    def __init__(self, num_classes):
        super(Pointnet2_Large, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(4096, [0.1, 0.2], [32, 64],        3,     [[32, 32, 64],     [64, 64, 128]])
        self.sa2 = PointNetSetAbstractionMsg(2048, [0.2, 0.4], [32, 64],   64+128,  [[128, 128, 256],    [256, 256, 512]])
        self.sa3 = PointNetSetAbstractionMsg( 512, [0.4, 0.8], [32, 64],  256+512,  [[256, 256, 512],   [512, 512, 1024]])
        self.sa4 = PointNetSetAbstractionMsg( 128, [0.8, 1.6], [32, 64], 512+1024, [[512, 512, 1024], [1024, 1024, 2048]])
        self.fp4 = PointNetFeaturePropagation(3072 + 1536, [1024, 512])
        self.fp3 = PointNetFeaturePropagation(512 + 768, [512, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 192, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, colors):

        mean = torch.mean(xyz, dim=2, keepdim=True)
        std = torch.std(xyz, dim=2, keepdim=True)
        std[std == 0] = 1e-8  # 避免除零错误
        xyz = (xyz - mean) / std
        # print(l0_points.shape, l0_xyz.shape)

        # # 向量化计算每个 batch 的质心并移动到原点
        # centroid = torch.mean(xyz, dim=2, keepdim=True)
        # xyz = xyz - centroid

        l0_points = xyz
        l0_xyz = xyz

        # print(l0_points.shape, l0_xyz.shape)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        # return x, l4_points
        return x


class Pointnet_2(nn.Module):
    def __init__(self, num_classes):
        super(Pointnet_2, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, colors):
        # print(l0_points.shape, l0_xyz.shape)

        # print(xyz.shape)
        # 向量化计算每个 batch 的每个 x, y, z 列的均值和标准差
        mean = torch.mean(xyz, dim=2, keepdim=True)
        std = torch.std(xyz, dim=2, keepdim=True)
        std[std == 0] = 1e-8  # 避免除零错误
        xyz = (xyz - mean) / std
        # print(l0_points.shape, l0_xyz.shape)

        # # 向量化计算每个 batch 的质心并移动到原点
        # centroid = torch.mean(xyz, dim=2, keepdim=True)
        # xyz = xyz - centroid

        l0_points = xyz
        l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        # return x, l4_points
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = Pointnet_2(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))