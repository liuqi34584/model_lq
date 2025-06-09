import torch
import os
import time
import numpy as np
from sklearn.neighbors import KDTree
from multiprocessing import freeze_support
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from model_pointnet_2.tools import save_point_cloud_with_labels, read_point_cloud_with_labels
from model_pointnet_2.tools import post_process_labels,visualize_labels

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def process_file_ply_or_pcd(input_file, npoints=25000, visualize=True):
    if os.path.exists(input_file):
        # 读取点云数据
        pcd = o3d.io.read_point_cloud(input_file)

        # 获取原始点云的点数
        original_point_count = len(pcd.points)
        if original_point_count == 0:
            print(f"文件 {input_file} 读取失败或点云为空，跳过后续处理。")
            return None, None

        # voxel_size = 0.005
        # pcd = pcd.voxel_down_sample(voxel_size)
        # # 获取下采样后点云的点数
        downsampled_point_count = len(pcd.points)
        # print(f"文件 {input_file} 原始点云点数: {original_point_count} 下采样点数：{downsampled_point_count}")

        # 假设这里获取这些变量的值
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros((len(points), 3))
        normals = np.asarray(pcd.normals)

        if len(normals) == 0:
            print(f"文件 {input_file} 没有法线数据，使用默认法线。")
            normals = points.copy()

    
        # # 打乱数据
        # indices = np.arange(len(points))
        # np.random.shuffle(indices)
        # points = points[indices]
        # colors = colors[indices]
        # normals = normals[indices]

        # 转换为numpy数组
        point_set = np.array(points, dtype=np.float32)
        colors_set = np.array(colors, dtype=np.int64)
        normals_set = np.array(normals, dtype=np.float32)

        # 重采样核心逻辑
        if len(point_set) == 0:
            print(f"[错误] 点云数据为空，无法采样！")
            return None, None
        if len(point_set) >= npoints:
            # 随机选择不重复的索引
            choice = np.random.choice(len(point_set), npoints, replace=False)
        else:
            # 允许重复采样
            choice = np.random.choice(len(point_set), npoints, replace=True)

        point_set = point_set[choice]
        colors_set = colors_set[choice]
        normals_set = normals_set[choice]

        point_set = np.array(point_set)
        colors_set = np.array(colors_set)
        normals_set = np.array(normals_set)

        print(f"文件 {input_file} 原始点云点数: {original_point_count} 下采样点数：{len(point_set)}")
        # print(point_set.shape, colors_set.shape, normals_set.shape)

        if visualize:
            vis0 = o3d.visualization.Visualizer()
            # 修改第一次可视化视窗的大小
            vis0.create_window(width=1000, height=800)
            vis0.add_geometry(pcd)
            ctr = vis0.get_view_control()
            opt = vis0.get_render_option()
            opt.point_size = 2.0
            vis0.run()
            vis0.destroy_window()

        return point_set, normals_set

    else:
        print(f"文件 {input_file} 不存在。")
        return None, None

def predict(points, colors, num_classes, model_path):
    # 加载显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.cuda.is_available()
    # device = "cpu"
    print("正在使用：{}".format(device))
    print("显卡名称:", torch.cuda.get_device_name())

    import model_PointTransformerV3.model as models  # 确保模型路径正确
    model = models.PointTransformerV3(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    model.eval()
    with torch.no_grad():
        points = torch.from_numpy(points).float().to(device).unsqueeze(0).transpose(2, 1)  # [1, 3, N]
        colors = torch.from_numpy(colors).float().to(device).unsqueeze(0).transpose(2, 1)  # [1, 3, N]

        pred = model(points, colors)
        pred = pred.reshape(-1, num_classes)
        pred_choice = pred.argmax(dim=1)  # [N]
    
    # 将 points 转换为 numpy 数组
    points = points.squeeze(0).transpose(0, 1).cpu().numpy()
    pred_labels = pred_choice.cpu().numpy()

    # # 统计每个类别的预测数量
    # class_counts = np.bincount(pred_labels)
    # for class_idx, count in enumerate(class_counts):
    #     print(f"类别 {class_idx} 的预测数量: {count}")
    
    return points, pred_labels


if __name__ == '__main__':
    freeze_support()

    # input_file = r"C:\code\dataset_AnShanBaiLai\AnShanBaiNai_ply\MB_2.ply"
    # input_file = r"C:\code\dataset_AnShanBaiLai\AnShanBaiNai_ply\MX_Obj.ply"
    # input_file = r"C:\code\dataset_AnShanBaiLai\AnShanBaiNai_ply\1.ply"
    # input_file = r"C:\code\dataset_AnShanBaiLai\AnShanBaiNai_ply\1_cut.ply"
    # input_file = r"C:\code\dataset_AnShanBaiLai\AnShanBaiNai_ply\0606\0606.ply"
    input_file = r"C:\code\dataset_AnShanBaiLai\AnShanBaiNai_ply\0606\0606_cut.ply"

    points, colors = process_file_ply_or_pcd(input_file, npoints=80000, visualize=False)
    points_scaled = points.copy()
    # points_scaled[:, 2] = points_scaled[:, 2] * 5

    # 记录开始时间
    start_time = time.time()
    points_scaled, pred_labels = predict(points_scaled, colors, num_classes = 5, model_path = r'./output/PointTransformerV3_output/PointTransformerV3_0609.pth')
    end_time = time.time()
    print(f"预测耗时: {end_time - start_time:.4f} 秒")

    # visualize_labels(points, pred_labels, [0,1,2,3,4,5,6])

    # 调用保存函数
    save_point_cloud_with_labels(points, pred_labels, r"C:\code\dataset_AnShanBaiLai\output\0606.ply")
    points, labels = read_point_cloud_with_labels(r"C:\code\dataset_AnShanBaiLai\output\0606.ply", visualize=True)
