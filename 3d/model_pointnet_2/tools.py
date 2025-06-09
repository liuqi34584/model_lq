import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
import matplotlib


# 该函数使用 RANSAC (RANdom SAmple Consensus) 算法对预测标签进行后处理，规整每个分割平面
def post_process_labels(points, pred_labels, target_label=2, num_iterations=10, distance_threshold=0.005):
    """
    使用RANSAC算法对预测标签进行后处理，识别并优化特定标签的平面结构。

    Args:
        points (np.ndarray): 点云数据，形状为 (N, 3)，表示N个点的三维坐标。
        pred_labels (np.ndarray): 预测标签数组，形状为 (N,)。
        target_label (int): 要优化的目标标签，默认为2。
        num_iterations (int): RANSAC算法的迭代次数，默认为1000。
        distance_threshold (float): 判断点是否属于平面的距离阈值，默认为0.01。

    Returns:
        np.ndarray: 优化后的标签数组，形状为 (N,)。
    """

    # 找到所有标签为 target_label 的点
    target_indices = np.where(pred_labels == target_label)[0]

    if len(target_indices) == 0:
        print(f"没有找到标签为 {target_label} 的点。")
        return pred_labels

    # 提取这些点的坐标
    target_points = points[target_indices]

    best_inliers = []
    best_model = None

    for _ in range(num_iterations):
        # 随机选择三个点来定义一个平面
        sample_indices = np.random.choice(target_points.shape[0], 3, replace=False)
        sample_points = target_points[sample_indices]

        # 使用这三个点来计算平面方程 Ax + By + Cz + D = 0
        A = sample_points[1] - sample_points[0]
        B = sample_points[2] - sample_points[0]
        normal = np.cross(A, B)
        D = -np.dot(normal, sample_points[0])

        # 计算所有点到平面的距离
        distances = np.abs(np.dot(target_points, normal) + D) / np.linalg.norm(normal)

        # 找到在距离阈值内的点（内点）
        inliers = target_points[distances < distance_threshold]

        # 如果当前内点数量比之前的最优解多，则更新最优解
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (normal, D)

    # 使用最优平面模型计算所有点到平面的距离
    normal, D = best_model
    distances = np.abs(np.dot(points, normal) + D) / np.linalg.norm(normal)

    # 将平面上的点标记为 target_label
    pred_labels[distances < distance_threshold] = target_label

    return pred_labels


# 将点云和标签保存到 PLY 文件中
def save_point_cloud_with_labels(points, labels, output_file):
    """
    保存点云和标签到 PLY 文件，直接构造数据结构避免临时文件

    Args:
        points (np.ndarray): 点云数据，形状为 (N, 3)
        labels (np.ndarray): 点云的标签，形状为 (N,)
        output_file (str): 输出 PLY 文件路径
    """
    import numpy as np
    from plyfile import PlyData, PlyElement

    # 生成随机颜色映射
    max_label = int(labels.max())
    colors = np.zeros((len(labels), 3))
    for i in range(max_label + 1):
        colors[labels == i] = np.random.rand(3)

    # 直接构造 PLY 顶点数据结构
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('label', 'i4')
    ]
    
    # 转换数据类型
    points_float32 = points.astype(np.float32)
    colors_uint8 = (colors * 255).astype(np.uint8)
    labels_int32 = labels.astype(np.int32)

    # 创建结构化数组
    vertex_data = np.core.records.fromarrays(
        [
            points_float32[:, 0],  # x
            points_float32[:, 1],  # y
            points_float32[:, 2],  # z
            colors_uint8[:, 0],    # red
            colors_uint8[:, 1],    # green
            colors_uint8[:, 2],    # blue
            labels_int32           # label
        ],
        dtype=vertex_dtype
    )

    # 创建并保存 PLY 文件
    PlyData([
        PlyElement.describe(vertex_data, 'vertex')
    ]).write(output_file)

    print(f"点云和标签已保存到 {output_file}")

def read_point_cloud_with_labels(file_path, visualize=False):
    """
    读取包含标签的PLY文件，支持Open3D可视化

    Args:
        file_path (str): PLY文件路径
        visualize (bool): 是否使用Open3D可视化点云

    Returns:
        points (np.ndarray): 点云坐标，形状为(N, 3)
        labels (np.ndarray): 点云标签，形状为(N,)
    """
    # 读取PLY文件
    ply_data = PlyData.read(file_path)
    
    # 验证文件结构
    if 'vertex' not in ply_data:
        raise ValueError("PLY文件缺少'vertex'元素")

    # 提取顶点数据
    vertices = ply_data['vertex'].data
    
    # 转换为numpy数组
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float32)
    labels = vertices['label'].astype(np.int32)
    
    # 可视化（如果需要）
    if visualize:

        # 固定色表
        default_colors = {
            -1: [1.0, 0, 0], 
            0: [0.5,0.5,0.5],    # 灰
            1: [0,1,0],    # 绿
            2: [0,0,1],    # 蓝
            3: [1,1,0],    # 黄
            4: [1,0,1],    # 紫
            5: [0,1,1],    # 青
            6: [1,0.5,0],  # 橙
            7: [0,0,0],    # 黑
        }

        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 如果没有颜色属性，则使用标签对应的颜色
        colors = np.zeros((len(points), 3))
        for label, color in default_colors.items():
            mask = labels == label
            colors[mask] = color
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
        # 设置可视化参数
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Point Cloud with Labels')
        vis.add_geometry(pcd)
        
        # # 添加坐标轴
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        # vis.add_geometry(mesh_frame)
        
        # 运行可视化
        vis.run()
        vis.destroy_window()

    return points, labels

def visualize_labels(points, pred_labels, show_labels):
    """
    显示带有标签颜色的点云，只显示指定标签的点。

    Args:
        points (np.ndarray): 点云坐标。
        pred_labels (np.ndarray): 点云预测标签。
        show_labels (list): 需要显示的标签列表，如 [0,1,2,3,4]。
    """

    # 固定色表（包含-1的新定义）
    default_colors = {
        -1: [1.0, 0, 0], 
        0: [0.5,0.5,0.5],    # 灰
        1: [0,1,0],    # 绿
        2: [0,0,1],    # 蓝
        3: [1,1,0],    # 黄
        4: [1,0,1],    # 紫
        5: [0,1,1],    # 青
        6: [1,0.5,0],  # 橙
        7: [0,0,0],    # 黑
    }

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 筛选指定标签的点
    valid_mask = np.isin(pred_labels, show_labels)
    filtered_points = points[valid_mask]
    filtered_labels = pred_labels[valid_mask]
    
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # 使用固定色卡分配颜色
    colors = np.zeros((len(filtered_points), 3))
    for label, color in default_colors.items():
        mask = filtered_labels == label
        colors[mask] = color

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 设置可视化参数
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud with Labels')
    vis.add_geometry(pcd)

    # 运行可视化
    vis.run()
    vis.destroy_window()

def region_growing_plane(points, pred_labels, target_label=0, distance_threshold=0.01, min_cluster_size=100):
    """
    区域生长平面分割，返回目标标签的主平面点索引。
    Args:
        points: (N,3) 点云坐标
        pred_labels: (N,) 标签
        target_label: 目标标签
        distance_threshold: 平面内点距离阈值
        min_cluster_size: 最小平面点数
    Returns:
        plane_indices: 匹配主平面点的索引数组
    """
    import open3d as o3d
    import numpy as np
    mask = (pred_labels == target_label)
    if np.sum(mask) < 3:
        print(f"标签{target_label}点数不足，无法拟合平面")
        return np.array([])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    labels = np.array(pcd.cluster_dbscan(eps=distance_threshold*2, min_points=10, print_progress=False))
    # 选最大簇
    best_plane_idx = []
    max_inliers = 0
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        cluster_mask = (labels == cluster_id)
        cluster_points = np.asarray(pcd.points)[cluster_mask]
        if len(cluster_points) < min_cluster_size:
            continue
        # 拟合平面
        plane_model, inliers = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cluster_points)).segment_plane(
            distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            # 转为原始点索引
            cluster_indices = np.where(mask)[0][cluster_mask]
            best_plane_idx = cluster_indices[inliers]
    if len(best_plane_idx) == 0:
        print("未找到主平面")
    else:
        print(f"主平面点数: {len(best_plane_idx)}")
    return np.array(best_plane_idx)