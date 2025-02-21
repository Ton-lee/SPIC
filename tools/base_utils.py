"""基本工具，包括图像压缩命令等"""
import collections
import os
default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
import cv2
import numpy as np
import time
import shutil
from openjpeg import encode, decode
from typing import Union
from shapely import Polygon, MultiPoint
from matplotlib.path import Path
from skimage import measure
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

temp_folder = "/home/Users/dqy/Projects/SPIC/temp"
suffix = {
    "jpeg": '.jpeg',
    "png": '.png',
    "webp": '.webp',
    "bpg": '.bpg',
    "flif": '.flif',
    "j2k": '.j2k',
    "cheng2020-anchor": '.bin',
    "cheng2020-attn": '.bin',
    "hific": '.tfci',
    "PostFilter-jpeg": ".log",
    "PostFilter-webp": ".log"
}


def compress_image(image: Union[np.ndarray, str],
                   method: str,
                   compress_factor: int,
                   compressed_path: str = None,
                   decompressed_path: str = None,
                   save_compressed: bool = False,
                   save_decompressed: bool = False,
                   return_array: bool = True):
    """图像压缩编码"""
    # 判断参数合法性
    if isinstance(image, str):
        assert os.path.isfile(image), f"Image not found: {image}"
        image: np.ndarray = cv2.imread(image)  # BGR image (H,W,3)
    assert (not save_compressed or compressed_path is not None
            and os.path.isdir(os.path.dirname(compressed_path))), "Compressed path not assigned!"
    assert (not save_decompressed or decompressed_path is not None
            and os.path.isdir(os.path.dirname(decompressed_path))), "Decompressed path not assigned!"
    # 获取临时路径
    name_stamp = method
    time_stamp = time.time()
    process_stamp = os.getpid()
    original_save_name = f"{name_stamp}-{process_stamp}#{time_stamp}_original.png"
    compressed_save_name = f"{name_stamp}-{process_stamp}#{time_stamp}_compressed{suffix[method.lower()]}"
    decompressed_save_name = f"{name_stamp}-{process_stamp}#{time_stamp}_decompressed.png"
    temp_abs_path_original = os.path.join(temp_folder, original_save_name)
    temp_abs_path_compressed = os.path.join(temp_folder, compressed_save_name)
    temp_abs_path_decompressed = os.path.join(temp_folder, decompressed_save_name)
    image_compressed = None  # RGB image
    # 执行压缩
    if method.lower() in ["jpeg", "png", "webp"]:
        if method.lower() == 'jpeg':  # JPEG 压缩
            assert 0 <= compress_factor <= 100, (f"Compression factor for JPEG must be 0-100 "
                                                 f"(higher for better, got {compress_factor})")
            # 编码
            cv2.imwrite(temp_abs_path_compressed, image, [int(cv2.IMWRITE_JPEG_QUALITY), compress_factor])
        if method.lower() == 'png':  # PNG
            assert 0 <= compress_factor <= 9, (f"Compression factor for PNG must be 0-9 "
                                               f"(lower for better, got {compress_factor})")
            # 编码
            cv2.imwrite(temp_abs_path_compressed, image, [int(cv2.IMWRITE_PNG_COMPRESSION), compress_factor])
        if method.lower() == 'webp':  # WEBP
            assert 0 <= compress_factor <= 100, (f"Compression factor for WEBP must be 0-100 "
                                                 f"(higher for better, got {compress_factor})")
            cv2.imwrite(temp_abs_path_compressed, image, [int(cv2.IMWRITE_WEBP_QUALITY), compress_factor])
        # 解码
        image_compressed = cv2.imread(temp_abs_path_compressed)[:, :, ::-1]
        cv2.imwrite(temp_abs_path_decompressed, image_compressed[:, :, ::-1])
    if method.lower() == 'bpg':  # BPG
        assert 0 <= compress_factor <= 51, (f"Compression factor for BPG must be 0-51 "
                                            f"(lower for better, got {compress_factor})")
        cv2.imwrite(temp_abs_path_original, image)
        command = (f"bash '/home/Users/dqy/scripts/compress_BPG.sh' "
                   f"'{temp_abs_path_original}' "
                   f"{compress_factor} "
                   f"'{temp_abs_path_compressed}' "
                   f"'{temp_abs_path_decompressed}'")
        os.system(command)
        image_compressed = cv2.imread(temp_abs_path_decompressed)[:, :, ::-1]
    if method.lower() == 'flif':  # FLIF
        assert 0 <= compress_factor <= 100, (f"Compression factor for FLIF must be 0-100 "
                                             f"(higher for better, got {compress_factor})")
        cv2.imwrite(temp_abs_path_original, image)
        command = (f"bash '/home/Users/dqy/scripts/compress_FLIF.sh' "
                   f"'{temp_abs_path_original}' "
                   f"{compress_factor} "
                   f"'{temp_abs_path_compressed}' "
                   f"'{temp_abs_path_decompressed}'")
        os.system(command)
        image_compressed = cv2.imread(temp_abs_path_decompressed)[:, :, ::-1]
    if method.lower() == 'j2k':  # J2K
        assert 1 <= compress_factor, (f"Compression ratio for J2K must be 1- "
                                      f"(1 for best, got {compress_factor})")
        codes = encode(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), compression_ratios=[compress_factor])
        with open(temp_abs_path_compressed, 'wb') as f:
            f.write(codes)
        image_compressed = decode(codes)  # RGB
        cv2.imwrite(temp_abs_path_decompressed, cv2.cvtColor(image_compressed, cv2.COLOR_RGB2BGR))
    # 处理保存与返回信息
    if save_compressed:
        shutil.copy(temp_abs_path_compressed, compressed_path)
    if save_decompressed:
        shutil.copy(temp_abs_path_decompressed, decompressed_path)
    # 清除临时文件
    if os.path.exists(temp_abs_path_original):
        os.remove(temp_abs_path_original)
    if os.path.exists(temp_abs_path_compressed):
        os.remove(temp_abs_path_compressed)
    if os.path.exists(temp_abs_path_decompressed):
        os.remove(temp_abs_path_decompressed)
    if return_array:
        return image_compressed
    else:
        return None


def get_mIoU(label_gt: np.ndarray, label_pred: np.ndarray, labels=None, ignore_label=19):
    """
    计算两个语义分割图的 mIoU
    """
    if labels is None:
        labels = list(set(label_gt.flatten().tolist()) | set(label_pred.flatten().tolist()))
    IoU = []
    for label_index in labels:
        union = np.bitwise_or(label_gt == label_index, label_pred == label_index) * (label_gt != ignore_label)
        intersection = np.bitwise_and(label_gt == label_index, label_pred == label_index) * (label_gt != ignore_label)
        union = np.sum(union)
        intersection = np.sum(intersection)
        if union == 0:
            continue
        IoU.append(intersection / union)
    if len(IoU) == 0:
        return 0.0
    return np.mean(IoU)


def pad_same(image: np.ndarray, pad_width=1) -> np.ndarray:
    """以与边界值相同的值对图像进行扩展"""
    map_pad = np.pad(image, pad_width=pad_width, mode='edge')
    return map_pad


def ssm2boundary_seed(ssm: np.ndarray) -> np.ndarray:
    """
    通过种子生长算法获取语义分割图的边界线
    """
    # 具体算法为，初始化边界蒙版为 0，采用区域生长的方法，在边界蒙版上以某一非边界点为种子向8邻域生长。生长规则为：
    # 1. 若种子为非边界点，且与其根类别相同，则将8邻域设置为种子；
    # 2. 若种子为边界点，则该种子不生长；
    # 3. 若种子为非边界点，但与其根类别不同，则将该种子修正为边界点。
    pass


def ssm2boundary(ssm: np.ndarray, semantic_orders=None, ignore_label=19) -> (np.ndarray, np.ndarray):
    """由语义分割图获取边界，采用迭代式边界检测的方法"""
    if semantic_orders is None:
        semantic_orders = set(ssm.flatten().tolist())
    # 具体算法为：遍历每一个标签类别并设置边界，设置边界时考虑已经存在的边界的影响
    assert len(ssm.shape) == 2, "dim of ssm.shape must be 2"
    H, W = ssm.shape
    boundary = np.zeros_like(ssm)
    for semantic_label in semantic_orders:  # 考察一个区域类别
        if semantic_label not in ssm:  # 只考虑存在于语义分割图中的标签
            continue
        inner_map = ssm == semantic_label  # 当前考察标签对应区域
        outer_map = ssm != semantic_label  # 考察标签以外的区域
        # 找到考察区域外，与考察标签相邻的像素点
        inner_map_pad = pad_same(inner_map, 1)
        boundary_pad = pad_same(boundary, 1, )
        shifts_x, shifts_y = (0, 0, 0, 1, 1, 2, 2, 2), (0, 1, 2, 0, 2, 0, 1, 2)
        candidate_map = np.zeros_like(ssm)  # 区域外与考察标签相邻的像素点为 1
        for shift_x, shift_y in zip(shifts_x, shifts_y):
            neighbor_map = inner_map_pad[shift_y: shift_y + H, shift_x: shift_x + W]
            neighbor_boundary = boundary_pad[shift_y: shift_y + H, shift_x: shift_x + W]
            candidate_map += outer_map * neighbor_map * (1 - neighbor_boundary) * (1 - boundary)  # 区域外的非边界点，且邻点为区域内的非边界点
        boundary[candidate_map > 0] = 1
    boundary[:, [0, -1]] = np.logical_or(boundary[:, [0, -1]], ssm[:, [0, -1]] != ignore_label)
    boundary[[0, -1], :] = np.logical_or(boundary[[0, -1], :], ssm[[0, -1], :] != ignore_label)
    return boundary


def get_inner_points(points):
    """获得多边形内部（包括边界）的点"""
    p = Polygon(points)
    xmin, ymin, xmax, ymax = p.bounds
    x = np.arange(np.floor(xmin), np.ceil(xmax) + 1)
    y = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
    grids = MultiPoint(np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))]))
    result: MultiPoint = grids.intersection(p)
    return [(int(p.x), int(p.y)) for p in result.geoms]


def get_inner_mask(polygon):
    """获得多边形内部点的蒙版"""
    min_l, min_c = np.min(polygon, axis=0)
    max_l, max_c = np.max(polygon, axis=0)
    h, w = max_l - min_l + 1, max_c - min_c + 1
    local_mask = np.zeros([h, w])
    path = Path(polygon)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    points = np.vstack((y.flatten(), x.flatten())).T + np.array([min_l, min_c])
    mask = path.contains_points(points, radius=0.5).reshape((h, w))  # 设置扩展半径使其包含路径上的点
    return mask, min_l, min_c


def neighbored(p1, p2, neighbor=8):
    """两个点是否相邻或相同"""
    if neighbor == 8:
        return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) <= 1
    else:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) <= 1


def get_cityscapes_ssm_paths(phase="val"):
    """获取 Cityscapes 数据集 val 子集的所有语义分割图的路径"""
    root_folder = f"/home/Users/dqy/Dataset/Cityscapes/gtFine/{phase}/"
    # 获取文件夹中所有 labelTrainIds.png 为后缀的文件绝对路径
    all_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith("labelTrainIds.png"):
                all_paths.append(os.path.join(root, file))
    return all_paths


def get_ade20k_ssm_paths(phase="val"):
    """获取 ADE20k 数据集 val 子集的所有语义分割图的路径"""
    root_path = "/home/Users/dqy/Dataset/ADE20K/ADEChallengeData2016/annotations/validation/" if phase == "val" else "/home/Users/dqy/Dataset/ADE20K/ADEChallengeData2016/annotations/training/"
    files = sorted(os.listdir(root_path))
    return [os.path.join(root_path, file) for file in files]


def parse_cityscapes_labels():
    cityscapes_colors_map = {
        0: (128, 64, 128),  # road
        1: (244, 35, 232),  # sidewalk
        2: (70, 70, 70),  # building
        3: (102, 102, 156),  # wall
        4: (190, 153, 153),  # fence
        5: (153, 153, 153),  # pole
        6: (250, 170, 30),  # traffic light
        7: (220, 220, 0),  # traffic sign
        8: (107, 142, 35),  # vegetation
        9: (152, 251, 152),  # terrain
        10: (70, 130, 180),  # sky
        11: (220, 20, 60),  # person
        12: (255, 0, 0),  # rider
        13: (0, 0, 142),  # car
        14: (0, 0, 70),  # truck
        15: (0, 60, 100),  # bus
        16: (0, 80, 100),  # train
        17: (0, 0, 230),  # motorcycle
        18: (119, 11, 32)  # bicycle
    }
    color2label = {}
    for key, val in color2label.items():
        color2label[val] = key
    cityscapes_semantic_orders = [11, 12, 18, 17, 13, 14, 15, 16, 6, 7, 4, 5, 8, 0, 1, 9, 2, 3, 10]  # 遍历类别的顺序，高优先级的顺序倾向于保留区域内的所有像素点，保证区域不收缩（如路灯）
    return cityscapes_colors_map, color2label, cityscapes_semantic_orders

def parse_ade20k_labels():
    """解析 ADE20k 的标签结构，包括获取类别到颜色、颜色到类别字典和类别优先级列表"""
    label2color = {}
    color2label = {}
    semantic_orders = [i+1 for i in range(150)]
    return label2color, color2label, semantic_orders


cityscapes_label2color, cityscapes_color2label, cityscapes_semantic_orders = parse_cityscapes_labels()
ade20k_label2color, ade20k_color2label, ade20k_semantic_orders = parse_ade20k_labels()


def get_ssm_colored_cityscapes(ssm):
    """对语义分割图上色"""
    H, W = ssm.shape[:2]
    color_image = np.zeros((H, W, 3), dtype=np.uint8)
    for key, color in cityscapes_label2color.items():
        mask = ssm == key
        color_image[mask] = color
    return color_image


def get_semantic_orders(ssm: np.ndarray, ignore_label=None):
    """根据像素数占比决定语义优先级，优先保证小目标的完整性"""
    if ignore_label is None:
        ignore_label = -1
    counter = collections.Counter(ssm.flatten().tolist())
    frequencies = counter.most_common()
    semantic_order = [frequency[0] for frequency in frequencies if frequency[0] != ignore_label]
    return semantic_order[::-1]  # 低频次标签具有更高优先级


def cost_driven_classification(X, C, num_classes, learning_rate=0.01, max_iter=1000, tol=1e-6):
    """
    最小化分类代价的线性分类算法。

    Parameters:
    - X: np.ndarray, 数据点的二维坐标 (N x 2)。
    - C: np.ndarray, 分类代价矩阵 (N x num_classes)，C[i, k] 表示点 i 分类到类别 k 的代价。
    - num_classes: int, 类别数量。
    - learning_rate: float, 梯度下降学习率。
    - max_iter: int, 最大迭代次数。
    - tol: float, 收敛判定阈值。

    Returns:
    - weights: np.ndarray, 分类界面参数 (num_classes x 3)。
    - history: list, 每次迭代的平均分类代价。
    - predicted_classes: np.ndarray, 每个样本被分到的类别
    """
    # 初始化参数
    N, d = X.shape  # 数据点数量和特征维度
    weights = np.random.randn(num_classes, d + 1)  # 每个类别一个线性分类器，包含偏置项
    X_h = np.hstack([X, np.ones((N, 1))])  # 扩展输入数据为齐次坐标
    history = []
    predicted_classes = np.zeros((N,))
    for iteration in range(max_iter):
        # 分类阶段
        scores = X_h @ weights.T  # 计算每个点对每个类别的分数 (N x num_classes)
        probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        predicted_classes = np.argmax(probs, axis=1)  # 每个点的预测类别 (N,)

        # 计算平均分类代价
        avg_cost = np.mean(C[np.arange(N), predicted_classes] * probs[np.arange(N), predicted_classes])
        history.append(avg_cost)

        # 判断收敛
        if iteration > 0 and abs(history[-1] - history[-2]) < tol:
            print(f"Converged at iteration {iteration}.")
            break

        # 梯度计算与参数更新
        gradients = np.zeros_like(weights)  # 初始化梯度 (num_classes x (d+1))
        for k in range(num_classes):
            indices = (predicted_classes == k)  # 找到被分类为类别 k 的数据点
            gradient_k = -np.sum(X_h[indices] * C[indices, k:k+1], axis=0)  # 累加误分类的点，以误分类权重加权
            gradients[k] = gradient_k

        # 梯度下降更新
        weights -= learning_rate * gradients

    return weights, history, predicted_classes


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def compute_loss(X, C, W, b):
    # 计算线性输出
    Y = np.dot(X, W) + b  # N x k
    # 计算softmax
    softmax_output = softmax(Y)  # N x k
    # 计算交叉熵损失
    loss = -np.sum(C * np.log(softmax_output)) / X.shape[0]
    return loss


def compute_gradients(X, C, W, b):
    # 计算线性输出
    Y = np.dot(X, W) + b  # N x k
    # 计算softmax
    softmax_output = softmax(Y)  # N x k
    # 计算梯度
    grad_W = np.dot(X.T, (softmax_output - C)) / X.shape[0]  # 3 x k
    grad_b = np.sum(softmax_output - C, axis=0) / X.shape[0]  # k
    return grad_W, grad_b


def train_classifier_naive(X, C, learning_rate=0.01, epochs=100):
    N, D = X.shape
    k = C.shape[1]

    # 初始化参数
    W = np.random.randn(D, k) * 0.01
    b = np.zeros(k)

    for epoch in range(epochs):
        # 计算损失
        loss = compute_loss(X, C, W, b)
        # 计算梯度
        grad_W, grad_b = compute_gradients(X, C, W, b)

        # 更新权重和偏置
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b

        # 每10步输出一次损失
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return W, b


# 定义一层的神经网络模型
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)


# 训练函数
def train_classifier(X, C, model, criterion, optimizer, epochs=100, batch_size=32):
    N = X.shape[0]
    for epoch in range(epochs):
        permutation = torch.randperm(N)

        for i in range(0, N, batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_C = X[indices], C[indices]

            # 正向传播
            outputs = model(batch_X)
            # loss = criterion(outputs, batch_C)
            loss = -torch.mean(torch.log(outputs + 1e-4) * (1 - batch_C))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每10步输出一次损失
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    weights = model.fc.weight.data.numpy()
    bias = model.fc.bias.data.numpy()
    return weights, bias


def train_weighted_svm(X, y, sample_weights, kernel='linear', C=1.0):
    """
    训练一个带有样本权重的SVM模型，并返回模型的准确度、训练后的权重和偏置。

    参数:
    - X: 特征矩阵，形状为 (N, D)，N 是样本数，D 是特征数
    - y: 标签向量，形状为 (N,)，包含每个样本的分类标签
    - sample_weights: 样本权重，形状为 (N,)，每个样本对应的权重
    - kernel: SVM的核函数，默认是'linear'（线性核）
    - C: 正则化参数，默认是1.0

    返回:
    - accuracy: 训练集的准确度
    - weights: 训练后的权重
    - bias: 训练后的偏置
    """

    # 创建并训练SVM模型
    svm = SVC(kernel=kernel, C=C, decision_function_shape='ovr')
    svm.fit(X, y, sample_weight=sample_weights)  # 使用样本权重训练模型

    # 在训练集上进行预测
    y_pred = svm.predict(X)

    # 计算准确度
    accuracy = accuracy_score(y, y_pred)

    # 获取训练后的权重和偏置
    weights = svm.coef_
    bias = svm.intercept_

    return weights, bias, y_pred


def get_center(positions, colors, k):
    """找到聚类中心，使像素点到与其最近的中心的 RGB 值的 MSE 平均值最小"""
    N, D = positions.shape
    # STEP 1: 对像素点颜色进行 k-means 聚类，找到 k 个颜色中心；
    # 进行 k-means 聚类
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(colors)
    # 获取 k-means 聚类结果
    color_centers = kmeans.cluster_centers_
    # STEP 2: 计算各个像素点颜色与聚类中心的 MSE
    mse_matrix = np.zeros((N, k))
    for i, color in enumerate(colors):
        mse_matrix[i] = np.mean((color_centers - colors[i, :]) ** 2, axis=-1)
    gt = np.argmin(mse_matrix, axis=1)
    sample_weights = mse_matrix.max(axis=1) - mse_matrix[np.arange(N), gt]
    # STEP 3: 带权的线性分类
    # SUBSTEP 1: 简单算法
    # weights, history, classes = cost_driven_classification(positions, mse_matrix, k)
    # Y = np.dot(positions, weights) + bias
    # classes = np.argmax(Y, axis=1)
    # SUBSTEP 2: 简单线性分类器
    # weights, bias = train_classifier_naive(positions, mse_matrix)
    # Y = np.dot(positions, weights) + bias
    # classes = np.argmax(Y, axis=1)
    # SUBSTEP 3: 单层感知机
    # model = LinearClassifier(D, k)  # 输入维度为3，输出维度为k（类别数）
    # criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失（适用于多标签问题）
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    # weights, bias = train_classifier(torch.Tensor(positions), torch.Tensor(mse_matrix), model, criterion, optimizer, epochs=100)
    # with torch.no_grad():
    #     Y = model(torch.Tensor(positions)).numpy()
    # classes = np.argmax(Y, axis=1)
    # SUBSTEP 4: SVM
    weights, bias, classes = train_weighted_svm(positions, gt, sample_weights)
    # STEP 4: 将权重连通偏置返回
    mse = mse_matrix[np.arange(N), classes].mean()
    weights = np.concatenate([weights, bias[:, None]], axis=1)
    return color_centers, weights, classes


def sample_color(image, boundary, alpha=1.0):
    """
    从给定的图像中采样像素点作为图像的稀疏表征
    :param image: RGB 图像，(H,W,3)，归一化到0~1
    :param boundary: 边界图，(H,W)，0 表示非边界、1表示边界像素点，边界四邻域连通
    :param alpha: 码率损失与 MSE 损失的权重，优化目标为码率损失+alpha*MSE，更高的alpha表示更高的采样细节
    :return: 采样点坐标和颜色
    """
    # STEP 1: 获取图像区域连通性
    H, W = image.shape[:2]
    # 根据边界像素点获取图像区域连通性
    labeled_image, num_labels = measure.label(1 - boundary, connectivity=1, return_num=True)
    # STEP 2: 对每个区域求平均 RGB 作为初始颜色采样点（k-means: k=1）
    # 考察每个连通区域，以像素平均 RGB 值作为初始颜色采样点
    sample_points = []
    for label in range(1, num_labels + 1):
        mask = labeled_image == label
        region_points = np.argwhere(mask)
        weight = (0, 0, 0)
        region_color = np.mean(image[region_points[:, 0], region_points[:, 1], :], axis=0)
        sample_points.append([(weight, tuple(region_color))])
    # STEP 3: 根据采样点计算数据量和误差
    # 计算采样点的数据量：坐标通过定长量化进行编码，颜色采用 24 bit 进行编码
    data_size = num_labels * (np.ceil(np.log2(H)) + np.ceil(np.log2(W)) + 24)
    # 分别计算各个label对应区域像素点与采样点之间的 MSE
    mse_values = []
    for label in range(1, num_labels + 1):
        mask = labeled_image == label
        mse_values.append(
            np.mean(np.square(
                image[mask] - np.array(sample_points[label - 1][0][1])[None])))
    # 对数据量和平均 mse 进行加权，获得总损失
    loss = data_size + alpha * sum(mse_values) / num_labels
    tried = [False] * num_labels  # 记录
    k = [1] * num_labels  # 每个区域的采样点个数
    # STEP 4: 按 mse 降序排序，依次考察每个连通区域，计算增加一个采样点后的损失
    while not all(tried):  # 直到尝试了所有的连通区域
        # 按照 mse 由高到低对各个区域降序排列，获取排序后的区域序号
        sorted_indices = np.argsort(mse_values)[::-1]
        sorted_indices = [sorted_index + 1 for sorted_index in sorted_indices]  # 按优先级由高到低排列的区域序号（1到N）
        # 依次考察各个区域
        for label in sorted_indices:
            # 获取该区域的像素RGB值
            mask = labeled_image == label
            region_points = np.argwhere(mask) / np.array((H, W))  # 将坐标也归一化到 0~1
            region_pixels = image[mask]
            # 找到分类界面和颜色
            color_centers, weights, classes = get_center(region_points, region_pixels, k[label - 1] + 1)
            # 计算区域内各个像素点与距离最近的聚类中心的颜色 MSE
            class_colors = color_centers[classes, :]
            mse_value = np.mean(np.square(region_pixels - class_colors))
            # 计算 loss 增加量
            delta_mse = mse_value - mse_values[label - 1]
            delta_data_amount = np.ceil(np.log2(H)) + np.ceil(np.log2(W)) + 24
            delta_loss = delta_data_amount + alpha * delta_mse / num_labels
            if delta_loss < 0:  # loss 降低
                # 更新区域 MSE
                mse_values[label - 1] = mse_value
                # 更新区域的聚类点数
                k[label - 1] += 1
                # 更新区域的采样点
                sample_points[label - 1] = [(weights[i], color_centers[i])
                                            for i in range(k[label - 1])]
                tried = [False] * num_labels
                break
            else:  # loss 没有有效降低，考察下一个重要的区域
                tried[label - 1] = True
    return sample_points


def test_color_sample():
    image = cv2.imread("/home/Users/dqy/Dataset/Cityscapes/leftImg8bit_merged(512x256)/val/"
                       "frankfurt_000000_000294_leftImg8bit.png")[:, :, ::-1] / 255
    boundary = cv2.imread("/home/Users/dqy/Dataset/Cityscapes/boundary_compressed/val"
                          "/ori/frankfurt_000000_000294_gtFine_instanceBoundary.png")[:, :, 0]
    boundary = 1 - boundary.astype('int') / 255
    sample_points = sample_color(image, boundary, alpha=1)
    print(sample_points)


if __name__ == '__main__':
    test_color_sample()
