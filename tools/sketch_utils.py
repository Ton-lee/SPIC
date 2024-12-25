import os
import time
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from scipy.signal import convolve2d as conv
import pickle
import json
from tools.utils import dec2bin, bitstring2bytes, bytes2bitstring, branch_count_2path
from tools.base_utils import cityscapes_semantic_orders, get_inner_points, neighbored, get_inner_mask
from typing import List
from skimage import measure
from tools.utils import save_graph_compressed, rearrange_graph
import os
import imageio
from tools.base_utils import ssm2boundary, cityscapes_label2color, get_mIoU, get_ssm_colored_cityscapes
from shapely import Point
from scipy.ndimage import distance_transform_edt

def calculate_ratio(edge_map, search_range=2, threshold_pixel=None):
    """
    计算各个像素点边缘强度高于周边像素的比例
    :param edge_map: 边缘强度图
    :param search_range: 搜索邻域的范围
    :param threshold_pixel: 强度低于阈值的像素点直接置为0
    """
    edge_map = edge_map.astype('int16')
    h, w = edge_map.shape
    if search_range != 2:
        # print('use simple method...')
        threshold_pixel = np.mean(edge_map.flatten()) if threshold_pixel is None else threshold_pixel
        ratio_pixel_wise = np.zeros([h, w])
        for i in range(h):
            for j in range(w):
                if edge_map[i, j] <= threshold_pixel:
                    continue
                min_i = max(0, i - search_range)
                max_i = min(h - 1, i + search_range)
                min_j = max(0, j - search_range)
                max_j = min(w - 1, j + search_range)
                active_matrix = (edge_map[i, j] - edge_map[min_i:max_i + 1, min_j:max_j + 1] > 0)
                ratio_pixel_wise[i, j] = np.sum(active_matrix) / ((max_i - min_i + 1) * (max_j - min_j + 1) - 1)
        return ratio_pixel_wise
    shift_list = [[-2, i] for i in range(-2, 3)] \
                 + [[-1, i] for i in range(-2, 3)] \
                 + [[0, -2], [0, -1]]
    ratio_matrix_full = np.zeros([24, h, w])
    sub_index_i = [[2, h], [1, h], [0, h], [0, h - 1], [0, h - 2]]
    sub_index_j = [[2, w], [1, w], [0, w], [0, w - 1], [0, w - 2]]

    for i in range(12):
        shift_i, shift_j = shift_list[i]
        error = edge_map[
                sub_index_i[shift_i + 2][0]: sub_index_i[shift_i + 2][1],
                sub_index_j[shift_j + 2][0]: sub_index_j[shift_j + 2][1]] - \
                edge_map[
                sub_index_i[-shift_i + 2][0]: sub_index_i[-shift_i + 2][1],
                sub_index_j[-shift_j + 2][0]: sub_index_j[-shift_j + 2][1]]
        ratio_matrix_full[i, sub_index_i[shift_i + 2][0]: sub_index_i[shift_i + 2][1],
        sub_index_j[shift_j + 2][0]: sub_index_j[shift_j + 2][1]] = error
        ratio_matrix_full[23 - i, sub_index_i[-shift_i + 2][0]: sub_index_i[-shift_i + 2][1],
        sub_index_j[-shift_j + 2][0]: sub_index_j[-shift_j + 2][1]] = -error
    neighbor_count = np.ones([h, w]) * 14
    neighbor_count[1:-1, 1:-1] = 19
    neighbor_count[2:-2, 2:-2] = 24
    neighbor_count[[0, 0, -1, -1], [0, -1, 0, -1]] = 8
    neighbor_count[[0, 0, 1, 1, -2, -2, -1, -1], [1, -2, 0, -1, 0, -1, 1, -2]] = 11
    neighbor_count[[1, 1, -2, -2], [1, -2, 1, -2]] = 15
    ratio_matrix = np.sum(ratio_matrix_full > 0, axis=0) / neighbor_count  # 比周边像素高的比例
    return ratio_matrix.astype('float16'), neighbor_count


def mark(valid, marked, pos, size=None, label=1, neighbor_range=8):
    """
标记指定位置及其邻域中的点，直到其所在连通分支全部被标记
    :param valid: 是否为有效点。指定点邻域的所有有效点均需被标记
    :param marked: 是否已经被标记的标志，避免重复标记
    :param pos: 需要进行标记的点的位置，二元数组或列表
    :param size: 输入矩阵的尺寸，表示行数与列数
    :param label: 需要标记的标签
    :param neighbor_range: 连通分支按照8邻域或4邻域
    """
    if size is None:
        size = valid.shape
    h, w = size
    l, c = pos
    marked[l, c] = label
    for di in [-1, 0, 1]:
        if (l == 0 and di == -1) or (l == h - 1 and di == 1):  # 越界处理
            continue
        for dj in [-1, 0, 1]:
            if (c == 0 and dj == -1) or (c == w - 1 and dj == 1):  # 越界处理
                continue
            if di == 0 and dj == 0:  # 不包含中心点
                continue
            if neighbor_range == 4 and di * dj != 0:
                continue
            if valid[l + di, c + dj] and not marked[l + di, c + dj]:
                mark(valid, marked, [l + di, c + dj], size, label, neighbor_range)


def branch_count_seed(valid, neighbor_range=8):  # 基于种子生长的连通数计算
    """
按照种子生长的方法计算连通分支数
    :param valid: 有效点，需要计入连通分支
    :param neighbor_range: 按照8邻域或4邻域
    :return: 连通分支数和各个点所属的分支序号
    """
    raise NotImplementedError("Not suggested because of low efficiency. 'skimage.measure.label' is suggested.")
    count = 0
    marked = np.zeros_like(valid)
    loc = np.array(np.where(valid == 1)).transpose()
    for valid_index in range(len(loc)):
        if not marked[loc[valid_index, 0], loc[valid_index, 1]]:
            count += 1
            mark(valid, marked, loc[valid_index], label=count, neighbor_range=neighbor_range)
    return count, marked


def branch_count_2path(valid, neighbor_range=8, return_branch_map=True):  # 基于两分支的连通数计算
    """
基于两分支方法计算连通分支数。
    :param valid: 有效点，需要计算连通分支
    :param neighbor_range: 4邻域或8邻域搜索
    :param return_branch_map: 是否需要返回分支序号标记
    :return: 连通分支数，或连通分支数和分支序号标记
    """
    raise NotImplementedError("Not suggested because of low efficiency. 'skimage.measure.label' is suggested.")
    edge = valid.copy().astype('int')
    idx = 2  # 从2开始标注，避免与边缘的标记1混淆
    equal_pairs = []
    h, w = valid.shape
    for i in range(h):
        min_i = i - 1 if i > 0 else 0
        for j in range(w):
            if edge[i, j] == 0 or edge[i, j] == -1:  # 非边缘部分
                continue
            min_j = j - 1 if j > 0 else 0
            max_j = j if j == w - 1 else j + 1
            neighbor = edge[min_i: i + 1, min_j:max_j + 1] if neighbor_range == 8 else edge[
                [min_i, i], [j, min_j]]  # 8邻域，考察其上方的6个点；4邻域，考察其上方和左方的两个点
            ids_previous = list(set(neighbor[neighbor > 1]))  # 上方标记过的序号，取>1的数值
            n_previous = ids_previous.__len__()
            if n_previous == 0:
                edge[i, j] = idx
                idx += 1  # continue here
                continue
            if n_previous == 1:
                edge[i, j] = ids_previous[0]  # continue here
                continue
            if n_previous > 3:
                continue
            if n_previous > 1:
                edge[i, j] = min(ids_previous)
                if ids_previous not in equal_pairs:
                    if n_previous == 3:
                        equal_pairs.append(ids_previous[:2])
                        equal_pairs.append(ids_previous[1:])
                        continue
                    else:
                        equal_pairs.append(ids_previous)  # continue here
                        continue
    eq_marks, eq_label, eq_count = np.zeros(idx - 2).astype('uint8'), 1, 0
    # 无等价类，则直接获得分支数和分支图
    if equal_pairs.__len__() == 0:
        if not return_branch_map:
            return idx - 2
        edge[edge > 0] -= 1
        edge[edge == -1] = 0
        return idx - 2, edge.astype('uint8')
    # 有等价类，需要进行等价类合并
    for i in range(len(equal_pairs)):
        i1, i2 = equal_pairs[i]
        i1, i2 = i1 - 2, i2 - 2
        m1, m2 = eq_marks[i1], eq_marks[i2]
        if m1 == 0:
            if m2 == 0:
                eq_marks[[i1, i2]] = eq_label
                eq_label += 1  # 等价类标签+1
                eq_count += 1  # 等价类数量+1
            else:
                eq_marks[i1] = m2
                eq_count += 1  # 等价类数量+1
        else:
            if m2 == 0:
                eq_marks[i2] = m1
                eq_count += 1  # 等价类数量+1
            else:
                if m1 != m2:
                    eq_marks[eq_marks == m2] = m1  # 等价类合并
                    eq_count += 1  # 等价类数量+1
    if not return_branch_map:
        return idx - 2 - eq_count
    # 根据合并后的等价类获得分支图
    label = 1
    branch_id_ori, branch_id_pre = [], []
    edge_merged = np.zeros([h, w], dtype='uint8')
    for i in range(idx - 2):
        if eq_marks[i] == 0:  # 没有标注等价类，单独执行分支划分
            edge_merged[edge == i + 2] = label
            label += 1
            continue
        if eq_marks[i] not in branch_id_ori:
            branch_id_ori.append(eq_marks[i])
            branch_id_pre.append(label)
            edge_merged[edge == i + 2] = label
            label += 1
        else:
            edge_merged[edge == i + 2] = branch_id_pre[branch_id_ori.index(eq_marks[i])]
    return idx - 2 - eq_count, edge_merged


def bad_connect(edge, i, j):  # 点附近有三个及以上边缘点，且加入该点后连接数不变，则为无效边缘点
    """
是否是无效边缘点，即加入该点后邻域内的连通分支数是否不减少
    :param edge: 局部的边缘标记，即连通分支计算的区域
    :param i: 目标位置所在行
    :param j: 目标位置所在列
    :return: 是否为无效边缘点
    """
    # edge_temp = edge.copy()
    h, w = edge.shape
    min_i = max(0, i - 1)
    max_i = min(h - 1, i + 1)
    min_j = max(0, j - 1)
    max_j = min(w - 1, j + 1)
    if np.sum(edge[min_i:max_i + 1, min_j:max_j + 1]) >= 3:
        # edge_temp[i, j] = 1
        # branch_count_ori, branch_map = branch_count_2path(edge)
        branch_map, branch_count_ori = measure.label(edge, background=0, return_num=True, connectivity=2)
        neighbor_branch = branch_map[min_i:max_i + 1, min_j:max_j + 1]
        # branch_count_now, _ = branch_count_2path(edge_temp)
        # if branch_count_ori == branch_count_now:
        if set(neighbor_branch.flatten()).__len__() < 3:
            return True
    return False


def get_index(line, column, h, w, neighbor_range):
    """
获取邻域范围
    """
    # min_i = max(0, line - neighbor_range)
    min_i = line - neighbor_range if line > neighbor_range else 0
    # max_i = min(h - 1, line + neighbor_range)
    max_i = line + neighbor_range if line < h - neighbor_range else h - 1
    # min_j = max(0, column - neighbor_range)
    min_j = column - neighbor_range if column > neighbor_range else 0
    # max_j = min(w - 1, column + neighbor_range)
    max_j = column + neighbor_range if column < w - neighbor_range else w - 1
    return min_i, max_i, min_j, max_j


def norm_index(i, h):
    """
合法化所给位置
    """
    i = 0 if i < 0 else h - 1 if i >= h else i
    return i


def check_connected(edge_matrix, target_pos, search_range=2):
    """
    检查target_pos中的点是否有连接
    :param edge_matrix: 标记矩阵，元素为0/1，两点之间有通路则表示有连接
    :param target_pos: 二维元组，考察的目标点坐标，(list_of_i, list_of_j)
    :param search_range: 对每个目标点搜索连通性的范围，如2表示搜索自身为中心的的5*5邻域
    :return: 只要存在target_pos中的点有额外的连接（在紧区域内未连接但扩展区域内有连接），则返回True
    """
    pos_count = len(target_pos[0])
    h, w = edge_matrix.shape
    min_i, min_j = np.min(target_pos, axis=1)
    max_i, max_j = np.max(target_pos, axis=1)
    # branch_n = branch_count_2path(edge_matrix[min_i:max_i + 1, min_j:max_j + 1], return_branch_map=False)
    _, branch_n = measure.label(edge_matrix[min_i:max_i + 1, min_j:max_j + 1], background=0, return_num=True, connectivity=2)
    # branch_n_2p, _ = branch_count_2path(edge_matrix[min_i:max_i + 1, min_j:max_j + 1])
    # if branch_n != branch_n_2p:
    #     print('Watch!')
    min_i_expand = norm_index(min_i - search_range, h)
    max_i_expand = norm_index(max_i + search_range, h)
    min_j_expand = norm_index(min_j - search_range, w)
    max_j_expand = norm_index(max_j + search_range, w)
    # _, branch_id = branch_count_2path(edge_matrix[min_i_expand:max_i_expand + 1, min_j_expand:max_j_expand + 1])
    branch_id = measure.label(edge_matrix[min_i_expand:max_i_expand + 1, min_j_expand:max_j_expand + 1],
                              background=0, return_num=False, connectivity=2)
    # _, branch_id_2p = branch_count_2path(edge_matrix[min_i_expand:max_i_expand + 1, min_j_expand:max_j_expand + 1])
    # if np.sum(np.sum(np.abs(branch_id-branch_id_2p))) != 0:
    #     print('Watch!')
    branch_id = [branch_id[target_pos[0][i] - min_i_expand, target_pos[1][i] - min_j_expand] for i in range(pos_count)]
    branch_n_expand = len(set(branch_id))
    if branch_n_expand < branch_n:
        return True
    return False


def get_degree(matrix, pos, neighbor=8, ignore0=True):
    """
    计算某个位置的度，即8邻域中的1个数-1
    """
    i, j = pos
    h, w = matrix.shape
    if ignore0:
        if matrix[i, j] == 0:
            return 0
    min_i, max_i, min_j, max_j = get_index(i, j, h, w, 1)
    if neighbor == 8:
        return np.sum(matrix[min_i:max_i + 1, min_j:max_j + 1]) - matrix[i, j]
    else:
        assert neighbor == 4
        return np.sum([matrix[min_i, j], matrix[max_i, j], matrix[i, min_j], matrix[i, max_j]])


def dist(a1, a2):
    """
    计算a1中的各点到a2中各点的汉明距离
    :param a1: 二维坐标组，m*2
    :param a2: 二维坐标组，n*2
    :return: m*n的距离矩阵
    """
    m, d1 = a1.shape
    n, d2 = a2.shape
    assert d1 == 2 and d2 == 2
    d = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d[i, j] = np.sum(np.abs(a1[i] - a2[j]))
    return np.min(d.flatten())


# def walk(matrix, start_pos):
#     """
#     从边缘图上的一个端点（8邻域度为1）开始搜寻周围的点，找到另一个点（终点），在：
#         - 该点与起始点连通
#     的条件下，使得：
#         - 通路附近的点到连线的距离不超过阈值
#         - 对于端点到通路上终点的下一个点的连线，存在通路上的点到其距离超过阈值
#     基于该先验知识：通路上的点越多，到连线的平均距离越大，可以要求：对于端点到通路上终点的下几个点的连线，通路上到其的平均距离始终增加。
#     即延长通路将在一定范围内使平均距离一直增加
#     :param matrix: 单像素边缘图，像素值为0/1
#     :param start_pos: 起始搜索点的位置，必须是端点
#     :return: 起始点和终止点的坐标，以及去除该通路后的边缘图
#     """
#     pass


def bypass(edge_local, line, column):
    """
    判断(i,j)是否为现有分支旁边的点
    """
    h, w = edge_local.shape
    min_i, max_i, min_j, max_j = get_index(line, column, h, w, 1)  # 邻域的坐标范围
    # branch_c, branch_id = branch_count_2path(edge_local)
    branch_id, branch_c = measure.label(edge_local, background=0, return_num=True, connectivity=2)
    edge_p = edge_local.copy()
    edge_p[line, column] = 1
    # branch_p, branch_id_p = branch_count_2path(edge_p)
    branch_id_p, branch_p = measure.label(edge_p, background=0, return_num=True, connectivity=2)
    if branch_p > branch_c:
        return True  # 新的边缘点使分支数增加，则是冗余点
    if branch_p < branch_c:
        return False  # 新的边缘点使分支数减小，则不是冗余点
    if branch_c == 1:
        neighbor_8 = edge_local[min_i:max_i + 1, min_j:max_j + 1]
        neighbor_8_count = np.sum(neighbor_8.flatten())
        if neighbor_8_count >= 3:
            return True
        if neighbor_8_count == 1:  # 8邻域有一个点，去掉该点看edge_local中剩余的4分支数
            neighbor = np.where(neighbor_8)
            neighbor_temp = edge_local.copy()
            neighbor_temp[neighbor[0][0] + min_i, neighbor[1][0] + min_j] = 0
            # branch_pp = branch_count_2path(neighbor_temp, neighbor_range=4, return_branch_map=False)
            _, branch_pp = measure.label(neighbor_temp, background=0, return_num=True, connectivity=1)
            if branch_pp == 1:
                return False
            return True
        # 8邻域内有2个点，考虑4邻域
        neighbor_4_pos = [[line, min_j], [line, max_j], [min_i, column], [max_i, column]]  # 4邻域的坐标
        neighbor_4 = [edge_local[neighbor_4_pos[i][0], neighbor_4_pos[i][1]] == 1 for i in range(4)]  # 4邻域是否为边缘点
        if np.sum(neighbor_4) == 1:  # 4邻域有一个边缘点
            if not is_bridge(edge_local, [neighbor_4_pos[neighbor_4.index(1)]])[0]:  # 4邻域的相邻点不是桥，则不是冗余点
                return False
        return True
    if branch_c == 2:
        d = dist(np.array(np.where(branch_id == 1)).transpose(),
                 np.array(np.where(branch_id == 2)).transpose())
        d_p = dist(np.array(np.where(branch_id_p == 1)).transpose(),
                   np.array(np.where(branch_id_p == 2)).transpose())
        if d > d_p:
            return False  # 能减小分支距离，则不是冗余点
        return True
    if branch_c == 3:
        # 加入该点后该分支到其他分支的距离
        b_id = branch_id_p[line, column]  # 该点所在分支序号
        other_id = [i for i in range(1, 4) if i != b_id]  # 其余分支序号
        d1 = dist(np.array(np.where(branch_id_p == b_id)).transpose(),
                  np.array(np.where(branch_id_p == other_id[0])).transpose())
        d2 = dist(np.array(np.where(branch_id_p == b_id)).transpose(),
                  np.array(np.where(branch_id_p == other_id[1])).transpose())
        d_p = d1 + d2
        # 加入该点前对应分支到其他分支的距离
        b_id = np.max(((branch_id_p == b_id) * branch_id).flatten())  # 该点对应的加入前的分支序号
        other_id = [i for i in range(1, 4) if i != b_id]  # 其余分支序号
        d1 = dist(np.array(np.where(branch_id == b_id)).transpose(),
                  np.array(np.where(branch_id == other_id[0])).transpose())
        d2 = dist(np.array(np.where(branch_id == b_id)).transpose(),
                  np.array(np.where(branch_id == other_id[1])).transpose())
        d = d1 + d2
        if d_p < d:  # 减小了分支距离，则不是冗余点
            return False
        return True
    return True  # 超过4个分支时，一概认为该点是冗余点


def is_bridge(edge_map, loc):
    """
    判断loc中的各个点是否是桥接点
    :param edge_map: 边缘图
    :param loc: list of n lists，list中的各个元素表示考察点的坐标
    :return: n个元素的list，表示各个点是否是桥接点
    """
    h, w = edge_map.shape
    bridge = []
    for i in range(len(loc)):
        l, c = loc[i]
        # assert edge_map[l, c] == 1
        edge_temp = edge_map.copy()
        edge_temp[l, c] = 0
        min_i, max_i, min_j, max_j = get_index(l, c, h, w, 1)
        neighbor_map = edge_temp[min_i:max_i + 1, min_j:max_j + 1]
        neighbor_count = int(np.sum((neighbor_map == 1).flatten()))
        if neighbor_count == 1:  # 8邻域内只有一个点，则肯定不是桥接点
            bridge.append(False)
            continue
        # if neighbor_count >= 3:
        #     bridge.append(True)
        #     continue
        _, branch_c = measure.label(neighbor_map, background=0, return_num=True, connectivity=1)
        if branch_c == 1:  # 去除该点后仍然只有一个4连通分支，则该点也不是桥接点
            bridge.append(False)
            continue
        bridge.append(True)
    return bridge


def mark_edge(ratio_map, edge_map, threshold, line, column, neighbor_range):
    """
检查指定位置是否为有效边缘点并标记
    :param ratio_map: 各个点作为边缘点的权重
    :param edge_map: 边缘点标记
    :param threshold: 阈值，对阈值上下的部分分别处理
    :param line: 考察行序号
    :param column: 考察列序号
    :param neighbor_range: 搜索邻域的大小
    :return: 是否进行了标记
    """
    # if np.sum(np.abs([line - 121, column - 176])) <= 2:
    #     print('watch!')
    h, w = ratio_map.shape
    min_i, max_i, min_j, max_j = get_index(line, column, h, w, neighbor_range)
    neighbor_map = edge_map[min_i:max_i + 1, min_j:max_j + 1].copy()
    neighbor_count = int(np.sum(neighbor_map == 1))
    # 1. ratio高于阈值
    if ratio_map[line, column] > threshold:
        ratio_neighbor = ratio_map[min_i:max_i + 1, min_j:max_j + 1]
        confident_pos = np.where(ratio_neighbor > threshold * 0.8)  # ratio 高于阈值的点
        if min(confident_pos[0]) == 0 or max(confident_pos[0]) == max_i - min_i \
                or min(confident_pos[1]) == 0 or max(confident_pos[1]) == max_j - min_j:  # 能与外界连通
            # 邻域内没有边缘点
            if neighbor_count == 0:
                edge_map[line, column] = 1  # 与外界连通
                return True
            # 以下考虑不能够成为边缘点的情况
            # 邻域内有1个或2个边缘点
            if neighbor_count in [1, 2]:
                loc_relative = np.where(neighbor_map == 1)
                loc = [[loc_relative[0][i] + min_i, loc_relative[1][i] + min_j] for i in range(neighbor_count)]
                degrees = [get_degree(edge_map, [loc[i][0], loc[i][1]]) for i in range(neighbor_count)]
                if (np.array(degrees) >= 2).all():  # 附近有1个或两个边缘点时，如果点的度都超过1:
                    if np.array(is_bridge(edge_map, loc)).all():  # 如果两个点都是桥接点，则该点不连接
                        return False
                edge_map[line, column] = 1
                return True
            # 邻域内有三个及以上边缘点
            if neighbor_count >= 3:  # 当附近已经有三个边缘点，需要考虑是否是旁支点
                if bypass(neighbor_map, line - min_i, column - min_j):  # 是旁支点，则不作为边缘
                    return False
                # 其余情况：
                # 1. 搜索邻域内下一个未标记、强度高的点，若也不是旁支点，则记为候选点
                # 2. 比较候选点和当前点标记后的4连通分支数，标记高的一个
                min_i_1, max_i_1, min_j_1, max_j_1 = get_index(line, column, h, w, 1)  # 1邻域的基准坐标
                ratio_neighbor_1 = ratio_map[min_i_1:max_i_1 + 1, min_j_1:max_j_1 + 1]
                neighbor_map_1 = edge_map[min_i_1:max_i_1 + 1, min_j_1:max_j_1 + 1].copy()
                candidate_i, candidate_j = np.where((ratio_neighbor_1 > threshold) * (neighbor_map_1 == 0))
                # branch_ori4 = branch_count_2path(neighbor_map, 4, False)
                _, branch_ori4 = measure.label(neighbor_map, background=0, return_num=True, connectivity=1)
                # branch_ori8 = branch_count_2path(neighbor_map, return_branch_map=False)
                _, branch_ori8 = measure.label(neighbor_map, background=0, return_num=True, connectivity=2)
                neighbor_center = neighbor_map.copy()
                neighbor_center[line - min_i, column - min_j] = 1  # 将当前像素点置为1，计算4分支数
                # branch_center4 = branch_count_2path(neighbor_center, 4, False)  # 当前位置当作边缘的4分支数
                _, branch_center4 = measure.label(neighbor_center, background=0, return_num=True, connectivity=1)
                # branch_center8 = branch_count_2path(neighbor_center, return_branch_map=False)  # 当前位置当作边缘的4分支数
                _, branch_center8 = measure.label(neighbor_center, background=0, return_num=True, connectivity=2)
                if branch_center4 != 1 and branch_center8 < branch_ori8 and branch_center4 >= branch_ori4:  # 当前位置4分支数大于1，且通过连接减少了8分支数、但没能减少4分支数，则有可能通过候选点改进
                    pos_possible, branch4_possible = [], []
                    for candidate_index in range(len(candidate_i)):
                        i_, j_ = candidate_i[candidate_index], candidate_j[candidate_index]
                        if min_i_1 + i_ == line and min_j_1 + j_ == column:  # 当前点或旁支点
                            continue
                        if bypass(neighbor_map, i_, j_):
                            continue
                        neighbor_candidate = neighbor_map.copy()
                        neighbor_candidate[min_i_1 + i_ - min_i, min_j_1 + j_ - min_j] = 1  # 将候选像素点置为1，计算4分支数
                        # branch_candidate = branch_count_2path(neighbor_candidate, 4, False)  # 候选位置当作边缘的4分支数
                        _, branch_candidate = measure.label(neighbor_candidate, background=0, return_num=True, connectivity=1)
                        if branch_candidate < branch_center4:  # 候选点能够实现更小的4连通分支，或避免更多的4连通分支
                            pos_possible.append(candidate_index)
                            branch4_possible.append(branch_candidate)
                    # 选择能实现更少4连通分支的候选点
                    if branch4_possible.__len__() != 0:
                        final_index = pos_possible[branch4_possible.index(min(branch4_possible))]
                        edge_map[min_i_1 + candidate_i[final_index], min_j_1 + candidate_j[final_index]] = 1
                        # print('revise a point')
                        return True
                edge_map[line, column] = 1  # 没有候选点抢占边缘，则将该点置为边缘
                return True
        else:
            return False
    # 2. ratio低于阈值，考虑其对现有边缘点的连接情况
    if neighbor_count <= 1:  # 跳过附近只有1个点的情况
        edge_map[line, column] = -1  # 此时抑制该点，附近其他点将有更低的权重
        return False
    # 2.1 若能连接现有分支，则视为边缘连接点
    # branch_original, branches_original = branch_count_2path(neighbor_map)
    branches_original, branch_original = measure.label(neighbor_map, background=0, return_num=True, connectivity=2)
    if branch_original == 1:  # 邻域只有一个分支，则直接跳过。否则一定能连接现有分支
        return False
    # 邻域内有超过一个分支时，考虑是否在邻域外有连接，避免产生多余的连接
    if check_connected(edge_map, np.array(np.where(neighbor_map == 1)) + np.array([[min_i], [min_j]])):
        return False
    neighbor_expand = conv(neighbor_map, np.ones([3, 3], dtype='int8'), mode='same')
    neighbor_map[neighbor_expand > 0] = 1
    neighbor_map[line - min_i, column - min_j] = 1
    ratio_local = ratio_map[min_i:max_i + 1, min_j:max_j + 1]
    edge_weights = conv(neighbor_map * ratio_local, np.ones([3, 3], dtype='int16'), mode='same')  # 在这里将边缘点的权重设为0，减少后续
    edge_local = edge_map[min_i:max_i + 1, min_j:max_j + 1]
    indexes_weights = np.argsort(edge_weights.flatten())[::-1]
    indexes_weights = np.unravel_index(indexes_weights, edge_weights.shape)
    i, branch_c, branch_idx = 0, branch_original, branch_original + 1
    h_neighbor, w_neighbor = neighbor_map.shape
    while True:
        if branch_c == 1:  # 赋予新的边缘点，直到连接数等于1
            break
        if i == 25:
            print('watch!')
        target_i, target_j = indexes_weights[0][i], indexes_weights[1][i]
        if edge_local[target_i, target_j] == 1:  # 已经是边缘点，则考虑下一个位置
            i += 1
            continue
        if not bad_connect(edge_local, target_i, target_j):  # 非多余边缘点，则赋值为边缘
            edge_local[target_i, target_j] = 1
            min_i, max_i, min_j, max_j = get_index(target_i, target_j, h_neighbor, w_neighbor, 1)  # 候选点的邻域
            branches_id = list(set(branches_original[min_i:max_i + 1, min_j:max_j + 1].flatten()))[1:]
            branch_count_neighbor = len(branches_id)
            if branch_count_neighbor == 0:  # 邻域没有分支，则产生一个新的分支
                branches_original[target_i, target_j] = branch_idx
                branch_c += 1
                branch_idx += 1
            elif branch_count_neighbor == 1:  # 邻域有分支但无法连接分支
                branches_original[target_i, target_j] = branches_id[0]
            else:  # 能在邻域连接现有分支
                for b_id in branches_id[1:]:
                    branches_original[branches_original == b_id] = branches_id[0]
                branches_original[target_i, target_j] = branches_id[0]
                branch_c -= branch_count_neighbor - 1
            # branch_c = branch_count_2path(edge_local, return_branch_map=False)  # 重新计算分支数
        i += 1
    return True


def get_degree4(image):
    """计算各个点的 4 邻域度"""
    up = np.roll(image, -1, axis=0)  # 向上移动，即获得下方像素点的值。后同
    down = np.roll(image, 1, axis=0)
    left = np.roll(image, -1, axis=1)
    right = np.roll(image, 1, axis=1)
    # Zero out edges since roll wraps around
    up[-1, :] = 0
    down[0, :] = 0
    left[:, -1] = 0
    right[:, 0] = 0
    # Sum the shifted maps to get the neighbor count
    neighbor_count = up + down + left + right
    return neighbor_count


def get_degree8(image):
    """计算各个点的 8 邻域度"""
    # Create shifted maps to count neighbors
    up = np.roll(image, -1, axis=0)
    down = np.roll(image, 1, axis=0)
    left = np.roll(image, -1, axis=1)
    right = np.roll(image, 1, axis=1)

    up_left = np.roll(up, -1, axis=1)
    up_right = np.roll(up, 1, axis=1)
    down_left = np.roll(down, -1, axis=1)
    down_right = np.roll(down, 1, axis=1)

    # Zero out edges since roll wraps around
    up[-1, :] = 0
    down[0, :] = 0
    left[:, -1] = 0
    right[:, 0] = 0

    up_left[-1, :] = 0
    up_left[:, -1] = 0
    up_right[-1, :] = 0
    up_right[:, 0] = 0
    down_left[0, :] = 0
    down_left[:, -1] = 0
    down_right[0, :] = 0
    down_right[:, 0] = 0

    # Sum the shifted maps to get the neighbor count
    neighbor_count = up + down + left + right + up_left + up_right + down_left + down_right

    return neighbor_count


def get_square(image):
    """找到出现的 2x2 正方形，返回正方形右下角的坐标"""
    down = np.roll(image, 1, axis=0)
    right = np.roll(image, 1, axis=1)
    down_right = np.roll(down, 1, axis=1)
    down[0, :] = 0
    right[:, 0] = 0
    down_right[0, :] = 0
    down_right[:, 0] = 0
    return np.where(image + down + right + down_right == 4)


def get_diag(image):
    """找到斜点，其模式为[1,0;0,1]或[0,1;1,0]，获得该模型右下角点的坐标"""
    down = np.roll(image, 1, axis=0)
    right = np.roll(image, 1, axis=1)
    down_right = np.roll(down, 1, axis=1)
    down[0, :] = 0
    right[:, 0] = 0
    down_right[0, :] = 0
    down_right[:, 0] = 0
    return np.where((image != down) * (image != right) * (down != down_right))

def process_shrink(edge_map, threshold_ratio=0.8 * 0.6, search_range=2):
    """
对边缘强度图进行缩减，获得单像素边缘图
    :param edge_map: 单通道边缘强度图，注意数据类型不能为 uint*
    :param threshold_ratio: 加权占优比阈值，对阈值以上、0.8倍阈值以上的点进行不同的处理
    :param search_range: 参与比较以及边缘判定的范围
    :return: 收缩之后的边缘图
    """
    h, w = edge_map.shape
    ratio_matrix, neighbor_count = calculate_ratio(edge_map, search_range)
    ratio_matrix *= edge_map / 255  # 用该点的边缘强度进行加权
    # ratio_matrix *= conv(edge_map, np.ones([5, 5], dtype='int8'), mode='same') / neighbor_count / 255  # 用该点的滤波后边缘强度进行加权
    # 边缘收缩
    indexes_sorted = np.argsort(-ratio_matrix.flatten())
    indexes_sorted = np.unravel_index(indexes_sorted, [h, w])
    edge_shrink = np.zeros([h, w], dtype='int8')
    for index in range(h * w):
        i, j = indexes_sorted[0][index], indexes_sorted[1][index]  # 从ratio最大处开始遍历
        if edge_shrink[i, j] == 1:  # 该点在之前的过程已经被标记
            continue
        if ratio_matrix[i, j] < threshold_ratio * 0.8:  # 遍历了所有高于阈值0.8倍的点
            break
        mark_edge(ratio_matrix, edge_shrink, threshold_ratio, i, j, search_range)
    edge_shrink[edge_shrink < 0] = 0
    return edge_shrink


def _process_fill(edge, h, w):
    """
对斜边进行填补，使得所有点均通过4邻域与其他点相连
    """
    edge_copied = edge.copy()
    # 找到斜点，其模式为[1,0;0,1]或[0,1;1,0]，获得该模型右下角点的坐标
    # pos_weak = np.where((np.abs(conv(edge_copied, np.array([[1, -1], [-1, 1]]), mode='same')) == 2))
    pos_weak = get_diag(edge_copied.astype('bool'))
    for index in range(len(pos_weak[0])):
        line, column = pos_weak[0][index], pos_weak[1][index]
        if edge_copied[line, column] == 0:
            column -= 1
        min_i, max_i, min_j, max_j = get_index(line, column, h, w, 1)  # 1邻域的范围
        for i, j in [[min_i, min_j], [min_i, max_j], [max_i, min_j], [max_i, max_j]]:
            if edge_copied[i, j] == 1 and edge_copied[line, j] == 0 and edge_copied[i, column] == 0:
                degree1 = get_degree(edge_copied, [line, j], 4, False)
                degree2 = get_degree(edge_copied, [i, column], 4, False)
                if degree1 <= degree2:
                    edge_copied[line, j] = 1
                else:
                    edge_copied[i, column] = 1
    # 0. 对4邻域度为0的点，若其8邻域度不为0，则进行填补
    # cv2.imwrite('results/edge_stage0.png', np.uint8(edge_copied * 255))
    return edge_copied


def _process_del(edge, h, w, neighbor=4):
    """
删除度为1且邻点度大于等于3的点，即分叉点
    """
    # kernel_degree4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='uint8')
    # kernel_degree8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype='uint8')
    # degree_raw = conv(edge, kernel_degree4, mode='same') if neighbor == 4 else conv(edge, kernel_degree8, mode='same')
    degree_raw = get_degree4(edge) if neighbor == 4 else get_degree8(edge)
    degree = degree_raw * edge  # 4邻域度
    edge_stage1 = edge.copy()
    pos_d1 = np.where(degree == 1)  # 4邻域度为1的点的位置
    for index in range(len(pos_d1[0])):
        line, column = pos_d1[0][index], pos_d1[1][index]  # 点的位置
        # if [line, column] == [66, 5]:
        #     print('watch')
        min_i, max_i, min_j, max_j = get_index(line, column, h, w, 1)  # 邻域的范围
        if neighbor == 8:
            mesh = np.meshgrid(np.arange(min_i, max_i + 1), np.arange(min_j, max_j + 1))
            candidate = list(np.array([mesh[0].flatten(), mesh[1].flatten()]).T)
            candidate = [[a[0], a[1]] for a in candidate if not (a[0] == line and a[1] == column)]
        else:
            candidate = [[a[0], a[1]] for a in [[min_i, column], [max_i, column], [line, min_j], [line, max_j]]
                         if not (a[0] == line and a[1] == column)]  # 四邻域点的位置
        is_edge = [edge[a[0], a[1]] for a in candidate]
        neighbor_line, neighbor_column = candidate[is_edge.index(1)]
        if degree[neighbor_line, neighbor_column] >= 3:
            if not is_bridge(edge, [[line, column]])[0]:  # 同时该点不是桥接点
                edge_stage1[line, column] = 0
                # print('erase point at (%d, %d)' % (line, column))
    # 1. 4邻域度为1的点，其邻点度应不超过2。桥接点除外
    # cv2.imwrite('results/edge_stage1.png', np.uint8(edge_stage1 * 255))
    return edge_stage1


def _process_shift(edge_stage1, h, w):
    """
对“凹”形点进行修正
    """
    edge_stage2 = edge_stage1.copy()
    # kernel_degree4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='uint8')
    # degree4_stage1 = conv(edge_stage2, kernel_degree4, mode='same')
    degree4_stage1 = get_degree4(edge_stage2)
    pos_missed = np.where((degree4_stage1 >= 3) * (edge_stage2 == 0))  # 凹形点
    for index in range(len(pos_missed[0])):
        line, column = pos_missed[0][index], pos_missed[1][index]
        # if [line, column] == [96, 273]:
        #     print('watch')
        if get_degree(edge_stage2, [line, column], 4, ignore0=False) < 3:
            continue
        min_i, max_i, min_j, max_j = get_index(line, column, h, w, 2)  # 2-邻域的范围
        # edge_neighbor = edge_stage2[min_i:max_i + 1, min_j:max_j + 1]  # 2邻域的边缘点。非拷贝，这里的赋值会影响edge
        min_i_1, max_i_1, min_j_1, max_j_1 = get_index(line, column, h, w, 1)  # 1-邻域的范围
        edge_stage2[min_i_1:max_i_1 + 1, min_j_1:max_j_1 + 1] = 0
        edge_stage2[line, column] = 1
        fill_pos = []
        for i in range(min_i, max_i + 1):
            for j in range(min_j, max_j + 1):
                if edge_stage2[i, j] == 0 or [i, j] == [line, column]:
                    continue
                if np.sum(edge_stage2[min(i, line):max(i, line) + 1, min(j, column):max(j, column) + 1]) == 2:
                    fill_pos.append([i, j])
        for i, j in fill_pos:
            edge_stage2[int(np.around((i + line) / 2)), int(np.around((j + column) / 2))] = 1
    # 2. 4邻域度大于等于3的非边缘点，应该充当起边缘点的作用，对现有边缘进行简化
    # 具体操作为：删除1邻域内所有边缘点；计算当前2邻域内的4连通分支数n；标记当前非边缘点为边缘点；从距离该点最近的2邻域内点起，标注4邻域度（在全局范围内计算）不超过1的点与与该点连线的终点，标注n个
    # cv2.imwrite('results/edge_stage2.png', np.uint8(edge_stage2 * 255))
    return edge_stage2


def _process_square(edge_stage2):
    """
对“田”形点进行缩减
    """
    edge_stage3 = edge_stage2.copy()
    # square_map = conv(edge_stage2, np.ones([2, 2], dtype='uint8'), mode='same')
    # pos_square = np.where(square_map == 4)  # (i,j)处为4，表示其左上角是一个正方形（该点位于右下角）
    pos_square = get_square(edge_stage2)
    for index in range(len(pos_square[0])):
        line, column = pos_square[0][index], pos_square[1][index]
        if np.sum(edge_stage3[line-1: line+1, column-1: column+1]) < 4:  # 受其他操作的影响，原本方形点不再是方形
            continue
        # 获取正方形中的四个点的坐标
        candidate_list = [[line - 1, column - 1], [line - 1, column], [line, column - 1], [line, column]]
        # 选择不影响联通分支数的点
        _, initial_connect_count = measure.label(
            edge_stage3[max(line - 2, 0): line + 2, max(column - 2, 0): column + 2],
            background=0, return_num=True, connectivity=1
        )  # 联通分支数
        finished = False
        for candidate_id in range(4):
            edge_stage3[candidate_list[candidate_id][0], candidate_list[candidate_id][1]] = 0  # 尝试将一个位置置零
            _, connect_count = measure.label(
                edge_stage3[max(line - 2, 0): line + 2, max(column - 2, 0): column + 2],
                background=0, return_num=True, connectivity=1
            )  # 联通分支数
            if connect_count == initial_connect_count:  # 连通分支数不变，则可接受
                finished = True
                break
            else:  # 否则考虑另外的点
                edge_stage3[candidate_list[candidate_id][0], candidate_list[candidate_id][1]] = 1
        # 没有成功消除方形块，说明发生风车状
        if not finished:
            if edge_stage3[line - 1, column + 1] == 1:  # 存在的四种消除-填充方案保证连通性
                candidate_fill = [
                    [line, column + 1],
                    [line + 1, column - 1],
                    [line - 1, column - 2],
                    [line - 2, column]
                ]
            else:
                candidate_fill = [
                    [line - 2, column - 1],
                    [line - 1, column + 1],
                    [line + 1, column],
                    [line, column - 2]
                ]
            candidate_del = [
                [line - 1, column],
                [line, column],
                [line, column - 1],
                [line - 1, column - 1]
            ]
            min_degree_idx, min_degree = 0, get_degree(edge_stage3, candidate_fill[0], neighbor=4, ignore0=False)
            for candidate_idx in range(1, 4):
                if min_degree == 2: break  # 最小度即为 2
                degree_candidate = get_degree(edge_stage3, candidate_fill[candidate_idx], neighbor=4, ignore0=False)
                if degree_candidate < min_degree:
                    min_degree_idx, min_degree = candidate_idx, degree_candidate
            edge_stage3[candidate_fill[min_degree_idx][0], candidate_fill[min_degree_idx][1]] = 1
            edge_stage3[candidate_del[min_degree_idx][0], candidate_del[min_degree_idx][1]] = 0
    # 3. 缩减方形节点。其出现是由于后加入的点能够减小与其他边缘的距离
    # cv2.imwrite('results/edge_stage3.png', np.uint8(edge_stage3 * 255))
    return edge_stage3


def _process_phase(edge, h, w, phase, neighbor=4):
    """
循环进行四种不同的处理
    """
    phase %= 4
    if phase == 0 and neighbor == 4:  # 如果按4邻域计算，则需要对斜点进行填充
        return _process_fill(edge, h, w)
    if phase == 1:
        return _process_del(edge, h, w, neighbor=neighbor)
    if phase == 2 and neighbor == 4:  # 如果按4邻域计算，需要对凹形区域进行修正
        return _process_shift(edge, h, w)
    if phase == 3 and neighbor == 4:  # 如果按4邻域计算，则在填充或凹形修正后可能出现正方形块儿
        return _process_square(edge)
    return edge


def process_clip(edge: np.ndarray, max_iter=100, neighbor=4, valid_shift=None):
    """
    对单像素宽边缘图进行修剪，使得最终的边缘图：
    1. 在4邻域度量下，大部分边缘点度为2，边缘端点为1，分支节点为3或4，不存在度为5的节点；
    2. 分支节点邻接点的度为2
    2. 不存在两个相邻的分支节点
    基于以上目标，对现有边缘图的操作包括：
    0. 对8邻域度为正但4邻域度为0的点进行扩展
    1. 删除4邻域度为1且4邻域邻点度大于等于3（且不为桥接点）的点
    2. 对“凹”型区域进行修正
    3. 对正方形区域进行修正，即删除多余的分支节点
    :param max_iter: 最大处理次数
    :param edge: h*w bool型边缘图
    :param neighbor: 4或8，表示邻域的计算方式
    :param valid_shift: 有效的线条变换列表，含义参考 process_phase
    :return: 经裁剪后的边缘图
    """
    if valid_shift is None:
        valid_shift = [0, 1, 2, 3]
    h, w = edge.shape
    edge_processed = edge.copy()
    stay = 0
    for i in range(max_iter):
        if i % 4 not in valid_shift:
            continue
        edge_ = _process_phase(edge_processed, h, w, phase=i, neighbor=neighbor)
        edge_: np.ndarray
        keep: np.ndarray
        keep = edge_ == edge_processed
        if keep.all():
            stay += 1
            if stay == 4:
                break
        else:
            stay = 0
        # print('Changed pixel:', h * w - np.sum(keep))
        edge_processed = edge_.copy()
    if 0 in valid_shift:
        edge_processed = _process_phase(edge_processed, h, w, phase=0, neighbor=neighbor)
    if 1 in valid_shift:
        edge_processed = _process_phase(edge_processed, h, w, phase=1, neighbor=neighbor)
    if 3 in valid_shift:
        edge_processed = _process_phase(edge_processed, h, w, phase=3, neighbor=neighbor)
    return edge_processed


def get_distance(x1, y1, x2, y2, x, y):
    """
计算点(x,y)到(x1,y1), (x2,y2)连线的距离
    """
    if x1 == x2 and y1 == y2:
        L = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    else:
        x, y = np.array(x), np.array(y)
        L = abs((y1 - y2) * x + (x2 - x1) * y + x1 * y2 - y1 * x2) \
            / np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
    return L


def split_path(path, threshold=1):
    """
    对所给path进行分割，使得每一段通过线段拟合的最大距离不超过阈值
    :param path: 所给路径坐标，二维数组的列表
    :param threshold: 阈值
    :return:
        split_indexes: 分割点的序号
        distances: path中各个点到所属段拟合线段的距离
    """
    if len(path) == 1:
        return [], [0], [0]
    if len(path) == 2:
        return [], [0, 0], [0, 0]
    # 二分法分割：对于所给path，先通过首尾拟合，若最大距离大于阈值，则在距离最大处分割
    split_indexes = []
    l, c = [], []
    for p in path:
        l.append(p[0])
        c.append(p[1])
    # print(l, c)
    distances = list(get_distance(l[0], c[0], l[-1], c[-1], l, c))
    max_index = int(np.argmax(distances))
    if distances[max_index] > threshold:
        split_indexes.append(max_index)
        split_indexes_1, distances_1, distance_max_1 = split_path(path[:max_index], threshold)
        split_indexes_2, distances_2, distance_max_2 = split_path(path[max_index:], threshold)
        split_indexes += split_indexes_1 + [index + max_index for index in split_indexes_2]
        distances = distances_1 + distances_2
        split_indexes.sort()
        return split_indexes, distances, distance_max_1 + distance_max_2
    else:
        return split_indexes, distances, [distances[max_index]]


def get_line_points(p1, p2):
    """
    获得直线段绘制出来各个点的坐标，格式与p1,p2一致
    :param p1: 二维数组或点坐标
    :param p2: 二维数组或点坐标
    :return: N*2数组，N为绘制点的个数
    """
    if p1.__class__ == Point:
        x1, y1 = p1.x, p1.y
    else:
        x1, y1 = p1[0], p1[1]
    if p2.__class__ == Point:
        x2, y2 = p2.x, p2.y
    else:
        x2, y2 = p2[0], p2[1]
    points_count = int(max(abs(x1 - x2), abs(y1 - y2)) + 1)
    plot_x = np.linspace(x1, x2, num=points_count).astype('int')
    plot_y = np.linspace(y1, y2, num=points_count).astype('int')
    return np.stack([plot_x, plot_y]).T


def find_neighbor(edge, degree, visited, location, neighbor=4, include_cross=False):
    """
找到location点附近未被访问的邻域点
    :param include_cross: 是否包含访问过的交叉点
    :param edge: 边缘图
    :param degree: 边缘点的度
    :param visited: 是否被访问的标志，避免重复与死循环
    :param location: 考察点的坐标
    :param neighbor: 邻域范围
    :return: 邻域点列表，元素为二元数组或列表
    """
    h, w = edge.shape
    line, column = location
    min_i, max_i, min_j, max_j = get_index(line, column, h, w, 1)
    neighbor_edge = edge[min_i:max_i + 1, min_j:max_j + 1].copy()
    neighbor_visited = visited[min_i:max_i + 1, min_j:max_j + 1]
    neighbor_degree = degree[min_i:max_i + 1, min_j:max_j + 1]
    neighbor_edge[line - min_i, column - min_j] = 0  # 邻点不包括自己
    if include_cross:
        neighbor_pos = np.where(
            (neighbor_edge == 1) * (neighbor_visited == 0) + (neighbor_edge == 1) * (neighbor_degree > 2))
    else:
        neighbor_pos = np.where((neighbor_edge == 1) * (neighbor_visited == 0))
    if neighbor == 4:
        neighbor = [[neighbor_pos[0][i] + min_i, neighbor_pos[1][i] + min_j] for i in range(len(neighbor_pos[0])) if
                    ((neighbor_pos[0][i] + min_i == line) + (neighbor_pos[1][i] + min_j == column)) == 1]
    else:
        neighbor = [[neighbor_pos[0][i] + min_i, neighbor_pos[1][i] + min_j] for i in range(len(neighbor_pos[0])) if
                    (neighbor_pos[0][i] + min_i == line) * (neighbor_pos[1][i] + min_j == column) == 0]
    return neighbor  # list of positions


def del_stick(edge, degree):
    """清除边缘图中的分叉线，即端点度为 1"""
    edge_cp = edge.copy()
    end_nodes = np.stack(np.where(degree == 1)).T  # 端点的位置
    visited = np.zeros_like(edge)
    for end_node in end_nodes:
        neighbors = [end_node]
        while neighbors:  # 存在邻点
            neighbor = neighbors.pop(0)
            if degree[neighbor[0], neighbor[1]] == 3:
                break
            visited[neighbor[0], neighbor[1]] = True
            neighbors = find_neighbor(edge, degree, visited, neighbor)
            edge_cp[neighbor[0], neighbor[1]] = 0  # 置为 0，表示该点已被清除
    return edge_cp


def save_pickle(path, data):
    """
将类对象存储到文件
    """
    with open(path, 'wb') as f:
        strings = pickle.dumps(data)
        f.write(strings)


def load_pickle(path):
    """
从文件加载类对象
    """
    with open(path, 'rb') as f:
        data = pickle.loads(f.read())
    return data


def get_edge(image, model, args, transform):
    """
通过神经网络获取边缘强度图
    :param image: h*w*3 uint8 RGB图像
    :param model: 神经网络模型
    :param args: 参数
    :param transform: 变换，包括维度变形和归一化
    :return: h*w uint8 边缘强度图
    """
    # 预处理
    image = transform(image.copy())[None]
    image = image.cuda() if args.use_cuda else image
    end = time.perf_counter()
    torch.cuda.synchronize()
    _, _, H, W = image.shape
    # 边缘检测
    with torch.no_grad():
        results = model(image)
        edge = (torch.squeeze(results[-1]).cpu().numpy() * 255).astype(np.uint8)
    # save edge detected
    Image.fromarray(edge).save(os.path.join(args.savedir, 'edge.png'))
    torch.cuda.synchronize()
    end = time.perf_counter() - end
    print('[Time] edge detection: %f' % end)
    return edge


def load_parameters():
    """
加载归一化参数和变换
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    return normalize, transform


def arg_parser():
    """
参数解析
    """
    parser = argparse.ArgumentParser(description='PyTorch Diff Convolutional Networks (Test)')

    parser.add_argument('--savedir', type=str, default='results/savedir',
                        help='path to save result and checkpoint')
    parser.add_argument('--datadir', type=str, default='../data',
                        help='dir to the dataset')
    parser.add_argument('--only-bsds', action='store_true',
                        help='only use bsds for training')
    parser.add_argument('--ablation', action='store_true',
                        help='not use bsds val set for training')
    parser.add_argument('--dataset', type=str, default='BSDS',
                        help='data settings for BSDS, Multicue and NYUD datasets')

    parser.add_argument('--model', type=str, default='baseline',
                        help='model to train the dataset')
    parser.add_argument('--sa', action='store_true',
                        help='use CSAM in pidinet')
    parser.add_argument('--dil', action='store_true',
                        help='use CDCM in pidinet')
    parser.add_argument('--config', type=str, default='carv4',
                        help='model configurations, please refer to models/config.py for possible configurations')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed (default: None)')
    parser.add_argument('--gpu', type=str, default='',
                        help='gpus available')
    parser.add_argument('--checkinfo', action='store_true',
                        help='only check the information about the model: model size, flops')

    parser.add_argument('--epochs', type=int, default=20,
                        help='number of total epochs to run')
    parser.add_argument('--iter-size', type=int, default=24,
                        help='number of samples in each iteration')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='initial learning rate for all weights')
    parser.add_argument('--lr-type', type=str, default='multistep',
                        help='learning rate strategy [cosine, multistep]')
    parser.add_argument('--lr-steps', type=str, default=None,
                        help='steps for multistep learning rate')
    parser.add_argument('--opt', type=str, default='adam',
                        help='optimizer')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay for all weights')
    parser.add_argument('-j', '--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--eta', type=float, default=0.3,
                        help='threshold to determine the ground truth (the eta parameter in the paper)')
    parser.add_argument('--lmbda', type=float, default=1.1,
                        help='weight on negative pixels (the beta parameter in the paper)')

    parser.add_argument('--resume', action='store_true',
                        help='use latest checkpoint if have any')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--evaluate', type=str, default=None,
                        help='full path to checkpoint to be evaluated')
    parser.add_argument('--evaluate-converted', action='store_true',
                        help='convert the checkpoint to vanilla cnn, then evaluate')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.use_cuda = torch.cuda.is_available()
    print(args)
    return args


def load_image(path):
    """
加载RGB图像
    :return: h*w*3 RGB uint8图像
    """
    img = cv2.imread(path)[:, :, ::-1]
    return img


def save_image(path, image, scale=False):
    """
保存图像
    :param path: 存储路径
    :param image: h*w*3 RGB图像或h*w灰度图像
    :param scale: 是否进行尺度变换。若是，则将[0,1]范围的数据映射到[0,255]
    """
    if len(image.shape) == 3:
        image = image[:, :, ::-1]
    if not scale:
        cv2.imwrite(path, image.astype('uint8'))
    else:
        if np.max(image) <= 1:
            cv2.imwrite(path, image.astype('uint8') * 255)


def save_branch(branch_recorder, save_path, shape):
    branches_list = [[b.start_pos] + b.mid_pos + [b.end_pos] for b in branch_recorder.branches]
    for branch_idx in range(len(branches_list)):
        branch = branches_list[branch_idx]
        for node_idx in range(len(branch)):
            branch[node_idx] = [int(branch[node_idx][0]), int(branch[node_idx][1])]
    # with open('temp.json', 'w') as f:
    #     json.dump(branches_list, f)
    with open(save_path, 'w') as f:  # txt格式比json格式存储空间更小
        f.write('%d,%d\n' % (shape[0], shape[1]))
        for branch in branches_list:
            for node in branch:
                f.write('%d,%d ' % (node[0], node[1]))
            f.write('\n')


def save_branch_compressed(branch_recorder, save_path, size):
    """
对素描图以分支的形式进行紧凑的编码，码流组成为 (bit)：
    [11 bit] 0~10: 图像的宽度W-1（表示0~2047，H同）
    [11 bit] 11~21: 图像的高度H-1
    [11 bit] 22~32: x的最小值（表示0~2047，y同）
    [11 bit] 33~43: y的最小值
    [ 4 bit] 44~47: X，对x的单个数值编码的比特数（表示0~15，y同）
    [ 4 bit] 48~51: Y，对y的单个数值编码的比特数
    [16 bit] 52~67: B，表示分支个数
    [ 4 bit] 68~71: M，对各个分支长度数值进行编码的比特数（表示0~15）
    [MB bit] 72~71+MB: 各个分支的长度，(L1,L2,...,LM)范围在（0~2^M-1），素描点的个数 N=sum{Li}
    [NX bit] 72+MB~71+MB+NX: 对N个点的横坐标x-x_min编码的结果
    [NY bit] 72+MB+NX~71+MN+NX+NY: 对N个点的纵坐标y-y_min编码的结果
    :param branch_recorder: 记录素描图中各个分支的结果
    :param save_path: 编码后文件的保存位置
    :param size: 素描图的图像大小
    :return:
    """
    branches_list = [[b.start_pos] + b.mid_pos + [b.end_pos] for b in branch_recorder.branches]
    branches_list_of_arrays = [np.array(item) for item in branches_list]
    branches_arrays = np.concatenate(branches_list_of_arrays, axis=0)
    y_min, x_min = np.min(branches_arrays, axis=0)
    y_max, x_max = np.max(branches_arrays, axis=0)
    y_range, x_range = y_max - y_min + 1, x_max - x_min + 1
    h, w = size
    bit_x = int(np.ceil(np.log2(x_range)))  # 为x编码的比特数
    bit_y = int(np.ceil(np.log2(y_range)))  # 为y编码的比特数
    x_delta = branches_arrays[:, 1] - x_min
    y_delta = branches_arrays[:, 0] - y_min
    branch_length = [len(item) for item in branches_list]
    branch_count = len(branch_length)  # 分支的数目
    branch_length_max = max(branch_length)  # 最长分支的长度，决定了对分支长度编码的比特数
    bit_branch_length = int(np.ceil(np.log2(branch_length_max)))  # 对分支长度编码的比特数
    # 进行编码
    bins_W = dec2bin(w - 1, 11)
    bins_H = dec2bin(h - 1, 11)
    bins_x_min = dec2bin(x_min, 11)
    bins_y_min = dec2bin(y_min, 11)
    bins_x_bit = dec2bin(bit_x, 4)
    bins_y_bit = dec2bin(bit_y, 4)
    bins_branch_count = dec2bin(branch_count, 16)
    bins_branch_length_bit = dec2bin(bit_branch_length, 4)
    bins_branch_length = [dec2bin(l, bit_branch_length) for l in branch_length]  # 可以进行熵编码
    bins_x_delta = [dec2bin(item, bit_x) for item in x_delta]
    bins_y_delta = [dec2bin(item, bit_y) for item in y_delta]
    bins = bins_W + bins_H + bins_x_min + bins_y_min + bins_x_bit + bins_y_bit + bins_branch_count + bins_branch_length_bit \
           + ''.join(bins_branch_length) + ''.join(bins_x_delta) + ''.join(bins_y_delta)
    bytes_series = bitstring2bytes(bins)
    with open(save_path, 'wb') as f:
        f.write(bytes_series)


def save_graph(graph, save_path_graph, size):
    positions, connections = graph['positions'], graph['connections']
    N = positions.shape[0]
    with open(save_path_graph, 'w') as f:
        f.write('%d,%d,%d\n' % (size[0], size[1], N))
        for position in positions:
            f.write('%d,%d ' % (int(position[0]), int(position[1])))
        for flag in connections.flatten():
            f.write('%d' % int(flag))


def preprocess_ssm(ssm, semantic_order, ignore_label=-1):
    """对语义分割图进行预处理，对通过斜边相邻但不联通的像素填充，以节省素描图表征数据量并保证区域的闭合性"""
    ssm_copied = ssm.copy()
    # 获得所有标签
    all_labels = np.unique(ssm_copied).tolist()
    all_labels.remove(ignore_label)
    # 逐个考虑各个标签
    for label in semantic_order:
        if label not in all_labels:
            continue
        label_mask = ssm_copied == label
        region4, region_count4 = measure.label(label_mask, background=0, connectivity=1, return_num=True)
        region8, region_count8 = measure.label(label_mask, background=0, connectivity=2, return_num=True)
        if region_count8 == region_count4:  # 区域数目相同，则不存在这样的像素点
            continue
        for region_id8 in range(1, region_count8 + 1):  # 考察 8 邻域区域中的每个区域
            region_mask = region8 == region_id8
            region_id4 = region4[region_mask]  # 对应区域中各个像素点的 4 邻域区域序号
            if len(np.unique(region_id4)) == 1:  # 对应的区域 4 邻域区域数值相同
                continue
            diag_pos = get_diag(region_mask)  # 找到存在斜点模式的方块位置，返回方块右下角坐标
            for pos_idx in range(len(diag_pos[0])):  # 考察每个斜点块
                line, column = diag_pos[0][pos_idx], diag_pos[1][pos_idx]
                # 斜点块只存在一个区域标签（去除背景的 0 标签），则跳过
                if len(np.unique(region4[line-1:line+1, column-1:column+1])) - 1 == 1:
                    continue
                # 否则在上方填充 1
                ssm_copied[line - 1, column - 1:column + 1] = label
    return ssm_copied


def find_central_point(map_array):
    # Step 1: 创建一个反转的布尔数组，True代表原来为False的区域

    # Step 2: 计算每个点到最近False点的距离
    distance = distance_transform_edt(map_array)

    # Step 3: 找到距离最大的点
    max_distance_index = np.unravel_index(np.argmax(distance), distance.shape)

    # 返回该点的坐标
    return max_distance_index