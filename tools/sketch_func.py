"""functions used for sketch generation with SAR Sketch"""
import numpy as np
import cv2
from typing import List
from collections import Counter
from skimage import measure
from tools.utils import save_graph_compressed, rearrange_graph
from tools.base_utils import ssm2boundary, get_mIoU, get_semantic_orders
from tools.sketch_utils import split_path, get_line_points, get_degree8, get_degree4, del_stick, find_neighbor
from tools.sketch_utils import process_clip, preprocess_ssm, get_ssm_colored_cityscapes
from tools.sketch_utils import find_central_point



class BranchMap:
    """
    类：branch图，二维矩阵，大小与图像大小相同，各个像素点标注其属于分支点序号（从1开始）
    """

    def __init__(self, h, w):
        self.map = np.zeros([h, w], dtype='uint')
        self.branch_count = 0
        self.height = h
        self.width = w

    def set_branch(self, pos, branch_id=-1):
        """
    设置指定位置的branch序号
        :param pos: 需要设定的位置，列表，列表元素为长度为2的数组或元组
        :param branch_id: 设定的branch序号，从 1 开始，背景为 0
        """
        if branch_id == -1:
            branch_id = self.branch_count + 1
        assert pos.__class__ == list
        assert len(pos[0]) == 2
        assert branch_id == self.branch_count + 1
        for i, j in pos:
            assert 0 <= i < self.height
            assert 0 <= j < self.width
            self.map[i, j] = branch_id
        self.branch_count += 1


class Branch:
    """
    分支类，记录一个分支的起点、终点、中间节点以及其他数据
    """

    def __init__(self, pos=None, threshold=1):
        if pos is None:
            pos = [0, 0]
        self.start_pos = pos  # 起点
        self.end_pos = []  # 终点
        self.mid_pos = []  # 中间节点，不包含起点和终点
        self.path = [pos]  # 途径的像素点位置，包含起点和终点
        self.section_mark = [1]  # 标注途径像素点所属的section序号，从1开始。中间节点的序号从属上一节
        self.path_length = 1  # 途经像素点的数量，即self.path的长度
        self.section_count = 1  # 节数，即中间节点的数量+1
        self.distances = [0]  # 记录各个途经点到其所属section线段的距离，长度等于path的长度
        self.max_distance = 0  # distance的最大值
        self.threshold = threshold
        if pos.__class__ == list and pos[0].__class__ == list:
            if len(pos[0]) == 2:  # 初始化参数是坐标列表，直接对其进行生成
                self.buildup(pos)
                # 如果该分支为环，且中间点数小于 2，则重新安排中间点，以保持环结构的有效性
                self.process_loop()

    def grow(self, path, distances, max_distance):
        """
        用新的一组点对当前分支进行生长，即增加新的section
        :param path: 二维数组列表，section上各点的位置（不包含上一个section的终点以避免重复）
        :param distances: 列表，新的section上各个点到这个section线段的距离
        :param max_distance: 该section各点到线段距离的最大值
        """
        assert distances.__class__ == list
        if self.path_length == 1:  # 第一个section，则新加入点的section标记为1
            section_mark = 1
        else:
            self.section_count += 1
            section_mark = self.section_count  # 否则标记为section数量+1
        self.mid_pos += [path[-1]]  # 将新section的最后一个点加入到中间节点列表
        self.path += path  # 将新section的各个点加入到path中
        self.section_mark += [section_mark] * len(path)  # 记录path各个点的section序号
        self.distances += distances  # 记录path各个点到section线段的距离
        self.max_distance = max(self.max_distance, max_distance)
        self.path_length += len(path)

    def end(self):
        """
        结束当前分支
        """
        self.end_pos = self.mid_pos.pop(len(self.mid_pos) - 1)  # 将mid_pos的最后一个点作为end_pos

    def buildup(self, path):
        """
        由所给path构建当前branch
        :param path: 路径上所有点的坐标（序列）列表
        """
        self.__init__(path[0], self.threshold)
        split_indexes, distances, distances_max = split_path(path, threshold=self.threshold)
        start_index = 1
        for split_index_index in range(len(split_indexes)):
            split_index = split_indexes[split_index_index]
            self.grow(path[start_index:split_index + 1], distances[start_index:split_index + 1],
                      distances_max[split_index_index])
            start_index = split_index + 1
        self.grow(path[start_index:], distances[start_index:], distances_max[-1])
        self.end()

    def process_loop(self):
        """当该分支为环，且中间点数小于 2，则重新采样中间点，以保持环结构"""
        if self.start_pos == self.end_pos and len(self.mid_pos) < 2:
            point_count = len(self.path)  # 总点数，为环上点数 + 1
            sample_idx = np.linspace(start=0, stop=point_count, num=4).astype('int')
            self.mid_pos = [self.path[idx] for idx in sample_idx[1:-1]]
            self.section_count = 3
            self.section_mark = ([0] * (sample_idx[1] + 1)
                                 + [1] * (sample_idx[2] - sample_idx[1])
                                 + [2] * (sample_idx[3] - sample_idx[2]))


class BranchRecorder:
    """
    分支记录器，记录一个素描图中所有的分支
    """

    def __init__(self):
        self.branches: List[Branch] = []
        self.count = 0

    def add_branch(self, branch: Branch):
        """
    新增一个分支
        """
        self.branches.append(branch)
        self.count += 1

    def arrange_branch(self):
        """按照规则为各分支分配可识别、可复现的序号"""
        all_points, index_point_branch = [], []  # 所有分支的拟合点，及拟合点所属分支的序号
        for branch_index, branch in enumerate(self.branches):
            for point in [branch.start_pos, branch.end_pos] + branch.mid_pos:
                all_points.append(point)
            index_point_branch += [branch_index] * (len(branch.mid_pos) + 2)
        max_line, max_column = np.max(np.array(all_points), axis=0)
        next_index = 0  # 下一个被分配的序号
        indexes = [-1] * self.count  # 每个 branch 的序号，初始化为 -1（无效值）
        # 对分支上的点进行排序，按照总体从上到下、同行从左到右的顺序排列
        sorted_ids = sorted(range(len(all_points)),
                            key=lambda k: all_points[k][0] + all_points[k][1] / (max_column + 1))
        all_points = [all_points[i] for i in sorted_ids]
        index_point_branch = [index_point_branch[i] for i in sorted_ids]
        # 依次访问各个分支拟合点并执行序号分配
        while all_points:  # 当有剩余待访问点时取出第一个点考察
            candidate_point = all_points.pop(0)  # 找到第一个考察点
            candidate_branches = [index_point_branch.pop(0)]  # 该考察点所在分支序号
            while candidate_point in all_points:
                index_ = all_points.index(candidate_point)
                all_points.pop(index_)
                candidate_branches.append(index_point_branch.pop(index_))
            candidate_branches = list(set(candidate_branches))
            # 如果只有一个分支包含考察点，则直接为该分支分配序号
            if len(candidate_branches) == 1:
                indexes[candidate_branches[0]] = next_index
                next_index += 1
            # 如果多个分支包含考察点，则考察这些分支下一个点的位置
            else:
                next_point_pos = []
                for index_ in candidate_branches:
                    next_point_pos.append(index_point_branch.index(index_))
                priors = np.argsort(next_point_pos)  # 按照序号由小到大排列，即为分支分配序号的优先级
                for prior_index in priors:
                    branch_id = candidate_branches[prior_index]  # 选出优先级最高的分支序号
                    indexes[branch_id] = next_index  # 分配序号
                    next_index += 1
            for branch_id in candidate_branches:  # 将已分配序号分支上的其余点从列表中移除
                while branch_id in index_point_branch:
                    branch_pos = index_point_branch.index(branch_id)
                    index_point_branch.pop(branch_pos)
                    all_points.pop(branch_pos)
        return indexes

    def generate_sketch(self, H, W, return_times=False):
        """由所记录的分支抽象和拟合结果恢复素描图"""
        edge_sketch = np.zeros((H, W, 3), dtype=np.uint8)
        times = np.zeros((H, W), dtype=np.uint8)
        for branch in self.branches:
            node_pos = [branch.start_pos] + branch.mid_pos + [branch.end_pos]
            for section_index in range(len(node_pos) - 1):

                start_node, end_node = node_pos[section_index], node_pos[section_index + 1]
                length = max(abs(start_node[0] - end_node[0]), abs(start_node[1] - end_node[1])) + 1
                fill_line = np.round(np.linspace(start_node[0], end_node[0], length))
                fill_column = np.round(np.linspace(start_node[1], end_node[1], length))
                for i in range(length):
                    edge_sketch[int(fill_line[i]), int(fill_column[i]), :] = [1, 1, 1]
                edge_sketch[start_node[0], start_node[1], :] = [0, 1, 0]
                edge_sketch[end_node[0], end_node[1], :] = [0, 1, 0]
            edge_sketch[branch.start_pos[0], branch.start_pos[1], :] = [1, 0, 0]
            edge_sketch[branch.end_pos[0], branch.end_pos[1], :] = [1, 0, 0]
            for path in branch.path:
                times[path[0], path[1]] += 1
        if not return_times:
            return edge_sketch
        else:
            return edge_sketch, times

    def generate_branch_map(self, H, W):
        """由所记录的分支抽象和拟合结果恢复素描图，其中各个素描点数值为所属分支的序号（从 1 开始，以与背景区分开）"""
        branch_map = BranchMap(H, W)
        for branch_index, branch in enumerate(self.branches):
            node_pos = [branch.start_pos] + branch.mid_pos + [branch.end_pos]
            fill_points = []
            for section_index in range(len(node_pos) - 1):
                start_node, end_node = node_pos[section_index], node_pos[section_index + 1]
                length = max(abs(start_node[0] - end_node[0]), abs(start_node[1] - end_node[1])) + 1
                fill_line = np.round(np.linspace(start_node[0], end_node[0], length)).astype("int")
                fill_column = np.round(np.linspace(start_node[1], end_node[1], length)).astype("int")
                fill_points += np.stack([fill_line, fill_column]).T.tolist()
            branch_map.set_branch(fill_points, branch_index + 1)
        return branch_map

    def generate_regions_seed(self, ssm: np.ndarray, indexes: list, ignore_label=19):
        """
        根据语义分割图、素描图和分支序号获得语义区域的表征，包括围成语义区域的分支序号和语义标签
        """
        H, W = ssm.shape[:2]
        SSM_regions = []
        # 设置分支图，将各个像素点所属分支序号（原始记录）标记在图中
        branch_map = self.generate_branch_map(H, W)
        # 设置有色素描图，分支的端点为红色，其余拟合点为绿色
        sketch_colored = self.generate_sketch(H, W)
        # 设置填充图，记录各个点是否被访问过
        flags_filled = (branch_map.map > 0).astype("int")  # 设置素描线为已被填充，作为区域生长的分割
        # 种子生长
        while not flags_filled.all():  # 存在未被填充的区域
            branch_indexes, label = [], -1  # 记录该区域围成的分支序号和语义标签
            region_points = []  # 记录该封闭区域内包含的点（不包括边界点）
            pixel_sample = np.where(flags_filled == 0)
            pixel_sample = (pixel_sample[0][0], pixel_sample[1][0])  # 采样一个未被填充且不在素描线上的点
            candidate_pixels = [pixel_sample]  # 维护待访问点队列
            while candidate_pixels:  # 当待访问队列不为空
                # 取出种子点并填充
                seed_point = candidate_pixels.pop(0)  # 从队列中取出一个点作为种子
                region_points.append(seed_point)  # 记录该点
                flags_filled[seed_point[0], seed_point[1]] = 1  # 填充种子点
                # 考察四邻域邻点
                shifts_l, shifts_c = (-1, 0, 0, 1), (0, -1, 1, 0)
                for shift_l, shift_c in zip(shifts_l, shifts_c):  # 考察每个邻点的偏移
                    neighbor = (seed_point[0] + shift_l, seed_point[1] + shift_c)  # 邻点坐标
                    if neighbor[0] < 0 or neighbor[0] > H - 1 or neighbor[1] < 0 or neighbor[1] > W - 1:
                        continue  # 该邻点位于边界以外
                    if flags_filled[neighbor[0], neighbor[1]] == 0:  # 该点未被访问且不是拟合分支上的点
                        if neighbor not in candidate_pixels:
                            candidate_pixels.append(neighbor)  # 将该点加入待访问点
                        else:
                            continue
                    elif branch_map.map[neighbor[0], neighbor[1]] > 0 and \
                            list(sketch_colored[neighbor[0], neighbor[1]]) == [1, 1, 1]:  # 邻点是分支上的点且为中间点
                        branch_indexes.append(int(branch_map.map[neighbor[0], neighbor[1]] - 1))  # 记录该邻点所属分支的序号
                    else:  # 其余情况不做处理
                        continue
            # 整合围成该区域的分支序号
            branch_indexes = list(set(branch_indexes))  # 保留唯一的序号
            branch_numbers = [indexes[i] for i in branch_indexes]  # 将分支的序号转化为重新分配的标签
            labels_region = [ssm[p[0], p[1]] for p in region_points]  # 该区域内所有点的标签
            label_region = max(set(labels_region), key=labels_region.count)  # 出现最多的标签作为该区域的标签
            if label_region != ignore_label:
                SSM_regions.append((sorted(branch_numbers), label_region))  # 记录该区域的包围分支序号和区域的语义标签
        return SSM_regions

    def generate_regions_connection(self, ssm: np.ndarray, indexes: list, ignore_label=19):
        """
        根据语义分割图、素描图和分支序号获得语义区域的表征，包括围成语义区域的分支序号和语义标签
        """
        H, W = ssm.shape[:2]
        SSM_regions = []
        # 设置分支图，将各个像素点所属分支序号（原始记录）标记在图中
        branch_map = self.generate_branch_map(H, W)
        # 设置有色素描图，分支的端点为红色，其余拟合点为绿色
        sketch_colored, times = self.generate_sketch(H, W, return_times=True)
        inner_branch_map = sketch_colored.sum(axis=2) == 3  # 各像素位于分支内标记为 1，其余为 0
        # 连通域划分，获得连通区域数目和各像素点所属区域（素描点为 0，不属于任何连通域）
        # regions_count, regions_map = branch_count_2path(branch_map.map == 0, neighbor_range=4)
        regions_map, regions_count = measure.label(branch_map.map == 0, background=0, return_num=True, connectivity=1)
        for region_idx in range(regions_count):
            # 获取区域边界像素
            region_points = regions_map == region_idx + 1  # 区域内所有点，通过四邻域连通
            # 取 region_points 中值为 True 的坐标
            seed_point = find_central_point(region_points)
            # if region_points[56, 397]:
            #     print("Watch!")
            labels_region = ssm[region_points].tolist()  # 该区域内所有点的标签
            label_region = max(set(labels_region), key=labels_region.count)  # 出现最多的标签作为该区域的标签
            if label_region == ignore_label:
                continue  # 忽略区域
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype='uint8')  # 4 邻域膨胀核
            region_dilated4 = cv2.dilate(region_points.astype('uint8'), kernel)  # 进行膨胀
            border_map4 = region_dilated4 != region_points  # 区域边界为 1，其余为 0
            kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='uint8')  # 8 邻域膨胀核
            region_dilated8 = cv2.dilate(region_points.astype('uint8'), kernel)  # 进行膨胀
            border_map8 = region_dilated8 != region_points  # 区域边界为 1，其余为 0
            # 将四邻域边界点作为内点所属的所有分支视为围成该区域的分支
            branch_indexes = list(set(list(branch_map.map[inner_branch_map * border_map4] - 1)))  # 四邻域点所属的所有分支
            # 存在分支完全位于边界上
            for branch_id, branch in enumerate(self.branches):
                if border_map8[branch.start_pos[0], branch.start_pos[1]] \
                    and border_map8[branch.end_pos[0], branch.end_pos[1]] \
                        and all([border_map8[pos[0], pos[1]] for pos in branch.mid_pos]):
                    branch_indexes.append(branch_id)
            branch_indexes = list(set(branch_indexes))  # 保留唯一的序号
            # 删除对区域包围没有影响的分支
            # 第一步：计算原包围区域与实际区域的交并比
            plotted_ori = self.plot_branch(branch_indexes, W, H)
            # 生成连通图
            region_map, region_count = measure.label(1 - plotted_ori,
                                                     background=0,
                                                     connectivity=1,
                                                     return_num=True)  # 按四邻域划分连通分支
            region_seed = region_map == region_map[seed_point[0], seed_point[1]]
            IoU_ori = np.sum(region_seed * region_points) / np.sum(region_seed + region_points)
            del_branches = []
            for candidate_branch_id in branch_indexes:  # 考虑每一个分支
                plotted = self.plot_branch([index for index in branch_indexes if index not in [candidate_branch_id] + del_branches],
                                           W, H)
                # 生成连通图
                region_map, region_count = measure.label(1 - plotted,
                              background=0,
                              connectivity=1,
                              return_num=True)  # 按四邻域划分连通分支
                region_seed = region_map == region_map[seed_point[0], seed_point[1]]
                IoU = np.sum(region_seed * region_points) / np.sum(region_seed + region_points)
                if IoU >= IoU_ori:  # 去除分支后 IoU 没有降低，表示可以去除
                    del_branches.append(candidate_branch_id)
            branch_indexes = [bid for bid in branch_indexes if bid not in del_branches]
            # 整合围成该区域的分支序号
            if not branch_indexes:  # 空区域
                continue
            branch_numbers = [indexes[i] for i in branch_indexes]  # 将分支的序号转化为重新分配的标签
            SSM_regions.append((sorted(branch_numbers), label_region))  # 记录该区域的包围分支序号和区域的语义标签
        return SSM_regions

    def plot_branch(self, branch_indexes, W, H):
        plotted = np.zeros((H, W), dtype="bool")
        sketch_points = np.zeros([0, 2]).astype('int')
        for plot_branch_id in branch_indexes:
            fit_points = ([self.branches[plot_branch_id].start_pos] +
                          self.branches[plot_branch_id].mid_pos +
                          [self.branches[plot_branch_id].end_pos])  # 分支上的拟合点
            for plot_line_idx in range(len(fit_points) - 1):
                line_points = get_line_points(fit_points[plot_line_idx], fit_points[plot_line_idx + 1])
                sketch_points = np.concatenate((sketch_points, line_points), axis=0)
            plotted[sketch_points[:, 0], sketch_points[:, 1]] = 1
        return plotted

    def generate_regions_makeup(self, ssm: np.ndarray, indexes: list, ignore_label=19):
        """
        根据语义分割图、素描图和分支序号获得语义区域的表征，包括围成语义区域的分支序号和语义标签。
        先挑出一定正确的分支，再从剩余分支中选适当的分支来补足
        """
        H, W = ssm.shape[:2]
        SSM_regions = []
        # 设置分支图，将各个像素点所属分支序号（原始记录）标记在图中
        branch_map = self.generate_branch_map(H, W)
        # 设置有色素描图，分支的端点为红色，其余拟合点为绿色
        sketch_colored, times = self.generate_sketch(H, W, return_times=True)
        inner_branch_map = sketch_colored.sum(axis=2) == 3  # 各像素位于分支内标记为 1，其余为 0
        # 连通域划分，获得连通区域数目和各像素点所属区域（素描点为 0，不属于任何连通域）
        regions_map, regions_count = measure.label(branch_map.map == 0, background=0, return_num=True, connectivity=1)
        for region_idx in range(regions_count):
            boundary = np.zeros_like(inner_branch_map)
            # 获取区域边界像素
            region_points = regions_map == region_idx + 1  # 区域内所有点
            if region_points[70, 300]:
                print("Watch!")
            labels_region = ssm[region_points].tolist()  # 该区域内所有点的标签
            label_region = max(set(labels_region), key=labels_region.count)  # 出现最多的标签作为该区域的标签
            if label_region == ignore_label:
                continue  # 忽略区域
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype='uint8')  # 4 邻域膨胀核
            region_dilated4 = cv2.dilate(region_points.astype('uint8'), kernel)  # 进行膨胀
            border_map4 = region_dilated4 != region_points  # 区域边界为 1，其余为 0
            kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='uint8')  # 8 邻域膨胀核
            region_dilated8 = cv2.dilate(region_points.astype('uint8'), kernel)  # 进行膨胀
            border_map8 = region_dilated8 != region_points  # 区域边界为 1，其余为 0
            branch_indexes = []
            # STEP1: 记录完全位于边界上的分支
            for branch_id, branch in enumerate(self.branches):
                if border_map8[branch.start_pos[0], branch.start_pos[1]] \
                    and border_map8[branch.end_pos[0], branch.end_pos[1]] \
                        and all([border_map8[pos[0], pos[1]] for pos in branch.mid_pos]):
                    branch_indexes.append(branch_id)
            # STEP2: 从剩余分支中挑选，补足区域缺口
            sketch_points = np.zeros([0, 2]).astype('int')
            for branch_id in branch_indexes:  # 找到各个分支绘制的点
                fit_points = ([self.branches[branch_id].start_pos] +
                              self.branches[branch_id].mid_pos +
                              [self.branches[branch_id].end_pos])  # 分支上的拟合点
                for plot_line_idx in range(len(fit_points) - 1):
                    line_points = get_line_points(fit_points[plot_line_idx], fit_points[plot_line_idx + 1])
                    sketch_points = np.concatenate((sketch_points, line_points), axis=0)
            boundary[sketch_points[:, 0], sketch_points[:, 1]] = 1
            gap_map = border_map4 * (1- boundary)  # 属于四邻域边界但未被分支覆盖的边界点图
            # todo: unfinished
            branch_indexes = list(set(branch_indexes))  # 保留唯一的序号
            # 整合围成该区域的分支序号
            if not branch_indexes:  # 空区域
                continue
            branch_numbers = [indexes[i] for i in branch_indexes]  # 将分支的序号转化为重新分配的标签
            SSM_regions.append((sorted(branch_numbers), label_region))  # 记录该区域的包围分支序号和区域的语义标签
        return SSM_regions

    def recover_ssm(self, regions, indexes: list, H: int, W: int, semantic_orders=None, ignore_label=19):
        """
        根据语义区域的包围分支序号和标签恢复语义分割图
        """
        if semantic_orders is None:
            semantic_orders = list(set([region[1] for region in regions]))
        ssm = np.ones([H, W], dtype='int') * ignore_label
        branches_ids = [region[0] for region in regions]
        region_labels = [region[1] for region in regions]
        for label in semantic_orders[::-1]:  # 按照优先级填充区域
            while label in region_labels:
                # 查找一个区域，获取其包围分支序号
                region_idx = region_labels.index(label)
                region_labels.pop(region_idx)
                branch_ids = branches_ids.pop(region_idx)
                branch_ids = [indexes.index(id_) for id_ in branch_ids]  # 转化为本地分支序号
                # 通过连通域法找到位于这些分支包围区域内的点
                sketch_points = np.zeros([0, 2]).astype('int')
                for branch_id in branch_ids:  # 找到各个分支绘制的点
                    fit_points = ([self.branches[branch_id].start_pos] +
                                  self.branches[branch_id].mid_pos +
                                  [self.branches[branch_id].end_pos])  # 分支上的拟合点
                    for plot_line_idx in range(len(fit_points) - 1):
                        line_points = get_line_points(fit_points[plot_line_idx], fit_points[plot_line_idx + 1])
                        sketch_points = np.concatenate((sketch_points, line_points), axis=0)
                min_l, min_c = np.min(sketch_points, axis=0)
                max_l, max_c = np.max(sketch_points, axis=0)
                inner_mask = np.zeros([max_l - min_l + 1, max_c - min_c + 1]).astype("uint8")  # 只对局部处理，降低复杂度
                inner_mask[sketch_points[:, 0] - min_l, sketch_points[:, 1] - min_c] = 1  # 绘制素描点
                region_map, region_count = measure.label(1 - inner_mask,
                                                         background=0,
                                                         connectivity=1,
                                                         return_num=True)  # 按四邻域划分连通分支
                for region_label in range(1, region_count + 1):
                    if region_label in region_map[0, :] or \
                        region_label in region_map[-1, :] or \
                        region_label in region_map[:, 0] or \
                        region_label in region_map[:, -1]:
                        region_map[region_map == region_label] = 0  # 将贴边的区域清除
                region_label = -1
                for line in range(max_l - min_l + 1):
                    if (region_map[line, :]).any():  # 第一次出现标签即为该区域内的标签值
                        region_label = region_map[line, :][region_map[line, :] != 0][0]
                        break
                inner_mask = np.logical_or(inner_mask, region_map == region_label)  # 区域内部的点（包括边界）
                # 获得该多边形包围的内部点（包括边界），舍弃的用法
                # inner_points = get_inner_points(polygon)
                # for point in inner_points:
                #     ssm[point[0], point[1]] = label
                # 获取内部点的 mask（包括边界），舍弃的用法
                # inner_mask, start_l, start_c = get_inner_mask(polygon)  # 区域面积较大时，mask 效率更高
                # h, w = inner_mask.shape
                # ssm[start_l:start_l + h, start_c:start_c + w][inner_mask] = (
                #         label * (ssm[start_l:start_l + h, start_c:start_c + w][inner_mask] == 0))  # 将未赋值的位置赋予标签
                # 填充内部点
                fill_mask = ssm[min_l:max_l + 1, min_c:max_c + 1] == ignore_label  # 填充未被填充过的区域
                fill_mask = fill_mask * inner_mask  # 这些区域中填充区域内的点
                ssm[min_l:max_l + 1, min_c:max_c + 1][fill_mask] = label
        return ssm


def process_branch(edge, ignore_length=3, neighbor=4, threshold=1, abort_stick=False):
    """
    对素描图进行分支抽象。
    1. 计算各个点的4-邻域度
    2. 度大于等于3的视为分支节点，度为1的视为边缘端点，度为2的视为中间节点
    3. 对于每个分支节点，搜索其周围的分支，记录分支的端点（端点是分支节点或边缘端点）以及拟合节点
        - 记录分支长度、分支节数、拟合误差
        - 记录分支上点的分支序号
    4. 对于没有被分支记录的点，视为孤立分支且对其进行逼近
    5. 所记录的分支长度需要超过ignore_length
    """
    h, w = edge.shape
    branch_map = BranchMap(h, w)  # 存储素描图中各个点所属的分支
    branch_recorder = BranchRecorder()  # 存储素描图中各个分支的信息
    visited_map = np.zeros([h, w], dtype='bool')  # 存储各个点的访问情况
    # kernel_degree4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='uint8')
    # kernel_degree8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype='uint8')
    # degree = conv(edge, kernel_degree4, mode='same') * edge if neighbor == 4 else conv(edge, kernel_degree8,
    #                                                                                    mode='same') * edge  # 4邻域度
    degree = get_degree4(edge) * edge if neighbor == 4 else get_degree8(edge) * edge
    # 清洗孤立的点，即度为0的边缘点
    edge[degree == 0] = 0
    # 清洗分叉线，即端点度为 1
    if abort_stick:
        edge = del_stick(edge, degree)
        degree = get_degree4(edge) * edge if neighbor == 4 else get_degree8(edge) * edge
    # 记录由分支节点出发的branch，特征为：端点一定有一个是分支节点
    cross_nodes = np.where(degree >= 3)  # 分支节点的序号
    for node_index in range(len(cross_nodes[0])):
        line, column = cross_nodes[0][node_index], cross_nodes[1][node_index]  # 分支节点坐标
        visited_map[line, column] = 1
        neighbors = find_neighbor(edge, degree, visited_map, [line, column], neighbor=neighbor,
                                  include_cross=True)  # 分支节点的4邻域坐标（未被访问的点或被访问过的其他分支点）
        for neighbor_index in range(len(neighbors)):
            neighbor_line, neighbor_column = neighbors[neighbor_index]  # 一个相邻点
            if visited_map[neighbor_line, neighbor_column] == 1:  # 未处理过的邻点已经访问过，说明是环，这是从环的另一个方向的访问（不会访问已访问过的交叉点）
                continue
            visited_map[neighbor_line, neighbor_column] = 1
            path = [[line, column], [neighbor_line, neighbor_column]]
            while degree[neighbor_line, neighbor_column] == 2:  # 继续寻找中间节点的下一个邻点
                neighbors_next = find_neighbor(edge, degree, visited_map, [neighbor_line, neighbor_column],
                                               neighbor=neighbor, include_cross=True)  # （未被访问的点或被访问过的其他分支点）
                if [line, column] in neighbors_next:  # 避免第一个邻点访问回起始交叉点
                    neighbors_next.pop(neighbors_next.index([line, column]))
                if len(neighbors_next) == 0:  # 中间节点度为2（未访问的邻点数为1）
                    # 不存在邻点，说明是一个环形，此时将该分支节点作为终点（该分支节点虽然是邻点但在上一步被去除）
                    neighbor_line, neighbor_column = line, column
                else:
                    neighbor_line, neighbor_column = neighbors_next[0]  # 邻点坐标
                if degree[neighbor_line, neighbor_column] in [1, 2]:
                    visited_map[neighbor_line, neighbor_column] = 1  # 只有下一个节点是端点或中间节点时，才将其标记为已访问
                path.append([neighbor_line, neighbor_column])  # 加入路径
            if len(path) > ignore_length or \
                    degree[path[0][0], path[0][1]] > 2 and degree[path[-1][0], path[-1][1]] > 2:  # 长度足够长或端点均为分支点
                branch_map.set_branch(path)  # 标记该分支的所有点的分支序号，序号自动设置
                branch_recorder.add_branch(Branch(path, threshold))
    # 记录孤立的条状branch，特征为：由两个端点和多个中间节点组成
    strip_nodes = np.where((degree == 1) * (visited_map == 0))  # 未访问的端点的序号
    for node_index in range(len(strip_nodes[0])):
        line, column = strip_nodes[0][node_index], strip_nodes[1][node_index]  # 端点坐标
        if visited_map[line, column] == 1:
            continue
        visited_map[line, column] = 1
        neighbors = find_neighbor(edge, degree, visited_map, [line, column], neighbor=neighbor)  # 端点的4邻域坐标
        assert len(neighbors) == 1
        neighbor_line, neighbor_column = neighbors[0]  # 相邻点
        visited_map[neighbor_line, neighbor_column] = 1
        path = [[line, column], [neighbor_line, neighbor_column]]
        while degree[neighbor_line, neighbor_column] == 2:  # 继续寻找中间节点的下一个邻点
            neighbors = find_neighbor(edge, degree, visited_map, [neighbor_line, neighbor_column],
                                      neighbor=neighbor)  # 中间节点的4邻域坐标
            assert len(neighbors) == 1  # 中间节点度为2（未访问的邻点数为1）
            neighbor_line, neighbor_column = neighbors[0]  # 一个相邻点
            visited_map[neighbor_line, neighbor_column] = 1  # 标记为已访问
            path.append([neighbor_line, neighbor_column])  # 加入路径
        if len(path) > ignore_length:
            branch_map.set_branch(path)  # 标记该分支的所有点的分支序号，序号自动设置
            branch_recorder.add_branch(Branch(path, threshold))
    # 记录孤立的环状branch，特征为：只由中间节点组成
    loop_nodes = np.where((degree == 2) * (visited_map == 0))  # 未访问的中间节点的序号
    for node_index in range(len(loop_nodes[0])):
        line, column = loop_nodes[0][node_index], loop_nodes[1][node_index]  # 任意一个点作为起始坐标
        if visited_map[line, column] == 1:
            continue
        visited_map[line, column] = 1
        neighbors = find_neighbor(edge, degree, visited_map, [line, column], neighbor=neighbor)  # 端点的4邻域坐标
        # try:
        #     assert len(neighbors) == 1
        # except AssertionError:
        #     print('Warning! Is here a start node of a loop?')
        neighbor_line, neighbor_column = neighbors[0]  # 相邻点
        visited_map[neighbor_line, neighbor_column] = 1
        path = [[line, column], [neighbor_line, neighbor_column]]
        end = 0
        while degree[neighbor_line, neighbor_column] == 2:  # 继续寻找中间节点的下一个邻点
            neighbors = find_neighbor(edge, degree, visited_map, [neighbor_line, neighbor_column],
                                      neighbor=neighbor)  # 中间节点的4邻域坐标
            if len(neighbors) == 0:  # 没有邻点时，说明回到了起点
                neighbors = [[line, column]]
                end = 1
            assert len(neighbors) == 1  # 中间节点度为2（未访问的邻点数为1）
            neighbor_line, neighbor_column = neighbors[0]  # 一个相邻点
            visited_map[neighbor_line, neighbor_column] = 1  # 标记为已访问
            path.append([neighbor_line, neighbor_column])  # 加入路径
            if end:
                break
        if len(path) > ignore_length:
            branch_map.set_branch(path)  # 标记该分支的所有点的分支序号，序号自动设置
            branch_recorder.add_branch(Branch(path, threshold))
    # 由所记录branch恢复素描图
    assert np.sum((visited_map == 0) * (degree <= 2) * (degree > 0)) == 0  # 不存在没有访问过的中间点和端点
    edge_sketch = branch_recorder.generate_sketch(h, w)
    return edge_sketch, branch_map, branch_recorder


def get_sketch(edge, neighbor=4, threshold_fit=1, return_base_edge=False, return_recorder=False,
               threshold_length=3, valid_shift=None, abort_stick=False):
    """
    对指定单像素边缘图像进行素描图抽象
    :param edge: 单像素二值化边缘，边缘部分为1，其余为0
    :param neighbor: 邻域搜索时基于4邻域或8邻域。算法会根据该设置首先对边缘进行修正。8邻域复杂度较高
    :param threshold_fit: 用于直线段拟合时的分段阈值。阈值越小，对原边缘的拟合越贴近
    :param return_base_edge: 是否返回基本边缘图像，即预处理后的边缘，与输入边缘可能不一致
    :param return_recorder: 是否返回分支记录的结果
    :param threshold_length: 长度阈值。只保留长度高于阈值的分支
    :param valid_shift: 有效的线条变换列表，含义参考 process_phase
    :param abort_stick: 若为True，则舍弃存在度为 1 的端点的分支
    """
    if valid_shift is None:
        valid_shift = [0, 1, 2, 3]
    # 预处理边缘
    edge_clipped = process_clip(edge, neighbor=neighbor, valid_shift=valid_shift)  # 对边缘图进行裁剪与修正
    # 获取素描线和素描分支
    edge_sketch, branch_map, branch_recorder = process_branch(edge_clipped, neighbor=neighbor, threshold=threshold_fit,
                                                              ignore_length=threshold_length,
                                                              abort_stick=abort_stick)  # 分支抽象
    # 所有分支的节点总数之和（充裕数），其中包含部分重复的分支
    total_knots_original = int(np.sum([b.section_count + 1 for b in branch_recorder.branches]))
    # 先以充裕数初始化节点坐标和连接矩阵数组
    positions = np.zeros([total_knots_original, 2], dtype='uint16')
    connections = np.zeros([total_knots_original, total_knots_original], dtype='bool')
    index_count = 0
    pt_index1 = -1  # 只是为了避免编译器警告
    # 根据素描分支抽象素描图，包括关键点和连接关系
    for branch in branch_recorder.branches:
        for pos_index in range(branch.section_count + 1):
            if pos_index == 0:  # 起始点
                if index_count > 0:
                    if np.sum((positions[:index_count, :] == branch.start_pos).all(1)):  # 起点位于现有端点中
                        index_pre = np.where((positions[:index_count, :] == branch.start_pos).all(1))[0][0]  # 之前序号的位置
                        pt_index1 = index_pre  # 端点1的序号
                        continue
                pt_index1 = index_count
                positions[index_count] = branch.start_pos
                index_count += 1  # 当前总序号+1
            elif 0 < pos_index < branch.section_count:  # 中间点
                positions[index_count] = branch.mid_pos[pos_index - 1]  # 中间节点加入节点列表中
                connections[pt_index1, index_count] = 1  # 记录连接情况
                pt_index1 = index_count
                index_count += 1
            else:
                assert pos_index == branch.section_count
                if np.sum((positions[:index_count, :] == branch.end_pos).all(1)):  # 终点位于现有端点中
                    index_pre = np.where((positions[:index_count, :] == branch.end_pos).all(1))[0][0]  # 之前序号的位置
                    pt_index2 = index_pre  # 端点2的序号
                else:
                    pt_index2 = index_count
                    positions[index_count] = branch.end_pos
                    index_count += 1  # 当前总序号+1
                connections[pt_index1, pt_index2] = 1
    if return_recorder:  # 返回分支抽象和重建的边缘图
        return positions[:index_count, :], connections[:index_count, :index_count], branch_recorder, edge_sketch
    if not return_base_edge:
        return positions[:index_count, :], connections[:index_count, :index_count], edge_sketch
    else:
        return positions[:index_count, :], connections[:index_count, :index_count], edge_sketch, edge_clipped


def compress_ssm(ssm: np.ndarray, threshold=2, ignore_label=19, class_count=19, semantic_priorities=None):
    """调用基本函数对语义分割图进行压缩"""
    # 根据语义区域面积排列优先级，优先保证小目标的完整性
    if semantic_priorities is None:
        semantic_priorities = get_semantic_orders(ssm, ignore_label=ignore_label)
    H, W = ssm.shape
    # 预处理语义分割图，使斜连但不连通的像素连通
    ssm_preprocessed = preprocess_ssm(ssm, semantic_priorities, ignore_label=ignore_label)
    # 根据语义分割图获取语义边界轮廓
    boundary = ssm2boundary(ssm_preprocessed, semantic_priorities, ignore_label=ignore_label)
    # 对语义边界轮廓进行分支抽象并执行直线段拟合。部分分支线对区域划分起到关键作用，因此不能根据分支长度进行舍弃
    points, connections, branch_object, edge_sketch = get_sketch(boundary,
                                                                 neighbor=4,
                                                                 threshold_fit=threshold,
                                                                 return_recorder=True,
                                                                 threshold_length=1,
                                                                 valid_shift=[1, 3],
                                                                 abort_stick=True)
    # 以分支抽象的格式保存压缩后的素描图
    # save_path = "/home/Users/dqy/Projects/SPIC/temp/compressed_branch.bin"
    # save_branch_compressed(branch_object, save_path, [H, W])
    # bpp_branch = os.path.getsize(save_path) * 8 / H / W
    # 以图格式保存压缩后的素描图
    graph = {"positions": points, "connections": connections}
    sketch_graph_rearranged = rearrange_graph(graph)
    bytes_ = save_graph_compressed(sketch_graph_rearranged, "", [H, W])
    bpp_sketch = len(bytes_) * 8 / H / W
    # 编码区域信息，包括围成区域的分支序号和区域的标签
    assigned_index = branch_object.arrange_branch()  # 为各个分支分配唯一的序号
    ssm_regions = branch_object.generate_regions_connection(ssm, assigned_index, ignore_label=ignore_label)
    bpp_region = sum([len(region[0]) * np.ceil(np.log2(branch_object.count)) \
                      + np.ceil(np.log2(class_count)) for region in ssm_regions]) / H / W
    # 恢复语义分割图
    ssm_recovered = branch_object.recover_ssm(ssm_regions, assigned_index, H, W, ignore_label=ignore_label,
                                              semantic_orders=semantic_priorities)
    # ssm_colored = get_ssm_colored_cityscapes(ssm_recovered)  # 为语义分割图填充颜色
    # 计算语义指标
    mIoU = get_mIoU(ssm, ssm_recovered, labels=semantic_priorities, ignore_label=ignore_label)
    return bpp_sketch + bpp_region, mIoU, ssm_recovered
