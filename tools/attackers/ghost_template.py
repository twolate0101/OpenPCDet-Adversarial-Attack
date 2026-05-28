"""
幽灵模板注入攻击（黑盒插入，基于物理可实现性思想）。

攻击机制:
  借鉴 KITTI 数据增强中 GT-Sampling（复制粘贴真实目标）的技术，
  将其反转用于攻击：在点云的空旷区域注入预存的目标模板，制造假阳性。

  与 Spawn/Scatter/Object 的区别:
    Spawn/Scatter/Object: 白盒，用梯度优化生成对抗点簇（数字幻觉）
    GhostTemplate: 黑盒，用真实目标模板复制粘贴（物理可实现）

  物理可实现性:
    模板来自真实目标的几何形状，可在物理世界中用 3D 打印或投影仪复现。
    这是 Robust3DOD 框架对攻击的核心要求之一。

  severity 映射:
    severity=0.1  -> 注入 1 个模板
    severity=0.3  -> 注入 3 个模板
    severity=0.5  -> 注入 5 个模板
    severity=1.0  -> 注入 10 个模板

理论来源:
  - Robust3DOD 物理可实现性要求
  - KITTI GT-Sampling 数据增强技术的攻击化应用
  - 对抗攻击中的"对抗补丁"(Adversarial Patch) 思想
"""

import numpy as np
import torch
from .base import BaseAttacker


class GhostTemplateAttacker(BaseAttacker):
    """幽灵模板注入攻击。

    在点云空旷区域注入预存的目标模板，制造假阳性检测。
    不需要模型梯度，属于黑盒攻击。
    """

    def __init__(self, severity=0.3, **kwargs):
        """
        Args:
            severity: 注入模板数量的缩放因子 (0.0 ~ 1.0)
        """
        super().__init__(severity, **kwargs)
        self.templates = self._create_templates()

    def _create_templates(self):
        """生成 Car / Pedestrian / Cyclist 的简化 3D 表面点云模板。

        用长方体表面均匀采样模拟真实目标的点云分布。
        每个模板包含 (K, 3) 的 xyz 坐标。
        """
        templates = []

        # Car: ~4.0 x 1.6 x 1.5 m, 表面采样 500 点
        car_pts = self._sample_box_surface(
            size=(4.0, 1.6, 1.5), n_points=500
        )
        templates.append(('Car', car_pts))

        # Pedestrian: ~0.6 x 0.6 x 1.7 m, 表面采样 200 点
        ped_pts = self._sample_box_surface(
            size=(0.6, 0.6, 1.7), n_points=200
        )
        templates.append(('Pedestrian', ped_pts))

        # Cyclist: ~1.7 x 0.6 x 1.7 m, 表面采样 300 点
        cyc_pts = self._sample_box_surface(
            size=(1.7, 0.6, 1.7), n_points=300
        )
        templates.append(('Cyclist', cyc_pts))

        return templates

    @staticmethod
    def _sample_box_surface(size, n_points):
        """在长方体表面均匀采样。

        Args:
            size: (dx, dy, dz) 长方体尺寸
            n_points: 采样点数
        Returns:
            points: (n_points, 3) xyz 坐标
        """
        dx, dy, dz = size
        # 六个面的面积
        areas = [dx * dy, dx * dy, dx * dz, dx * dz, dy * dz, dy * dz]
        total_area = sum(areas)
        # 按面积比例分配采样点数
        n_per_face = [max(1, int(n_points * a / total_area)) for a in areas]

        points = []
        for face_idx, n in enumerate(n_per_face):
            if face_idx == 0:  # bottom (z=0)
                pts = np.column_stack([
                    np.random.uniform(-dx / 2, dx / 2, n),
                    np.random.uniform(-dy / 2, dy / 2, n),
                    np.zeros(n)
                ])
            elif face_idx == 1:  # top (z=dz)
                pts = np.column_stack([
                    np.random.uniform(-dx / 2, dx / 2, n),
                    np.random.uniform(-dy / 2, dy / 2, n),
                    np.full(n, dz)
                ])
            elif face_idx == 2:  # front (y=-dy/2)
                pts = np.column_stack([
                    np.random.uniform(-dx / 2, dx / 2, n),
                    np.full(n, -dy / 2),
                    np.random.uniform(0, dz, n)
                ])
            elif face_idx == 3:  # back (y=dy/2)
                pts = np.column_stack([
                    np.random.uniform(-dx / 2, dx / 2, n),
                    np.full(n, dy / 2),
                    np.random.uniform(0, dz, n)
                ])
            elif face_idx == 4:  # left (x=-dx/2)
                pts = np.column_stack([
                    np.full(n, -dx / 2),
                    np.random.uniform(-dy / 2, dy / 2, n),
                    np.random.uniform(0, dz, n)
                ])
            else:  # right (x=dx/2)
                pts = np.column_stack([
                    np.full(n, dx / 2),
                    np.random.uniform(-dy / 2, dy / 2, n),
                    np.random.uniform(0, dz, n)
                ])
            points.append(pts)

        points = np.concatenate(points, axis=0)
        # 随机打乱并截取到目标数量
        np.random.shuffle(points)
        return points[:n_points].astype(np.float32)

    def _find_empty_position(self, points):
        """在点云中找到一个相对空旷的位置作为插入点。

        策略: 在远处 (x > 30m) 且横向居中的区域随机选点。
        这个区域通常是路面，点云稀疏。

        Args:
            points: (N, 4) 原始点云
        Returns:
            position: (3,) 插入位置 [x, y, z]
        """
        # 远处路面区域
        x = np.random.uniform(30, 60)
        y = np.random.uniform(-10, 10)
        z = -1.0  # 路面高度
        return np.array([x, y, z], dtype=np.float32)

    def forward(self, data_dict):
        """
        执行幽灵模板注入。

        流程:
          1. 读取原始 points (N, 4)
          2. 按 severity 计算注入模板数量
          3. 随机选择模板类型（Car/Ped/Cyc）
          4. 找空旷位置，平移模板到该位置
          5. 拼接模板点云到原始 points

        Args:
            data_dict: 含 data_dict['points'] 的数据字典
                       格式: numpy (N, 4) [x, y, z, intensity]
                              或 tensor (N, 4)
        Returns:
            修改后的 data_dict（points 数量增加）
        """
        points = data_dict['points']
        is_numpy = isinstance(points, np.ndarray)
        if not is_numpy:
            points = points.cpu().numpy()

        # severity<=0 按约定表示"不攻击"
        if self.severity <= 0:
            return data_dict

        # 注入模板数量: severity * 10, 至少 1 个
        n_templates = max(1, int(self.severity * 10))

        new_segments = [points]
        default_intensity = 0.3  # 默认反射率

        for _ in range(n_templates):
            # 随机选模板
            name, template_pts = self.templates[np.random.randint(len(self.templates))]

            # 找空旷位置
            position = self._find_empty_position(points)

            # 平移模板
            placed = template_pts + position  # (K, 3)

            # 添加默认 intensity 列
            intensity_col = np.full(
                (len(placed), 1), default_intensity, dtype=np.float32
            )
            placed_full = np.hstack([placed, intensity_col])  # (K, 4)

            new_segments.append(placed_full)

        result = np.concatenate(new_segments, axis=0)
        data_dict['points'] = result if is_numpy else torch.from_numpy(result).float()

        return data_dict
