"""
随机点云删除攻击（CVPR 2019 论文实验基准方法之一）。

模拟 LiDAR 传感器物理失效场景:
  - 物体遮挡导致部分激光束无法返回
  - 远距离/低反射率表面导致的信号丢失
  - 传感器稀疏采样模式

攻击机制:
  按 severity 比例随机丢弃点云中的点，使检测器因几何信息缺失而漏检。

  severity=0.0  → 保留所有点
  severity=0.3  → 随机丢弃 30% 的点
  severity=0.5  → 随机丢弃 50% 的点
  severity=1.0  → 丢弃所有点（极值情况）

参考: Xiang, Qi, Li — "Generating 3D Adversarial Point Clouds" (CVPR 2019)
"""

import torch
import numpy as np
from .base import BaseAttacker


class DropAttacker(BaseAttacker):
    """随机点云删除攻击。

    按 severity 比例随机丢弃点，模拟 LiDAR 传感器信号丢失。
    论文中作为黑盒攻击的基准方法之一。
    """

    def forward(self, data_dict):
        points = data_dict['points']
        is_numpy = isinstance(points, np.ndarray)
        if is_numpy:
            points = torch.from_numpy(points).float()

        # 生成保留掩码: 以概率 (1 - severity) 保留每个点
        keep_ratio = 1.0 - self.severity
        keep_mask = torch.rand(points.shape[0]) < keep_ratio

        # 保留至少 1% 或 100 个点，确保体素化不崩溃
        min_keep = max(100, int(points.shape[0] * 0.01))
        if keep_mask.sum() < min_keep:
            alive = torch.where(keep_mask)[0]
            extra_needed = min_keep - len(alive)
            dead = torch.where(~keep_mask)[0]
            resurrect = dead[torch.randperm(len(dead))[:extra_needed]]
            keep_mask[resurrect] = True

        result = points[keep_mask]
        data_dict['points'] = result.cpu().numpy() if is_numpy else result

        return data_dict
