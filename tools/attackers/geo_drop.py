"""
几何感知点云删除攻击（比随机删除更有效的定向攻击）。

攻击机制:
  随机删除（DropAttacker）无差别地丢弃点，效率低下。
  本攻击器基于 3D 目标检测的核心依赖——目标区域的点云密度——进行定向破坏：
  计算点云质心，优先删除距质心近的点（即目标表面附近的高密度区域），
  等价于削弱目标的几何特征表达，使检测器因关键区域信息缺失而漏检。

  与 DropAttacker 的对比:
    DropAttacker:  随机删 50% 的点 → 可能只删了远处无用的背景点
    GeoDropAttacker: 定向删 50% 的点 → 优先删除目标物体表面的关键点

  severity 映射:
    severity=0.1  → 删除距质心最近的 10% 的点
    severity=0.5  → 删除距质心最近的 50% 的点
    severity=1.0  → 删除距质心最近的 90% 的点

参考: Xiang, Qi, Li — "Generating 3D Adversarial Point Clouds" (CVPR 2019)
"""

import torch
import numpy as np
from .base import BaseAttacker


class GeoDropAttacker(BaseAttacker):
    """几何感知点云删除攻击。

    优先删除距点云质心近的点（目标区域），比随机删除更有效地破坏检测。
    不需要模型梯度，属于黑盒攻击。
    """

    def forward(self, data_dict):
        """
        对点云执行几何感知删除。

        流程:
          1. 计算所有点的质心 (x, y, z 均值)
          2. 计算每个点到质心的欧氏距离
          3. 按 severity 比例删除距离最近的点（目标区域）
          4. 保留剩余的点（背景区域）

        Args:
            data_dict: 含 data_dict['points'] 的数据字典
                       格式: tensor (N, 4) [x, y, z, intensity]
                              或 numpy (N, 4)

        Returns:
            修改后的 data_dict（points 数量减少）
        """
        points = data_dict['points']
        is_numpy = isinstance(points, np.ndarray)
        if is_numpy:
            points = torch.from_numpy(points).float()

        num_pts = points.shape[0]
        if num_pts == 0:
            return data_dict

        # 计算删除/保留数量
        drop_count = int(num_pts * self.severity)
        keep_count = num_pts - drop_count

        # 保留至少 1% 或 100 个点，确保体素化不崩溃
        min_keep = max(100, int(num_pts * 0.01))
        keep_count = max(keep_count, min_keep)

        # 计算每个点到质心的距离
        centroid = points[:, :3].mean(dim=0)  # (3,)
        distances = torch.norm(points[:, :3] - centroid, dim=1)  # (N,)

        # 保留距离远的点（删除距离近的目标区域点）
        _, keep_indices = distances.topk(keep_count, largest=True)

        result = points[keep_indices]
        data_dict['points'] = result.cpu().numpy() if is_numpy else result

        print(f"[GeoDrop] {num_pts} → {keep_count} 点 "
              f"({keep_count / num_pts * 100:.1f}% 保留, "
              f"删除距质心最近的 {drop_count} 个点)")

        return data_dict
