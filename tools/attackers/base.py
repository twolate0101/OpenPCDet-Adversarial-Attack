"""
攻击器基类 — 所有对抗攻击方法的统一接口。

所有攻击器继承此基类并实现 forward(data_dict) 方法。
severity ∈ [0, 1] 统一控制攻击强度:
  0.0 = 无攻击（保留原始数据）
  1.0 = 最强攻击
"""

import torch


class BaseAttacker:
    """对抗攻击器基类。

    severity 控制攻击强度，具体含义由子类定义:
      - noise/drop: 噪声标准差 / 删除比例
      - pgd/perturb: L2 扰动预算
      - spawn/scatter/object: 插入的点/簇/物体数量
    """

    def __init__(self, severity=1.0, **kwargs):
        self.severity = severity

    def forward(self, data_dict):
        """
        核心攻击接口，子类必须重写。

        Args:
            data_dict: OpenPCDet 数据字典。
                       黑盒攻击阶段含 data_dict['points'] (N, 5) [batch_idx, x, y, z, intensity]。
                       白盒/插入攻击阶段含 data_dict['voxels'] (M, max_pts, 4) [x, y, z, intensity]。

        Returns:
            修改后的 data_dict
        """
        raise NotImplementedError("每个攻击方法必须实现 forward 逻辑")
