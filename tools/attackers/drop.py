import torch
from .base import BaseAttacker


class DropAttacker(BaseAttacker):
    """
    随机点云删除攻击 (CVPR 2019 论文核心方法之一)。

    模拟 LiDAR 传感器被遮挡、反射丢失或远距离稀疏采样等场景：
    按 severity 比例随机丢弃点云中的点，使检测器因信息缺失而漏检。

    severity=0.0  → 保留所有点
    severity=0.3  → 随机丢弃 30% 的点
    severity=0.5  → 随机丢弃 50% 的点
    severity=1.0  → 丢弃所有点
    """

    def forward(self, data_dict):
        points = data_dict['points']

        # 生成保留掩码：概率 (1 - severity) 保留每个点
        keep_ratio = 1.0 - self.severity
        keep_mask = torch.rand(points.shape[0], device=points.device) < keep_ratio

        # 至少保留一个点（确保后续体素化不崩溃）
        if keep_mask.sum() == 0:
            keep_mask[0] = True

        data_dict['points'] = points[keep_mask]

        n_total = points.shape[0]
        n_kept = keep_mask.sum().item()
        print(f"[Drop] {n_total} → {n_kept} points ({n_kept / n_total * 100:.1f}% kept)")

        return data_dict
