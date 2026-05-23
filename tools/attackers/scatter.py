import torch
from .base import BaseAttacker


class ScatterAttacker(BaseAttacker):
    """
    独立散点插入攻击 (论文 Section 3.3.1, scatter.py)。

    在 LiDAR 有效范围内随机生成互不关联的独立点，模拟攻击者注入
    无结构噪声以混淆检测器的特征提取。

    severity 控制散点数量:
      severity=0.2  → ~200 个散点
      severity=0.5  → ~500 个散点
      severity=1.0  → ~2000 个散点

    点坐标范围 (KITTI LiDAR 典型视野):
      x ∈ [0, 70]   前向
      y ∈ [-40, 40] 横向
      z ∈ [-3, 1]   高度

    与 SpawnAttacker 的区别: 散点互不关联 (非聚类), 更像白噪声
    """

    def forward(self, data_dict):
        points = data_dict['points']
        device = points.device
        severity = self.severity

        n_scatter = max(50, int(severity * 2000))

        x = torch.empty(n_scatter, device=device).uniform_(0, 70)
        y = torch.empty(n_scatter, device=device).uniform_(-40, 40)
        z = torch.empty(n_scatter, device=device).uniform_(-3, 1)
        intensity = torch.zeros(n_scatter, 1, device=device)

        scattered = torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1), intensity], dim=1)
        data_dict['points'] = torch.cat([points, scattered], dim=0)

        print(f"[Scatter] {points.shape[0]} orig + {n_scatter} scattered → "
              f"{data_dict['points'].shape[0]} total")
        return data_dict
