import torch
from .base import BaseAttacker


class SpawnAttacker(BaseAttacker):
    """
    对抗性点云插入攻击 (CVPR 2019 论文核心方法之三)。

    在场景中随机注入虚假点簇，模拟攻击者向 LiDAR 信号中注入伪造目标，
    诱导检测器产生虚警 (false positives)。

    severity 控制注入簇的数量和规模：
      severity=0.2  → 2 个小簇 (~40 points, 0.4m 半径)
      severity=0.5  → 5 个中簇 (~250 points, 1.0m 半径)
      severity=1.0  → 10 个大簇 (~1000 points, 2.0m 半径)

    簇在 LiDAR 有效范围内随机放置：x∈[5,65] y∈[-35,35] z∈[-2,0]
    """

    def forward(self, data_dict):
        points = data_dict['points']  # (N, 4): [x, y, z, intensity]
        device = points.device
        severity = self.severity

        # ---- 根据 severity 计算簇参数 ----
        n_clusters = max(1, int(severity * 10))            # 1~10 个簇
        radius = severity * 2.0                             # 0.2~2.0m 簇半径
        n_points_per_cluster = max(20, int(severity * 200)) # 20~200 点/簇

        spawned_list = []
        for _ in range(n_clusters):
            # 簇中心：在 LiDAR 前方范围内随机采样
            cx = torch.empty(1, device=device).uniform_(5, 65)
            cy = torch.empty(1, device=device).uniform_(-35, 35)
            cz = torch.empty(1, device=device).uniform_(-2, 0)

            # 在球体内均匀采样点
            # 使用球坐标：u=cos(theta)∈[-1,1], phi∈[0,2π], r∈[0,radius]
            u = torch.empty(n_points_per_cluster, device=device).uniform_(-1, 1)
            theta = torch.acos(u)
            phi = torch.empty(n_points_per_cluster, device=device).uniform_(0, 2 * 3.14159)
            r = radius * torch.rand(n_points_per_cluster, device=device) ** (1 / 3)

            # 球坐标 → 笛卡尔坐标
            x = r * torch.sin(theta) * torch.cos(phi) + cx
            y = r * torch.sin(theta) * torch.sin(phi) + cy
            z = r * torch.cos(theta) + cz

            # intensity 设 0（无反射信号的人造点）
            intensity = torch.zeros(n_points_per_cluster, 1, device=device)

            cluster = torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1), intensity], dim=1)
            spawned_list.append(cluster)

        spawned = torch.cat(spawned_list, dim=0)
        data_dict['points'] = torch.cat([points, spawned], dim=0)

        n_orig = points.shape[0]
        n_new = spawned.shape[0]
        print(f"[Spawn] {n_orig} orig + {n_new} fake → {n_orig + n_new} total "
              f"({n_clusters} clusters, r={radius:.1f}m)")

        return data_dict
