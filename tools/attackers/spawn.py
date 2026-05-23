"""
梯度优化对抗点簇生成攻击（CVPR 2019 论文 Section 5.2 适配）。

论文 Section 5.2 "Generating Adversarial Point Clouds":
  1. 通过 PointNet 的 critical points 定位目标物体上的脆弱表面区域
  2. 在这些区域放置对抗点簇
  3. 梯度优化点簇位置，最大化分类错误

PointPillar 检测器适配:
  使用 CriticalPointFinder（梯度分析 + DBSCAN 聚类）替代 PointNet 的 critical points，
  在 BEV 平面上找到对检测分数影响最大的脆弱区域，然后在这些区域生成对抗点簇并梯度优化。

攻击流程:
  阶段 1 — 关键点分析: CriticalPointFinder 通过梯度反传找到检测器最敏感的空间区域
  阶段 2 — 点簇初始化: 在每个脆弱区域中心附近随机散布点云簇
  阶段 3 — 梯度优化: 通过 Adam 优化器迭代调整点簇 XYZ 位置，最小化检测置信度

参考: Xiang, Qi, Li — "Generating 3D Adversarial Point Clouds" (CVPR 2019, Section 5.2)
"""

import torch
import numpy as np
from .base import BaseAttacker
from .critical_points import CriticalPointFinder


class SpawnAttacker(BaseAttacker):
    """梯度引导的点簇生成攻击，结合 DBSCAN 脆弱区域检测。

    对应论文 Section 5.2 的对抗点云生成方法。
    每个簇 = 一个独立的 pillar 体素（32 个表面点），
    簇的 XYZ 位置通过梯度下降联合优化。
    """

    def __init__(self, severity=1.0, model=None, iterations=30, lr=0.05, **kwargs):
        super().__init__(severity, **kwargs)
        self.model = model
        self.iterations = iterations
        self.lr = lr

        # severity 控制点簇数量: severity=0.5 → ~5 簇, 1.0 → ~10 簇
        self.num_clusters = max(1, int(severity * 10))
        self.pts_per_cluster = 32
        self.cluster_radius = 0.8 * severity

    def _xyz_to_pillar_coords(self, x, y, z):
        """连续 xyz → pillar 索引 (batch, z_idx, y_idx, x_idx)。

        PointPillar 网格参数:
          voxel_size = [0.16, 0.16, 4]
          POINT_CLOUD_RANGE = [0, -39.68, -3, 69.12, 39.68, 1]
          网格: 432×496×1
        """
        xp = torch.clamp((x / 0.16).long(), 0, 431)     # (0, 69.12) / 0.16 → 432
        yp = torch.clamp(((y + 39.68) / 0.16).long(), 0, 495)  # 496
        return xp, yp

    def _add_voxels(self, data_dict, adv_xyz, adv_intensity):
        """将对抗体素插入 data_dict。

        每个簇构成一个独立的 pillar 体素，pillar 坐标由簇的第一个点（中心点）决定。
        使用 torch.cat 拼接保持梯度链完整。

        Args:
            data_dict: collate 后的 GPU 数据字典
            adv_xyz: (N_new, 32, 3) 对抗点坐标
            adv_intensity: (N_new, 32, 1) 伪造的强度值

        Returns:
            更新后的 data_dict（voxels, voxel_coords, voxel_num_points 已扩展）
        """
        device = data_dict['voxels'].device
        N_new = adv_xyz.shape[0]
        max_pts = data_dict['voxels'].shape[1]

        # 填充或裁剪以匹配原始体素的 max_pts
        if max_pts > adv_xyz.shape[1]:
            pad_xyz = torch.zeros(N_new, max_pts - adv_xyz.shape[1], 3, device=device)
            pad_i = torch.zeros(N_new, max_pts - adv_intensity.shape[1], 1, device=device)
            adv_xyz = torch.cat([adv_xyz, pad_xyz], dim=1)
            adv_intensity = torch.cat([adv_intensity, pad_i], dim=1)

        # 用每个簇的第一个点（簇中心）确定 pillar 坐标
        first_pts = adv_xyz[:, 0, :]  # (N_new, 3)
        xp, yp = self._xyz_to_pillar_coords(first_pts[:, 0], first_pts[:, 1], first_pts[:, 2])
        batch_idx = data_dict['voxel_coords'][0, 0].item()
        zp = torch.zeros(N_new, dtype=torch.long, device=device)
        new_coords = torch.stack([
            torch.full((N_new,), batch_idx, dtype=torch.long, device=device),
            zp, yp, xp
        ], dim=1)  # (N_new, 4)

        # 构建新体素: [xyz, intensity] in last dim
        new_voxels = torch.cat([adv_xyz, adv_intensity], dim=-1)  # (N_new, max_pts, 4)

        # 每个体素的有效点数（不含 padding）
        pts_per = min(self.pts_per_cluster, max_pts)
        new_num_pts = torch.full((N_new,), pts_per, dtype=torch.long, device=device)

        # 拼接到原始数据
        data_dict['voxels'] = torch.cat([data_dict['voxels'].float(), new_voxels.float()], dim=0)
        data_dict['voxel_coords'] = torch.cat([data_dict['voxel_coords'], new_coords], dim=0)
        data_dict['voxel_num_points'] = torch.cat([
            data_dict['voxel_num_points'], new_num_pts.to(data_dict['voxel_num_points'].device)])

        return data_dict

    def forward(self, data_dict):
        """
        三阶段对抗点簇攻击（对应论文 Section 5.2 方法）。

        阶段 1: CriticalPointFinder 梯度分析 → DBSCAN 聚类 → 脆弱区域中心。
        阶段 2: 在每个脆弱区域初始化点簇（32 个点，球状散布）。
        阶段 3: Adam 优化器迭代调整点簇 XYZ → 最小化检测分数和。
        """
        original_voxels = data_dict['voxels'].clone()
        original_coords = data_dict['voxel_coords'].clone()
        original_num_pts = data_dict['voxel_num_points'].clone()

        # ── 阶段 1: 梯度分析查找脆弱区域（论文 Section 5.2 步骤 1-2）──
        finder = CriticalPointFinder(
            self.model, eps=0.8, min_samples=2,
            top_k_pillars=max(64, self.num_clusters * 10))
        centers = finder.find(data_dict)

        # 梯度分析失效时的回退: 随机选取现有 pillar 中心
        if len(centers) == 0:
            voxels = data_dict['voxels'][:, :, :3]
            npts = data_dict['voxel_num_points']
            idxs = np.random.choice(len(npts), min(self.num_clusters, len(npts)), replace=False)
            for idx in idxs:
                n = max(1, npts[idx].item())
                center = voxels[idx, :n].mean(dim=0).cpu().numpy()
                centers.append(tuple(center))

        # 限制簇数量
        centers = centers[:self.num_clusters]

        # ── 阶段 2: 创建对抗点簇（论文 Section 5.2 步骤 3）──
        device = data_dict['voxels'].device
        N = len(centers)
        r = self.cluster_radius

        # 每个簇: 中心 + 球体内随机偏移
        adv_xyz_list = []
        for cx, cy, cz in centers:
            offsets = (torch.rand(32, 3, device=device) - 0.5) * 2 * r
            cluster = torch.tensor([cx, cy, cz], device=device) + offsets
            adv_xyz_list.append(cluster)
        adv_xyz_init = torch.stack(adv_xyz_list)  # (N, 32, 3)

        # 伪造反射强度（中等反射率）
        adv_intensity = torch.full((N, 32, 1), 0.3, device=device)

        # ── 阶段 3: 梯度优化点簇位置（论文 Section 5.2 步骤 4）──
        data_dict['voxels'] = original_voxels.clone()
        data_dict['voxel_coords'] = original_coords.clone()
        data_dict['voxel_num_points'] = original_num_pts.clone()

        adv_xyz = adv_xyz_init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([adv_xyz], lr=self.lr)

        for iteration in range(self.iterations):
            optimizer.zero_grad()

            # 每轮重建 data_dict（保持梯度链）
            data_dict['voxels'] = original_voxels.clone()
            data_dict['voxel_coords'] = original_coords.clone()
            data_dict['voxel_num_points'] = original_num_pts.clone()
            data_dict = self._add_voxels(data_dict, adv_xyz, adv_intensity)

            with torch.enable_grad():
                pred_dicts, _ = self.model(data_dict)
                loss = torch.tensor(0.0, device=device)
                for pred in pred_dicts:
                    scores = pred['pred_scores']
                    if scores.numel() > 0:
                        loss = loss + scores.sum()

            if loss.requires_grad:
                loss.backward()

            with torch.no_grad():
                if adv_xyz.grad is not None:
                    adv_xyz -= self.lr * adv_xyz.grad.sign()
                    adv_xyz.grad.zero_()

        # 最终重建（使用优化后的点位置，断开梯度）
        data_dict['voxels'] = original_voxels.clone()
        data_dict['voxel_coords'] = original_coords.clone()
        data_dict['voxel_num_points'] = original_num_pts.clone()
        data_dict = self._add_voxels(data_dict, adv_xyz.detach(), adv_intensity)

        print(f"[Spawn] severity={self.severity:.2f}: "
              f"{N} 个点簇 @ {len(centers)} 个脆弱区域, "
              f"r={r:.1f}m, {self.iterations} 轮迭代")

        return data_dict
