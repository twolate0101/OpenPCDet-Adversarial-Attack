"""
梯度优化对抗散点攻击（CVPR 2019 论文 Section 5.1 适配）。

论文 Section 5.1 "Generating Adversarial Independent Points":
  在点云中生成独立的对抗散点（而非扰动现有点），通过 initialize-and-shift 方法
  将初始点优化移动到关键位置附近，使分类器产生错误预测。

PointPillar 检测器适配（本文的 scatter 攻击器）:
  原始 Section 5.1 针对 PointNet 分类任务，本实现适配到 PointPillar 检测框架：
  使用 CriticalPointFinder 定位脆弱区域，然后在高斯散布初始化 + sign-梯度优化。
  每个散点 = 一个独立的 1 点 pillar 体素，所有散点的 XYZ 坐标联合优化。

攻击流程:
  阶段 1 — 关键点分析: CriticalPointFinder 找到检测器最敏感的 BEV 区域
  阶段 2 — 散点初始化: 在每个脆弱区域附近以高斯散布生成独立散点
  阶段 3 — 梯度优化: 通过 sign-梯度迭代微调散点 XYZ，最小化检测分数和

参考: Xiang, Qi, Li — "Generating 3D Adversarial Point Clouds" (CVPR 2019, Section 5.1)
"""

import torch
import numpy as np
from .base import BaseAttacker
from .critical_points import CriticalPointFinder


class ScatterAttacker(BaseAttacker):
    """梯度引导的散点放置攻击，含位置优化。

    对应论文 Section 5.1 的独立散点生成方法，适配为在脆弱区域插入独立散点。
    每个散点 = 1 个独立 pillar 体素，通过 sign-梯度迭代优化 XYZ 位置。
    """

    def __init__(self, severity=1.0, model=None, iterations=30, lr=0.05, **kwargs):
        super().__init__(severity, **kwargs)
        self.model = model
        self.iterations = iterations
        self.lr = lr

        # severity 控制散点数量: severity=0.3 → ~300, severity=1.0 → ~1000
        self.num_points = max(100, int(severity * 1000))
        self.scatter_sigma = severity * 2.0

    def _xyz_to_pillar_coords(self, x, y):
        """连续 xy 坐标 → pillar 网格索引。

        PointPillar 网格: 432×496, voxel_size [0.16, 0.16, 4]。
        """
        xp = torch.clamp((x / 0.16).long(), 0, 431)
        yp = torch.clamp(((y + 39.68) / 0.16).long(), 0, 495)
        return xp, yp

    def forward(self, data_dict):
        """
        三阶段散点攻击（论文 Section 5.1 方法扩展）。

        阶段 1: CriticalPointFinder → 脆弱区域中心。
        阶段 2: 在每个脆弱区域周围高斯散布独立散点。
        阶段 3: sign-梯度迭代优化散点 XYZ 位置。
        """
        vault = dict(
            voxels=data_dict['voxels'].clone(),
            voxel_coords=data_dict['voxel_coords'].clone(),
            voxel_num_points=data_dict['voxel_num_points'].clone(),
        )
        device = data_dict['voxels'].device
        batch_idx = vault['voxel_coords'][0, 0].item()

        # ── 阶段 1: 查找脆弱区域（论文 Section 5.2 关键点分析）──
        finder = CriticalPointFinder(
            self.model, eps=1.0, min_samples=5, top_k_pillars=64)
        centers = finder.find(data_dict)

        # 梯度分析失败时的回退: 在场景范围内随机选点
        if len(centers) == 0:
            centers = []
            for _ in range(min(self.num_points // 5, 20)):
                cx = np.random.uniform(5, 60)
                cy = np.random.uniform(-30, 30)
                cz = np.random.uniform(-1.5, 0.5)
                centers.append((cx, cy, cz))

        # ── 阶段 2: 在脆弱区域附近生成散点 ──
        centers_np = np.array(centers)
        pts_per_center = self.num_points // max(len(centers), 1)

        adv_points = []
        for cx, cy, cz in centers:
            # 以脆弱区域为中心，高斯散布
            offsets = torch.randn(pts_per_center, 3, device=device) * self.scatter_sigma
            cluster = torch.tensor([cx, cy, cz], device=device) + offsets
            # 裁剪到场景范围
            cluster[:, 0].clamp_(0.5, 69.0)
            cluster[:, 1].clamp_(-39.0, 39.0)
            cluster[:, 2].clamp_(-2.5, 0.5)
            adv_points.append(cluster)

        adv_points = torch.cat(adv_points, dim=0)  # (N_total, 3)
        adv_points = adv_points[:self.num_points]   # 强制执行数量预算
        N_total = adv_points.shape[0]

        adv_points.requires_grad_(True)

        # ── 阶段 3: 梯度优化散点位置（论文 Section 5.1 优化框架）──
        optimizer = torch.optim.Adam([adv_points], lr=self.lr)

        for iteration in range(self.iterations):
            optimizer.zero_grad()

            # 每个散点 → 独立的 1 点 pillar 体素
            xp, yp = self._xyz_to_pillar_coords(adv_points[:, 0], adv_points[:, 1])
            zp = torch.zeros(N_total, dtype=torch.long, device=device)
            new_coords = torch.stack([
                torch.full((N_total,), batch_idx, dtype=torch.long, device=device),
                zp, yp, xp
            ], dim=1)

            # 构建体素: 使用 torch.cat（非 in-place 赋值）保持梯度链完整
            max_pts = vault['voxels'].shape[1]
            adv_xyz_pad = torch.cat([
                adv_points.unsqueeze(1),
                torch.zeros(N_total, max_pts - 1, 3, device=device)
            ], dim=1)
            adv_i_pad = torch.full((N_total, max_pts, 1), 0.3, device=device)
            new_voxels = torch.cat([adv_xyz_pad, adv_i_pad], dim=-1)

            data_dict['voxels'] = torch.cat([vault['voxels'], new_voxels], dim=0)
            data_dict['voxel_coords'] = torch.cat([vault['voxel_coords'], new_coords], dim=0)
            new_npts = torch.ones(N_total, dtype=torch.long, device=device)
            data_dict['voxel_num_points'] = torch.cat([vault['voxel_num_points'], new_npts], dim=0)

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
                if adv_points.grad is not None:
                    adv_points -= self.lr * adv_points.grad.sign()
                    adv_points.grad.zero_()
                    # 裁剪到场景范围
                    adv_points[:, 0].clamp_(0.5, 69.0)
                    adv_points[:, 1].clamp_(-39.0, 39.0)
                    adv_points[:, 2].clamp_(-2.5, 0.5)

        # 最终重建（断开梯度）
        xp, yp = self._xyz_to_pillar_coords(adv_points[:, 0], adv_points[:, 1])
        zp = torch.zeros(N_total, dtype=torch.long, device=device)
        new_coords = torch.stack([
            torch.full((N_total,), batch_idx, dtype=torch.long, device=device),
            zp, yp, xp
        ], dim=1)
        max_pts = vault['voxels'].shape[1]
        adv_xyz_pad = torch.cat([
            adv_points.detach().unsqueeze(1),
            torch.zeros(N_total, max_pts - 1, 3, device=device)
        ], dim=1)
        adv_i_pad = torch.full((N_total, max_pts, 1), 0.3, device=device)
        new_voxels = torch.cat([adv_xyz_pad, adv_i_pad], dim=-1)

        data_dict['voxels'] = torch.cat([vault['voxels'], new_voxels], dim=0)
        data_dict['voxel_coords'] = torch.cat([vault['voxel_coords'], new_coords], dim=0)
        new_npts = torch.ones(N_total, dtype=torch.long, device=device)
        data_dict['voxel_num_points'] = torch.cat([vault['voxel_num_points'], new_npts], dim=0)

        print(f"[Scatter] severity={self.severity:.2f}: "
              f"{N_total} 个散点 @ {len(centers)} 个脆弱区域, "
              f"散布={self.scatter_sigma:.1f}m, {self.iterations} 轮迭代")

        return data_dict
