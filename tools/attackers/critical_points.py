"""
关键点查找器 — 基于梯度分析的脆弱区域检测（CVPR 2019 论文 Section 5.2-5.3）。

论文原始方法：
  Section 5.2 对抗点生成: 利用 PointNet 的 critical points（max-pool 激活点）定位目标物体上的
  脆弱表面区域，在这些区域放置对抗点簇以诱导错误分类。
  Section 5.3 对抗物体放置: 扩展 Section 5.2 的思路，将对抗扰动从离散点扩展到连续的几何物体
  （立方体、球体），对物体的位置、尺寸、旋转进行联合梯度优化。

PointPillar 检测器适配：
  原始论文针对 PointNet 分类任务（点→全局特征→分类），梯度信号通过 max-pool 层的激活点（critical
  points）直接传递。PointPillar 是检测框架（体素化→PFN→2D CNN→检测头），梯度路径较长。

  本实现使用体素 xyz 坐标上的梯度幅值作为"关键性"的替代指标：对检测分数影响越大的点位置
  越"关键"。通过 DBSCAN 在 BEV 平面上聚类这些关键点的空间位置 → 得到脆弱区域中心。

核心流程:
  1. 前向+反向传播，计算 voxels xyz 梯度
  2. 逐点梯度幅值 → 逐 pillar 关键性得分（取 max）
  3. Top-K 最关键的 pillar
  4. 提取每个关键 pillar 中梯度最大的点的 xyz 坐标
  5. DBSCAN 在 XY（BEV）平面聚类 → 脆弱区域中心

参考: Xiang, Qi, Li — "Generating 3D Adversarial Point Clouds" (CVPR 2019)
"""

import numpy as np
import torch
from sklearn.cluster import DBSCAN


class CriticalPointFinder:
    """通过梯度分析 + DBSCAN 聚类查找 3D 检测场景中的脆弱空间区域。

    对应论文 Section 5.2 的 critical point 分析流程，适配 PointPillar 检测器。
    """

    def __init__(self, model, eps=0.5, min_samples=3, top_k_pillars=64):
        """
        Args:
            model: PointPillar（或类似）检测器，需支持可微前向传播
            eps: DBSCAN 聚类半径（米），用于合并空间上相近的关键点
            min_samples: DBSCAN 最小样本数，少于此数的点视为噪声
            top_k_pillars: 选取最关键的 pillar 数量
        """
        self.model = model
        self.eps = eps
        self.min_samples = min_samples
        self.top_k_pillars = top_k_pillars

    def find(self, data_dict):
        """
        通过梯度分析 + DBSCAN 聚类查找脆弱区域中心。

        流程（对应论文 Section 5.2 步骤）:
          1. 对体素 xyz 坐标求检测分数和的梯度
          2. 计算逐点梯度 L2 幅值
          3. 逐 pillar 关键性 = pillar 内所有点梯度幅值的最大值
          4. 选取 Top-K 最关键的 pillar
          5. 提取每个关键 pillar 中梯度最大点的 xyz 坐标
          6. DBSCAN 在 XY 平面（BEV）聚类 → 聚类中心即脆弱区域

        Args:
            data_dict: collate 后的 GPU 数据字典，需包含 'voxels', 'voxel_coords',
                       'voxel_num_points'

        Returns:
            list of (x, y, z) 元组: 脆弱区域的聚类中心。
            未找到足够聚类时返回空列表。
        """
        voxels = data_dict['voxels']            # (M, max_pts, C)
        coords = data_dict['voxel_coords']       # (M, 4): (batch, z, y, x)
        num_points = data_dict['voxel_num_points']  # (M,)
        max_pts = voxels.shape[1]

        # ── 步骤 1: 计算体素 xyz 坐标对检测分数的梯度 ──
        xyz = voxels[:, :, :3].clone()
        intensity = voxels[:, :, 3:].clone()

        # 构建有效点掩码（排除 padding 零点）
        pt_idx = torch.arange(max_pts, device=voxels.device).unsqueeze(0)
        valid_mask = (pt_idx < num_points.unsqueeze(1)).float().unsqueeze(-1)

        xyz_grad = xyz.detach().requires_grad_(True)
        iter_dict = {}
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                iter_dict[k] = v.clone() if k != 'voxels' else torch.cat([xyz_grad, intensity], dim=-1)
            else:
                iter_dict[k] = v

        self.model.zero_grad()
        with torch.enable_grad():
            pred_dicts, _ = self.model(iter_dict)
            # 攻击目标: 最大化检测分数和 → loss 取正和
            loss = 0.0
            for pred in pred_dicts:
                scores = pred['pred_scores']
                if scores.numel() > 0:
                    loss = loss + scores.sum()

        if isinstance(loss, float):
            return []

        loss.backward()
        grad = xyz_grad.grad  # (M, max_pts, 3)

        if grad is None:
            return []

        # ── 步骤 2: 逐点梯度幅值 ──
        point_grad_norm = (grad * valid_mask).norm(dim=-1)  # (M, max_pts)

        # ── 步骤 3: 逐 pillar 关键性 = 该 pillar 内梯度幅值的最大值 ──
        pillar_criticality = point_grad_norm.max(dim=-1).values  # (M,)

        # ── 步骤 4: 选取 Top-K 最关键的 pillar ──
        k = min(self.top_k_pillars, pillar_criticality.shape[0])
        top_k_indices = torch.topk(pillar_criticality, k).indices

        # ── 步骤 5: 提取每个关键 pillar 中梯度最大点的 xyz ──
        critical_pts_xyz = []
        for idx in top_k_indices:
            pgrad_norm = point_grad_norm[idx]  # (max_pts,)
            npts = int(num_points[idx].item())
            if npts > 0:
                best_pt_idx = torch.argmax(pgrad_norm[:npts])
                pt_xyz = xyz[idx, best_pt_idx].detach().cpu().numpy()
                critical_pts_xyz.append(pt_xyz)

        if len(critical_pts_xyz) < self.min_samples:
            # 关键点不足，直接返回重心
            if len(critical_pts_xyz) > 0:
                pts = np.array(critical_pts_xyz)
                return [tuple(pts.mean(axis=0))]
            return []

        critical_pts_xyz = np.array(critical_pts_xyz)

        # ── 步骤 6: DBSCAN 在 XY 平面（BEV）聚类 ──
        # PointPillar 检测主要依赖 BEV 特征，因此在 XY 平面聚类即可
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(
            critical_pts_xyz[:, :2]
        )

        # ── 步骤 7: 计算聚类中心（3D: XY 中心 + 平均 Z）──
        labels = clustering.labels_
        centers = []
        for label in set(labels):
            if label == -1:  # DBSCAN 噪声点，跳过
                continue
            cluster_pts = critical_pts_xyz[labels == label]
            center = cluster_pts.mean(axis=0)
            centers.append(tuple(center))

        return centers
