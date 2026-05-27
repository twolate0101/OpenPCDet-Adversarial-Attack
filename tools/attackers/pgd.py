"""
PGD 迭代梯度攻击器（CVPR 2019 论文 Section 4 投影梯度下降适配）。

论文原始框架:
  Section 3 (Problem Formulation): 定义 C&W 优化目标
    min  f(x') + λ · D(x, x')                                    (Eq. 2)
  Section 4 (Adversarial Point Perturbation): 对点云坐标执行迭代扰动
    使用 Lp 范数约束扰动幅度，通过 Eq. 2 优化

PGD 实现:
  对点云坐标执行多轮 sign-梯度上升，每轮扰动后投影回 L2 epsilon 球内。
  目标: 最大化检测分数和（= 最小化负分数和）→ 检测器失效。

PointPillar 适配:
  梯度路径: voxels.xyz → PillarVFE → Scatter → Backbone (2D CNN) → Head → pred_dicts
  每轮对 voxels 的 xyz 坐标执行 sign-梯度上升，然后投影回 L2 epsilon 球。

与 perturb.py (C&W) 的区别:
  - PGD: sign 梯度 + 硬投影（epsilon 球约束）
  - C&W: 完整梯度 + Adam + 软正则（λ 加权 L2 损失）

参考: Xiang, Qi, Li — "Generating 3D Adversarial Point Clouds" (CVPR 2019, Section 3-4)
"""

import torch
from .base import BaseAttacker


class PGDAttacker(BaseAttacker):
    """PGD (Projected Gradient Descent) 迭代梯度攻击。

    对体素化后的点云坐标 (voxels xyz) 执行多轮 sign-梯度上升，
    每轮扰动后投影回 L2 epsilon 球内。

    对应论文 Section 4 的点扰动框架。
    """

    def __init__(self, severity=1.0, model=None, iterations=10, **kwargs):
        super().__init__(severity, **kwargs)
        self.model = model
        self.iterations = iterations
        # 每步步长 = 总预算 / 迭代次数
        self.alpha = severity / iterations

    def _compute_attack_loss(self, pred_dicts):
        """攻击损失: 负的检测置信度和。

        梯度上升 → 置信度降低 → 检测器失效。
        """
        device = next((p['pred_scores'].device for p in pred_dicts if p['pred_scores'].numel() > 0), 'cpu')
        loss = torch.tensor(0.0, device=device)
        for pred in pred_dicts:
            scores = pred['pred_scores']
            if scores.numel() > 0:
                loss = loss - scores.sum()
        return loss

    def forward(self, data_dict):
        """
        在 data_dict 的 voxels 上执行 PGD 迭代攻击。

        每轮: 梯度上升 → sign 更新 → L2 投影到 epsilon 球内。

        Args:
            data_dict: collate + load_data_to_gpu 后的完整字典
        Returns:
            修改后的 data_dict（voxels xyz 已扰动）
        """
        voxels = data_dict['voxels']
        num_points = data_dict['voxel_num_points']
        max_pts = voxels.shape[1]
        epsilon = self.severity

        original_xyz = voxels[:, :, :3].clone()
        intensity = voxels[:, :, 3:].clone()

        # 有效点掩码
        point_idx = torch.arange(max_pts, device=voxels.device).unsqueeze(0)
        valid_mask = (point_idx < num_points.unsqueeze(1)).float()

        xyz = original_xyz.clone()

        for iteration in range(self.iterations):
            xyz = xyz.detach().requires_grad_(True)

            iter_dict = {}
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    iter_dict[k] = v.clone() if k != 'voxels' else torch.cat([xyz, intensity], dim=-1)
                else:
                    iter_dict[k] = v

            self.model.zero_grad()
            with torch.enable_grad():
                pred_dicts, _ = self.model(iter_dict)
                loss = self._compute_attack_loss(pred_dicts)
            if loss.requires_grad:
                loss.backward()

            with torch.no_grad():
                grad = xyz.grad
                if grad is not None:
                    # sign 梯度上升
                    update = self.alpha * grad.sign()
                    update = update * valid_mask.unsqueeze(-1)
                    xyz = xyz + update

                    # L2 投影到 epsilon 球内（论文 Section 4 约束）
                    delta = xyz - original_xyz
                    delta_norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
                    scale = torch.clamp(delta_norm / (epsilon + 1e-12), min=1.0)
                    delta = delta / scale
                    xyz = original_xyz + delta * valid_mask.unsqueeze(-1)


        data_dict['voxels'] = torch.cat([xyz, intensity], dim=-1)
        return data_dict
