"""
C&W 风格对抗点扰动攻击（CVPR 2019 论文 Section 4 适配）。

论文原始框架:
  Section 3 (Problem Formulation): 定义对抗优化目标
    min  f(x') + λ · D(x, x')                                    (Eq. 2)
    其中 f(x') 为对抗损失，D 为扰动度量
  Section 4 (Adversarial Point Perturbation): 对现有点云施加微小坐标扰动
    使用 L2 范数作为扰动度量 D，通过 Eq. 2 优化扰动 δ

本实现适配:
  - f(x')  = sum(detection_scores)       ← 攻击损失（使检测置信度归零）
  - D       = 所有有效点的平均 L2 位移    ← 扰动正则（保持不可见性）
  - λ 控制攻击强度与不可见性的权衡:
    severity 大 → λ 小 → 扰动大（强力攻击）
    severity 小 → λ 大 → 扰动小（隐蔽攻击）

与 PGDAttacker 的区别:
  - 使用完整梯度（非 sign），Adam 优化器 → 更精细的扰动方向
  - 软正则 λ（非硬投影 epsilon 球）→ 更接近论文原始 C&W 公式
  - 扰动更均匀地分布在所有点上

参考: Xiang, Qi, Li — "Generating 3D Adversarial Point Clouds" (CVPR 2019, Eq. 2-3, Section 3-4)
"""

import torch
from .base import BaseAttacker


class PerturbationAttacker(BaseAttacker):
    """C&W 风格体素坐标扰动攻击。

    对 voxels 中的 xyz 坐标执行 Adam 迭代扰动，
    目标: 最小化检测分数和 + λ·均方位移。

    对应论文 Section 4 的点扰动方法 + Section 3 Eq.2 的 C&W 优化框架。
    """

    def __init__(self, severity=1.0, model=None, iterations=50, lr=0.01, **kwargs):
        super().__init__(severity, **kwargs)
        self.model = model
        self.iterations = iterations
        self.lr = lr

        # λ 与 severity 反比: severity 越大 → λ 越小 → 允许更大扰动
        # 映射: severity ∈ [0.05, 1.0] → λ ∈ [100, 1]
        self.lambda_reg = max(0.5, 1.0 / (severity + 1e-6))

    def _compute_attack_loss(self, pred_dicts):
        """f(x'): 负的检测置信度和。梯度上升 → 置信度降低 → 检测失效。"""
        device = next((p['pred_scores'].device for p in pred_dicts if p['pred_scores'].numel() > 0), 'cpu')
        loss = torch.tensor(0.0, device=device)
        for pred in pred_dicts:
            scores = pred['pred_scores']
            if scores.numel() > 0:
                loss = loss - scores.sum()
        return loss

    def forward(self, data_dict):
        """
        对 voxels 中的 xyz 坐标执行 C&W 迭代扰动。

        优化目标:
          min_δ  -sum(scores) + λ · mean(||δ||₂²)

        即: 在最小化检测置信度的同时，约束每个点的位移量。

        Args:
            data_dict: collate + load_data_to_gpu 后的完整字典
        Returns:
            修改后的 data_dict（voxels xyz 已扰动）
        """
        voxels = data_dict['voxels']                       # (M, max_pts, C)
        num_points = data_dict['voxel_num_points']         # (M,)
        max_pts = voxels.shape[1]

        # 分离坐标与特征（强度值不参与扰动）
        original_xyz = voxels[:, :, :3].clone()            # (M, max_pts, 3)
        intensity = voxels[:, :, 3:].clone()               # (M, max_pts, C-3)

        # 有效点掩码: padding 点不加扰动
        point_idx = torch.arange(max_pts, device=voxels.device).unsqueeze(0)
        valid_mask = (point_idx < num_points.unsqueeze(1)).float().unsqueeze(-1)  # (M, max_pts, 1)

        # 初始化扰动 δ = 0
        delta = torch.zeros_like(original_xyz, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for iteration in range(self.iterations):
            optimizer.zero_grad()

            # 构建扰动后的 dict: x' = x + δ（仅有效点）
            perturbed_xyz = original_xyz + delta * valid_mask
            iter_dict = {}
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    iter_dict[k] = v.clone() if k != 'voxels' else torch.cat(
                        [perturbed_xyz, intensity], dim=-1)
                else:
                    iter_dict[k] = v

            with torch.enable_grad():
                pred_dicts, _ = self.model(iter_dict)

                # C&W 联合损失（论文 Eq. 2-3）
                f_loss = self._compute_attack_loss(pred_dicts)  # -sum(scores)

                # L2 扰动正则: 有效点的平均平方位移
                per_point_l2 = (delta * valid_mask).norm(p=2, dim=-1)  # (M, max_pts)
                n_valid = valid_mask.sum() + 1e-8
                d_loss = (per_point_l2 ** 2).sum() / n_valid           # 均方 L2 位移

                # total = -f_loss + λ·d_loss
                #   -f_loss = sum(scores) → 最小化 total → scores 变小 → 检测失效
                total_loss = -f_loss + self.lambda_reg * d_loss

            total_loss.backward()
            optimizer.step()

        # 写回扰动后的坐标
        with torch.no_grad():
            data_dict['voxels'] = torch.cat(
                [original_xyz + delta * valid_mask, intensity], dim=-1)
        return data_dict
