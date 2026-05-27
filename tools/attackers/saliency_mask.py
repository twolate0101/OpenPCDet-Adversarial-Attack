"""
显著性体素掩码攻击（白盒删除，基于 Robust3DOD 梯度分析框架）。

攻击机制:
  借鉴 Robust3DOD 的梯度脆弱区域分析思想，将其从注入攻击扩展到删除攻击。
  通过一次前向+反向传播获取 voxels 的梯度，按梯度绝对值排序，
  将对检测贡献最大的 Top K% voxels 从数据中真正移除。

  实现方式: 同步过滤 voxels / voxel_coords / voxel_num_points，
  而非简单置零（置零 xyz 会在 PillarVFE 中产生虚假的 f_center 偏移特征，
  效果更接近强扰动而非真正删除）。

  与 PGD/Perturb 的区别:
    PGD/Perturb: 微调已有 voxels 的坐标（扰动）
    SaliencyMask: 直接移除最关键的 voxels（删除）

  与 Drop/GeoDrop 的区别:
    Drop/GeoDrop: 黑盒随机/几何删除（不依赖梯度）
    SaliencyMask: 白盒梯度指导删除（精准定位最关键 voxels）

  severity 映射:
    severity=0.1  -> 移除梯度最大的 10% voxels
    severity=0.3  -> 移除梯度最大的 30% voxels
    severity=0.5  -> 移除梯度最大的 50% voxels

理论来源:
  - CVPR 2019 "Generating 3D Adversarial Point Clouds" Section 5 (Dropping)
  - JSMA (Jacobian-based Saliency Map Attack) 梯度显著性思想
  - Robust3DOD 梯度脆弱区域分析框架
"""

import torch
from .base import BaseAttacker


class SaliencyMaskAttacker(BaseAttacker):
    """显著性体素掩码攻击。

    用梯度精确定位对检测贡献最大的 voxels，将其从数据中真正移除。
    属于白盒攻击，需要模型梯度。
    """

    def __init__(self, severity=0.1, model=None, **kwargs):
        """
        Args:
            severity: 删除比例 (0.0 ~ 1.0)，即 Top K% 的 voxels 被移除
            model: PointPillars 模型实例（用于计算梯度）
        """
        super().__init__(severity, **kwargs)
        self.model = model

    def _compute_attack_loss(self, pred_dicts):
        """攻击损失: 负的检测置信度和。"""
        device = next(
            (p['pred_scores'].device for p in pred_dicts if p['pred_scores'].numel() > 0),
            'cpu'
        )
        loss = torch.tensor(0.0, device=device)
        for pred in pred_dicts:
            scores = pred['pred_scores']
            if scores.numel() > 0:
                loss = loss - scores.sum()
        return loss

    def forward(self, data_dict):
        """
        执行显著性掩码攻击（真正移除 voxels）。

        流程:
          1. 克隆 voxels，开启梯度
          2. 前向传播计算检测分数
          3. 反向传播获取梯度
          4. 按梯度绝对值排序，选 Top K% voxels
          5. 从 voxels / voxel_coords / voxel_num_points 中同步移除这些 voxels

        为什么不直接置零 xyz:
          PillarVFE 计算 f_center = xyz - voxel_center，其中 voxel_center 由
          voxel_coords 的离散索引决定。如果把 xyz 置零而 voxel_coords 保持不变，
          f_center = (0,0,0) - voxel_center 会产生很大的虚假偏移特征，
          效果更像是强扰动而非真正删除。真正移除 voxel 后，对应的 BEV 伪图像
          位置保持为零（PointPillarScatter 不会在该位置放置特征）。

        Args:
            data_dict: collate + load_data_to_gpu 后的完整字典
        Returns:
            修改后的 data_dict（关键 voxels 已被真正移除）
        """
        voxels = data_dict['voxels']
        num_voxels = voxels.shape[0]

        # 移除数量（最多移除 90%，保证至少保留部分 voxels）
        k = min(max(1, int(num_voxels * self.severity)), num_voxels - 1)

        # 克隆并开启梯度
        voxels_grad = voxels.clone().detach().requires_grad_(True)

        # 构造临时字典，替换 voxels 为带梯度的版本
        tmp_dict = {}
        for key, val in data_dict.items():
            if isinstance(val, torch.Tensor):
                tmp_dict[key] = val if key != 'voxels' else voxels_grad
            else:
                tmp_dict[key] = val

        # 前向 + 反向
        self.model.zero_grad()
        with torch.enable_grad():
            pred_dicts, _ = self.model(tmp_dict)
            loss = self._compute_attack_loss(pred_dicts)

        if loss.requires_grad:
            loss.backward()

        # 按梯度绝对值排序，真正移除这些 voxels
        grad = voxels_grad.grad
        if grad is not None:
            # 每个 voxel 的梯度强度: 所有点、所有特征的绝对值之和
            grad_abs = grad.abs().sum(dim=(1, 2))  # (num_voxels,)
            _, top_indices = grad_abs.topk(k)

            # 构造保留掩码: 保留未被选中的 voxels
            keep_mask = torch.ones(num_voxels, dtype=torch.bool, device=voxels.device)
            keep_mask[top_indices] = False
            # 确保至少保留 1 个 voxel，避免 PointPillarScatter 空 coords 崩溃
            if not keep_mask.any():
                keep_mask[0] = True

            # 同步过滤三个张量
            data_dict['voxels'] = voxels[keep_mask]
            data_dict['voxel_coords'] = data_dict['voxel_coords'][keep_mask]
            data_dict['voxel_num_points'] = data_dict['voxel_num_points'][keep_mask]

        # 如果梯度为 None（所有检测为空），不做修改
        return data_dict
