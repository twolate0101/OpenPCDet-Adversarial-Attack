import torch
from .base import BaseAttacker


class PerturbationAttacker(BaseAttacker):
    """
    C&W 风格对抗点扰动攻击 (论文 Section 3.2, perturbation.py)。

    核心公式:  min  f(x') + λ · ||x' - x||₂²

    其中:
      f(x')  = sum(detection_scores)           (攻击损失，使检测器失明)
      ||·||₂² = 所有体素内有效点的平均 L2 位移   (扰动正则)

    severity 控制扰动预算:
      severity=0.2 → λ 很大, 扰动极小 (<0.1m), 适合"不可见攻击"
      severity=0.5 → λ 中等, 扰动适中 (~0.3m)
      severity=1.0 → λ 很小, 扰动较大 (~0.5m+), 强力攻击

    与 PGDAttacker 的区别:
      - 使用完整梯度 (非 sign), Adam 优化器
      - 软正则 λ (非硬投影 epsilon)
      - 更精细的扰动, 更接近论文原始方法

    参考: Xiang, Qi, Li - "Generating 3D Adversarial Point Clouds" (CVPR 2019, Eq.2-3)
    """

    def __init__(self, severity=1.0, model=None, iterations=50, lr=0.01, **kwargs):
        super().__init__(severity, **kwargs)
        self.model = model
        self.iterations = iterations
        self.lr = lr

        # λ 与 severity 反比: severity 越大 → λ 越小 → 扰动越大
        # 映射: severity ∈ [0.05, 1.0] → λ ∈ [100, 1]
        self.lambda_reg = max(0.5, 1.0 / (severity + 1e-6))

    def _compute_attack_loss(self, pred_dicts):
        """f(x'): 负的检测置信度和。归零=成功使所有目标消失."""
        device = next((p['pred_scores'].device for p in pred_dicts if p['pred_scores'].numel() > 0), 'cpu')
        loss = torch.tensor(0.0, device=device)
        for pred in pred_dicts:
            scores = pred['pred_scores']
            if scores.numel() > 0:
                loss = loss - scores.sum()
        return loss

    def forward(self, data_dict):
        """
        对 voxels 中的 xyz 坐标执行 C&W 扰动攻击。

        Args:
            data_dict: 经过 collate_batch + load_data_to_gpu 后的完整字典
        Returns:
            修改后的 data_dict
        """
        voxels = data_dict['voxels']                       # (M, max_pts, C)
        num_points = data_dict['voxel_num_points']         # (M,)
        max_pts = voxels.shape[1]

        # ---- 分离坐标与特征 ----
        original_xyz = voxels[:, :, :3].clone()            # (M, max_pts, 3)
        intensity = voxels[:, :, 3:].clone()               # (M, max_pts, C-3)

        # 有效点 mask: padding 点不参与扰动
        point_idx = torch.arange(max_pts, device=voxels.device).unsqueeze(0)
        valid_mask = (point_idx < num_points.unsqueeze(1)).float().unsqueeze(-1)  # (M, max_pts, 1)

        # ---- 初始化扰动变量 (全零, 可训练) ----
        delta = torch.zeros_like(original_xyz, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for iteration in range(self.iterations):
            optimizer.zero_grad()

            # 构建扰动后的 dict
            perturbed_xyz = original_xyz + delta * valid_mask
            iter_dict = {}
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    iter_dict[k] = v.clone() if k != 'voxels' else torch.cat(
                        [perturbed_xyz, intensity], dim=-1)
                else:
                    iter_dict[k] = v

            # 前向传播
            with torch.enable_grad():
                pred_dicts, _ = self.model(iter_dict)

                # ---- C&W 联合损失: f(x') + λ · D(x, x') ----
                # f_loss = -sum(scores), 最小化 f_loss 会使 scores 增大
                # 攻击目标是最小化 sum(scores), 所以取反: min (-f_loss) = min sum(scores)
                f_loss = self._compute_attack_loss(pred_dicts)

                # L2 扰动正则: 有效点的平均平方位移
                per_point_l2 = (delta * valid_mask).norm(p=2, dim=-1)  # (M, max_pts)
                n_valid = valid_mask.sum() + 1e-8
                d_loss = (per_point_l2 ** 2).sum() / n_valid           # 均方位移

                total_loss = -f_loss + self.lambda_reg * d_loss

            total_loss.backward()
            optimizer.step()

            # 日志
            if iteration % 10 == 0 or iteration == self.iterations - 1:
                with torch.no_grad():
                    mean_delta = per_point_l2.sum().item() / (n_valid.item())
                    tf = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
                    ff = (-f_loss).item() if isinstance(f_loss, torch.Tensor) else -f_loss
                    df = d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss
                    print(f"[C&W Perturb] iter {iteration + 1}/{self.iterations}, "
                          f"total={tf:.4f}, sum_scores={ff:.4f}, "
                          f"d={df:.4f}, mean_delta={mean_delta:.4f}m, "
                          f"λ={self.lambda_reg:.1f}")

        # ---- 写回扰动后的坐标 ----
        with torch.no_grad():
            data_dict['voxels'] = torch.cat(
                [original_xyz + delta * valid_mask, intensity], dim=-1)
        return data_dict
