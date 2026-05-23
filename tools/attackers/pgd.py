import torch
from .base import BaseAttacker


class PGDAttacker(BaseAttacker):
    """
    PGD (Projected Gradient Descent) 迭代梯度攻击器。

    对体素化后的点云坐标 (voxels 中的 xyz) 执行多轮梯度上升攻击，
    使检测器置信度持续下降。每轮扰动后投影回 L2 epsilon 球内。

    梯度路径: voxels.xyz → PillarVFE → Scatter → Backbone → Head → pred_dicts
    """

    def __init__(self, severity=1.0, model=None, iterations=10, **kwargs):
        super().__init__(severity, **kwargs)
        self.model = model
        self.iterations = iterations
        self.alpha = severity / iterations

    def _compute_attack_loss(self, pred_dicts):
        """攻击损失：负的检测置信度和。梯度上升 → 置信度降低。"""
        loss = 0.0
        for pred in pred_dicts:
            scores = pred['pred_scores']
            if scores.numel() > 0:
                loss = loss - scores.sum()
        return loss

    def forward(self, data_dict):
        """
        在 data_dict 的 voxels 上执行 PGD 迭代攻击。

        Args:
            data_dict: 经过 collate_batch + load_data_to_gpu 后的完整字典，
                       必须包含 'voxels', 'voxel_coords', 'voxel_num_points'
        Returns:
            修改后的 data_dict，其中 voxels 的 xyz 坐标已被扰动
        """
        voxels = data_dict['voxels']
        num_points = data_dict['voxel_num_points']
        max_pts = voxels.shape[1]
        epsilon = self.severity

        original_xyz = voxels[:, :, :3].clone()
        intensity = voxels[:, :, 3:].clone()

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
                loss.backward()

            with torch.no_grad():
                grad = xyz.grad
                if grad is not None:
                    update = self.alpha * grad.sign()
                    update = update * valid_mask.unsqueeze(-1)
                    xyz = xyz + update

                    delta = xyz - original_xyz
                    delta_norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
                    scale = torch.clamp(delta_norm / (epsilon + 1e-12), min=1.0)
                    delta = delta / scale
                    xyz = original_xyz + delta * valid_mask.unsqueeze(-1)

            if iteration % 2 == 0 or iteration == self.iterations - 1:
                with torch.no_grad():
                    mean_delta = (xyz - original_xyz).norm(dim=-1)[valid_mask.bool()].mean().item()
                    grad_norm = grad.norm().item() if grad is not None else 0.0
                    print(f"[PGD] iter {iteration + 1}/{self.iterations}, "
                          f"loss={loss.item():.4f}, |grad|={grad_norm:.4f}, "
                          f"mean_delta={mean_delta:.4f}m")

        data_dict['voxels'] = torch.cat([xyz, intensity], dim=-1)
        return data_dict
