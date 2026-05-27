"""
梯度优化对抗物体放置攻击（CVPR 2019 论文 Section 5.3 适配）。

论文 Section 5.3 "Generating 3D Adversarial Objects":
  在点云场景中放置 3D 几何物体（立方体、球体），通过梯度优化联合调整物体的位置、
  尺寸和旋转角度，最大化攻击效果。

  核心思路: 每个物体用一个可微的 3D 几何基元表示，其表面点通过 sphere-to-cube 投影生成。
  物体的位置（中心偏移）、尺寸（log-size）和绕 Z 轴旋转角度（theta）作为可学习参数，
  通过 Adam 优化器联合优化。

攻击流程:
  阶段 1 — 关键点分析: CriticalPointFinder 找到检测器最敏感的 BEV 区域
  阶段 2 — 物体初始化: 在每个脆弱区域放置小型 3D 立方体（初始尺寸 ~0.8m）
  阶段 3 — 梯度优化: Adam 优化 center_delta, log_size, theta → 最小化检测分数

可学习参数:
  - center_delta (N, 3): 物体中心相对初始脆弱区域中心的偏移
  - log_size (N, 3): 物体三轴尺寸的对数（exp 后 clamp 到 [0.3, 3.0]m）
  - theta (N,): 绕 Z 轴旋转角度

梯度流保证:
  使用 torch.cat 而非 in-place 赋值构建体素张量，确保从 loss → 体素 xyz →
  object surface points → center/size/theta 的完整可微路径。

参考: Xiang, Qi, Li — "Generating 3D Adversarial Point Clouds" (CVPR 2019, Section 5.3)
"""

import torch
import numpy as np
from .base import BaseAttacker
from .critical_points import CriticalPointFinder


class ObjectAttacker(BaseAttacker):
    """梯度引导的对抗物体放置攻击，含物体形状/姿态优化。

    对应论文 Section 5.3 的 3D 对抗物体生成方法。
    每个物体 = 一个 pillar 体素（32 个表面点），
    物体的位置、尺寸、旋转角度通过 Adam 联合优化。
    """

    def __init__(self, severity=1.0, model=None, iterations=30, lr=0.05, **kwargs):
        super().__init__(severity, **kwargs)
        self.model = model
        self.iterations = iterations
        self.lr = lr

        # severity 控制物体数量: sev=0.3 → 2, sev=1.0 → 8
        self.num_objects = max(1, int(severity * 8))
        self.pts_per_object = 32

    def _xyz_to_pillar_coords(self, x, y):
        """连续 xy 坐标 → pillar 网格索引。

        PointPillar 网格: 432×496, voxel_size [0.16, 0.16, 4]。
        """
        xp = torch.clamp((x / 0.16).long(), 0, 431)
        yp = torch.clamp(((y + 39.68) / 0.16).long(), 0, 495)
        return xp, yp

    def _generate_box_points(self, center, size, theta, n_pts):
        """生成 3D 立方体表面点（可微，对应论文 Section 5.3 的几何物体）。

        使用 sphere-to-cube 投影: 在单位球面上采样随机方向 → 投影到立方体表面
        → 按 size 缩放 → 绕 Z 轴旋转 → 平移到 center。

        所有操作（缩放、旋转、平移）对 size, theta, center 可微。

        Args:
            center: (N, 3) 物体中心
            size: (N, 3) 三轴尺寸 (dx, dy, dz)
            theta: (N,) 绕 Z 轴旋转角度
            n_pts: 每个物体的表面点数量

        Returns:
            (N, n_pts, 3) 表面点坐标
        """
        N = center.shape[0]
        device = center.device

        # 球面随机采样 → 投影到单位立方体表面 [-0.5, 0.5]
        with torch.no_grad():
            rand_dir = torch.randn(N, n_pts, 3, device=device)
            rand_dir = rand_dir / (rand_dir.norm(dim=-1, keepdim=True) + 1e-8)
            # 投影: 缩放使 max(|coord|) = 0.5
            max_abs = rand_dir.abs().max(dim=-1, keepdim=True).values
            pts_local = rand_dir * (0.5 / (max_abs + 1e-8))

        # 按物体尺寸缩放
        pts_scaled = pts_local * size.unsqueeze(1)  # (N, n_pts, 3) * (N, 1, 3)

        # 绕 Z 轴旋转（论文 Section 5.3 中物体的姿态优化）
        cos_t = torch.cos(theta).view(N, 1)
        sin_t = torch.sin(theta).view(N, 1)
        x, y, z = pts_scaled[..., 0], pts_scaled[..., 1], pts_scaled[..., 2]
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t

        return torch.stack([x_rot, y_rot, z], dim=-1) + center.unsqueeze(1)

    def forward(self, data_dict):
        """
        三阶段对抗物体放置攻击（论文 Section 5.3 方法）。

        阶段 1: CriticalPointFinder → 脆弱区域中心。
        阶段 2: 在每个脆弱区域初始化小型立方体（~0.8m, 零旋转）。
        阶段 3: Adam 联合优化 center_delta, log_size, theta。
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
            self.model, eps=1.5, min_samples=3, top_k_pillars=64)
        centers = finder.find(data_dict)

        # 梯度分析失败时的回退: 场景内随机选点
        if len(centers) == 0:
            centers = [(np.random.uniform(5, 60), np.random.uniform(-30, 30), -0.5)
                       for _ in range(self.num_objects)]
        centers = centers[:self.num_objects]

        # ── 阶段 2: 在脆弱区域初始化物体 ──
        N = len(centers)
        center_tensor = torch.tensor(np.array(centers), dtype=torch.float32, device=device)

        # 初始尺寸 ~0.8m, 零旋转
        init_size = torch.full((N, 3), 0.8, device=device, dtype=torch.float32)
        init_theta = torch.zeros(N, device=device, dtype=torch.float32)

        # 可学习参数（论文 Section 5.3 的优化变量）
        center_delta = torch.zeros(N, 3, device=device, requires_grad=True)
        log_size = torch.log(init_size).detach().requires_grad_(True)
        theta = init_theta.detach().requires_grad_(True)

        optim_params = [center_delta, log_size, theta]
        optimizer = torch.optim.Adam(optim_params, lr=self.lr)

        # ── 阶段 3: 梯度联合优化物体参数 ──
        for iteration in range(self.iterations):
            optimizer.zero_grad()

            cur_center = center_tensor + center_delta
            cur_size = torch.exp(log_size).clamp(0.3, 3.0)
            cur_theta = theta

            # 生成物体表面点（可微）
            obj_pts = self._generate_box_points(cur_center, cur_size, cur_theta,
                                                self.pts_per_object)

            # 转换为体素格式
            first_pts = obj_pts[:, 0, :]
            xp, yp = self._xyz_to_pillar_coords(first_pts[:, 0], first_pts[:, 1])
            zp = torch.zeros(N, dtype=torch.long, device=device)
            new_coords = torch.stack([
                torch.full((N,), batch_idx, dtype=torch.long, device=device),
                zp, yp, xp
            ], dim=1)

            # 使用 torch.cat 构建体素（非 in-place 赋值），保持梯度链完整
            max_pts = vault['voxels'].shape[1]
            n_actual = min(self.pts_per_object, max_pts)
            adv_xyz_pad = torch.cat([
                obj_pts[:, :n_actual],
                torch.zeros(N, max_pts - n_actual, 3, device=device)
            ], dim=1)
            adv_i_pad = torch.full((N, max_pts, 1), 0.5, device=device)
            new_voxels = torch.cat([adv_xyz_pad, adv_i_pad], dim=-1)

            data_dict['voxels'] = torch.cat([vault['voxels'], new_voxels], dim=0)
            data_dict['voxel_coords'] = torch.cat([vault['voxel_coords'], new_coords], dim=0)
            new_npts = torch.full((N,), min(self.pts_per_object, max_pts),
                                  dtype=torch.long, device=device)
            data_dict['voxel_num_points'] = torch.cat([vault['voxel_num_points'], new_npts], dim=0)

            with torch.enable_grad():
                pred_dicts, _ = self.model(data_dict)
                loss = torch.tensor(0.0, device=device)
                for pred in pred_dicts:
                    scores = pred['pred_scores']
                    if scores.numel() > 0:
                        loss = loss + scores.sum()

            if loss.requires_grad:
                loss.backward(retain_graph=False)

            optimizer.step()

        # ── 最终重建（断开梯度）──
        with torch.no_grad():
            final_center = center_tensor + center_delta
            final_size = torch.exp(log_size).clamp(0.3, 3.0)
            final_theta = theta
            obj_pts = self._generate_box_points(final_center, final_size, final_theta,
                                                self.pts_per_object)

            first_pts = obj_pts[:, 0, :]
            xp, yp = self._xyz_to_pillar_coords(first_pts[:, 0], first_pts[:, 1])
            zp = torch.zeros(N, dtype=torch.long, device=device)
            new_coords = torch.stack([
                torch.full((N,), batch_idx, dtype=torch.long, device=device),
                zp, yp, xp
            ], dim=1)

            max_pts = vault['voxels'].shape[1]
            n_actual = min(self.pts_per_object, max_pts)
            adv_xyz_pad = torch.cat([
                obj_pts[:, :n_actual],
                torch.zeros(N, max_pts - n_actual, 3, device=device)
            ], dim=1)
            adv_i_pad = torch.full((N, max_pts, 1), 0.5, device=device)
            new_voxels = torch.cat([adv_xyz_pad, adv_i_pad], dim=-1)

            data_dict['voxels'] = torch.cat([vault['voxels'], new_voxels], dim=0)
            data_dict['voxel_coords'] = torch.cat([vault['voxel_coords'], new_coords], dim=0)
            new_npts = torch.full((N,), n_actual, dtype=torch.long, device=device)
            data_dict['voxel_num_points'] = torch.cat([vault['voxel_num_points'], new_npts], dim=0)

        print(f"[Object] severity={self.severity:.2f}: "
              f"{N} 个物体 @ 脆弱区域, "
              f"尺寸={final_size.mean(dim=0).cpu().numpy()}, {self.iterations} 轮迭代")

        return data_dict
