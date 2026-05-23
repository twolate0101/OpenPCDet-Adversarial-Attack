"""
Gradient-optimized Adversarial Object Placement (CVPR 2019 Section 5.3 adaptation).

1. Finds vulnerable regions via gradient analysis
2. Places 3D geometric objects (boxes, spheres) at those regions
3. Gradient-optimizes object position, size, and rotation to maximize attack effect

Each object is a pillar voxel filled with geometric surface points. The position,
scale, and rotation of each object are jointly optimized.

Reference: Xiang, Qi, Li — "Generating 3D Adversarial Point Clouds" (CVPR 2019)
"""

import torch
import numpy as np
from .base import BaseAttacker
from .critical_points import CriticalPointFinder


class ObjectAttacker(BaseAttacker):
    """Gradient-guided adversarial object placement with shape/pose optimization."""

    def __init__(self, severity=1.0, model=None, iterations=30, lr=0.05, **kwargs):
        super().__init__(severity, **kwargs)
        self.model = model
        self.iterations = iterations
        self.lr = lr

        self.num_objects = max(1, int(severity * 8))  # sev=0.3→2, sev=1.0→8
        self.pts_per_object = 32

    def _xyz_to_pillar_coords(self, x, y):
        xp = torch.clamp((x / 0.16).long(), 0, 431)
        yp = torch.clamp(((y + 39.68) / 0.16).long(), 0, 495)
        return xp, yp

    def _generate_box_points(self, center, size, theta, n_pts):
        """Generate points on the surface of a 3D box.

        Uses sphere-to-cube projection: samples random 3D directions, projects
        onto cube surface, scales by size, then rotates and translates.
        All operations are differentiable w.r.t. size, theta, and center.

        Args:
            center: (N, 3) object centers
            size: (N, 3) box dimensions (dx, dy, dz)
            theta: (N,) rotation angle around Z axis
            n_pts: number of surface points per object

        Returns:
            (N, n_pts, 3) surface points
        """
        N = center.shape[0]
        device = center.device

        # Random directions on sphere → project to cube surface (unit cube [-0.5, 0.5])
        with torch.no_grad():
            rand_dir = torch.randn(N, n_pts, 3, device=device)
            rand_dir = rand_dir / (rand_dir.norm(dim=-1, keepdim=True) + 1e-8)
            # Project: scale so max(|coord|) = 0.5
            max_abs = rand_dir.abs().max(dim=-1, keepdim=True).values
            pts_local = rand_dir * (0.5 / (max_abs + 1e-8))

        # Scale by size
        pts_scaled = pts_local * size.unsqueeze(1)  # (N, n_pts, 3) * (N, 1, 3) = (N, n_pts, 3)

        # Rotate around Z axis
        cos_t = torch.cos(theta).view(N, 1)
        sin_t = torch.sin(theta).view(N, 1)
        x, y, z = pts_scaled[..., 0], pts_scaled[..., 1], pts_scaled[..., 2]
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t

        return torch.stack([x_rot, y_rot, z], dim=-1) + center.unsqueeze(1)

    def forward(self, data_dict):
        vault = dict(
            voxels=data_dict['voxels'].clone(),
            voxel_coords=data_dict['voxel_coords'].clone(),
            voxel_num_points=data_dict['voxel_num_points'].clone(),
        )
        device = data_dict['voxels'].device
        batch_idx = vault['voxel_coords'][0, 0].item()

        # ── Phase 1: Find vulnerable regions ──
        finder = CriticalPointFinder(
            self.model, eps=1.5, min_samples=3, top_k_pillars=64)
        centers = finder.find(data_dict)

        if len(centers) == 0:
            centers = [(np.random.uniform(5, 60), np.random.uniform(-30, 30), -0.5)
                       for _ in range(self.num_objects)]
        centers = centers[:self.num_objects]

        # ── Phase 2: Initialize objects at vulnerable regions ──
        N = len(centers)
        center_tensor = torch.tensor(np.array(centers), dtype=torch.float32, device=device)

        # Initial size (small, ~1m) and zero rotation
        init_size = torch.full((N, 3), 0.8, device=device, dtype=torch.float32)
        init_theta = torch.zeros(N, device=device, dtype=torch.float32)

        # Learnable parameters: center delta, log-size, theta
        center_delta = torch.zeros(N, 3, device=device, requires_grad=True)
        log_size = torch.log(init_size).detach().requires_grad_(True)
        theta = init_theta.detach().requires_grad_(True)

        optim_params = [center_delta, log_size, theta]
        optimizer = torch.optim.Adam(optim_params, lr=self.lr)

        # ── Phase 3: Gradient optimization ──
        for iteration in range(self.iterations):
            optimizer.zero_grad()

            cur_center = center_tensor + center_delta
            cur_size = torch.exp(log_size).clamp(0.3, 3.0)
            cur_theta = theta

            # Generate object surface points
            obj_pts = self._generate_box_points(cur_center, cur_size, cur_theta,
                                                self.pts_per_object)

            # Convert to voxels
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

        # ── Final reconstruction ──
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
              f"{N} objects at vulnerable regions, "
              f"size={final_size.mean(dim=0).cpu().numpy()}, {self.iterations} iters")

        return data_dict
