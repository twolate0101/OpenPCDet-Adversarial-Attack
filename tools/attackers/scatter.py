"""
Gradient-optimized Adversarial Scattered Points (CVPR 2019 Section 5.1 adaptation).

1. Finds vulnerable regions via gradient analysis
2. Places scattered independent points near those regions
3. Gradient-optimizes their positions to maximize detection suppression

Each adversarial point is placed in its own pillar voxel near vulnerable regions,
with their xyz coordinates jointly optimized via gradient descent.

Reference: Xiang, Qi, Li — "Generating 3D Adversarial Point Clouds" (CVPR 2019)
"""

import torch
import numpy as np
from .base import BaseAttacker
from .critical_points import CriticalPointFinder


class ScatterAttacker(BaseAttacker):
    """Gradient-guided scattered point placement with position optimization."""

    def __init__(self, severity=1.0, model=None, iterations=30, lr=0.05, **kwargs):
        super().__init__(severity, **kwargs)
        self.model = model
        self.iterations = iterations
        self.lr = lr

        # severity controls point count: severity=0.3 → ~200, severity=1.0 → ~800
        self.num_points = max(100, int(severity * 1000))
        self.scatter_sigma = severity * 2.0

    def _xyz_to_pillar_coords(self, x, y):
        """Convert continuous xy to pillar indices."""
        xp = torch.clamp((x / 0.16).long(), 0, 431)
        yp = torch.clamp(((y + 39.68) / 0.16).long(), 0, 495)
        return xp, yp

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
            self.model, eps=1.0, min_samples=5, top_k_pillars=64)
        centers = finder.find(data_dict)

        if len(centers) == 0:
            # Fallback: use random positions in the scene
            centers = []
            for _ in range(min(self.num_points // 5, 20)):
                cx = np.random.uniform(5, 60)
                cy = np.random.uniform(-30, 30)
                cz = np.random.uniform(-1.5, 0.5)
                centers.append((cx, cy, cz))

        # ── Phase 2: Create scattered points near vulnerable regions ──
        centers_np = np.array(centers)
        pts_per_center = self.num_points // max(len(centers), 1)

        adv_points = []
        for cx, cy, cz in centers:
            # Scatter points around each center with Gaussian spread
            offsets = torch.randn(pts_per_center, 3, device=device) * self.scatter_sigma
            cluster = torch.tensor([cx, cy, cz], device=device) + offsets
            # Clamp to scene bounds
            cluster[:, 0].clamp_(0.5, 69.0)
            cluster[:, 1].clamp_(-39.0, 39.0)
            cluster[:, 2].clamp_(-2.5, 0.5)
            adv_points.append(cluster)

        adv_points = torch.cat(adv_points, dim=0)  # (N_total, 3)
        adv_points = adv_points[:self.num_points]  # enforce budget
        N_total = adv_points.shape[0]

        adv_points.requires_grad_(True)

        # ── Phase 3: Gradient optimization ──
        optimizer = torch.optim.Adam([adv_points], lr=self.lr)

        for iteration in range(self.iterations):
            optimizer.zero_grad()

            # Build voxels from current point positions
            # Each scattered point becomes its own 1-point voxel
            xp, yp = self._xyz_to_pillar_coords(adv_points[:, 0], adv_points[:, 1])
            zp = torch.zeros(N_total, dtype=torch.long, device=device)
            new_coords = torch.stack([
                torch.full((N_total,), batch_idx, dtype=torch.long, device=device),
                zp, yp, xp
            ], dim=1)

            # Build voxels: pad to max_pts with zeros (use cat to preserve grad)
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
                    # Clamp to scene bounds
                    adv_points[:, 0].clamp_(0.5, 69.0)
                    adv_points[:, 1].clamp_(-39.0, 39.0)
                    adv_points[:, 2].clamp_(-2.5, 0.5)

        # Final reconstruction
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
              f"{N_total} points near {len(centers)} vulnerable regions, "
              f"spread={self.scatter_sigma:.1f}m, {self.iterations} iters")

        return data_dict
