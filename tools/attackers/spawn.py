"""
Gradient-optimized Adversarial Cluster Spawn (CVPR 2019 Section 5.2 adaptation).

1. Finds vulnerable regions via gradient analysis (CriticalPointFinder)
2. Places point clusters at those regions
3. Gradient-optimizes cluster positions to maximize detection suppression

Reference: Xiang, Qi, Li — "Generating 3D Adversarial Point Clouds" (CVPR 2019)
"""

import torch
import numpy as np
from .base import BaseAttacker
from .critical_points import CriticalPointFinder


class SpawnAttacker(BaseAttacker):
    """Gradient-guided cluster spawn with DBSCAN-based vulnerable region detection."""

    def __init__(self, severity=1.0, model=None, iterations=30, lr=0.05, **kwargs):
        super().__init__(severity, **kwargs)
        self.model = model
        self.iterations = iterations
        self.lr = lr

        # severity controls number of clusters: severity=0.5 → ~3 clusters, 1.0 → ~6
        self.num_clusters = max(1, int(severity * 10))
        self.pts_per_cluster = 32
        self.cluster_radius = 0.8 * severity

    def _xyz_to_pillar_coords(self, x, y, z):
        """Convert continuous xyz to (batch, z_idx, y_idx, x_idx) for voxel_coords."""
        xp = torch.clamp((x / 0.16).long(), 0, 431)    # (0, 69.12) / 0.16 → 432
        yp = torch.clamp(((y + 39.68) / 0.16).long(), 0, 495)  # 496
        return xp, yp

    def _add_voxels(self, data_dict, adv_xyz, adv_intensity):
        """Insert new adversarial voxels into the data_dict.

        Args:
            data_dict: existing post-collate GPU data
            adv_xyz: (N_new, 32, 3) adversarial point coordinates
            adv_intensity: (N_new, 32, 1) fabricatd intensity values

        Returns:
            updated data_dict with expanded voxels, voxel_coords, voxel_num_points
        """
        device = data_dict['voxels'].device
        N_new = adv_xyz.shape[0]
        max_pts = data_dict['voxels'].shape[1]

        # Pad or trim to match max_pts
        if max_pts > adv_xyz.shape[1]:
            pad_xyz = torch.zeros(N_new, max_pts - adv_xyz.shape[1], 3, device=device)
            pad_i = torch.zeros(N_new, max_pts - adv_intensity.shape[1], 1, device=device)
            adv_xyz = torch.cat([adv_xyz, pad_xyz], dim=1)
            adv_intensity = torch.cat([adv_intensity, pad_i], dim=1)

        # Build voxel_coords for each new pillar
        # Use the first point of each cluster to determine pillar position
        first_pts = adv_xyz[:, 0, :]  # (N_new, 3) — cluster center
        xp, yp = self._xyz_to_pillar_coords(first_pts[:, 0], first_pts[:, 1], first_pts[:, 2])
        batch_idx = data_dict['voxel_coords'][0, 0].item()
        zp = torch.zeros(N_new, dtype=torch.long, device=device)
        new_coords = torch.stack([
            torch.full((N_new,), batch_idx, dtype=torch.long, device=device),
            zp, yp, xp
        ], dim=1)  # (N_new, 4)

        # Build new voxels tensor
        new_voxels = torch.cat([adv_xyz, adv_intensity], dim=-1)  # (N_new, max_pts, 4)

        # Point counts (excluding padding)
        pts_per = min(self.pts_per_cluster, max_pts)
        new_num_pts = torch.full((N_new,), pts_per, dtype=torch.long, device=device)

        # Merge
        data_dict['voxels'] = torch.cat([data_dict['voxels'].float(), new_voxels.float()], dim=0)
        data_dict['voxel_coords'] = torch.cat([data_dict['voxel_coords'], new_coords], dim=0)
        data_dict['voxel_num_points'] = torch.cat([
            data_dict['voxel_num_points'], new_num_pts.to(data_dict['voxel_num_points'].device)])

        return data_dict

    def forward(self, data_dict):
        """
        Phase 1: Find vulnerable regions via gradient analysis.
        Phase 2: Place adversarial clusters at those regions.
        Phase 3: Gradient-optimize cluster positions to maximize detection suppression.
        """
        original_voxels = data_dict['voxels'].clone()
        original_coords = data_dict['voxel_coords'].clone()
        original_num_pts = data_dict['voxel_num_points'].clone()

        # ── Phase 1: Find vulnerable regions ──
        finder = CriticalPointFinder(
            self.model, eps=0.8, min_samples=2,
            top_k_pillars=max(64, self.num_clusters * 10))
        centers = finder.find(data_dict)

        # Fallback: if gradient analysis finds nothing, use random regions
        if len(centers) == 0:
            # Pick random existing pillar centers
            voxels = data_dict['voxels'][:, :, :3]
            npts = data_dict['voxel_num_points']
            idxs = np.random.choice(len(npts), min(self.num_clusters, len(npts)), replace=False)
            for idx in idxs:
                n = max(1, npts[idx].item())
                center = voxels[idx, :n].mean(dim=0).cpu().numpy()
                centers.append(tuple(center))

        # Limit number of clusters
        centers = centers[:self.num_clusters]

        # ── Phase 2: Create adversarial clusters ──
        device = data_dict['voxels'].device
        N = len(centers)
        r = self.cluster_radius

        # Initialize cluster points: center + random offsets within radius
        adv_xyz_list = []
        for cx, cy, cz in centers:
            offsets = (torch.rand(32, 3, device=device) - 0.5) * 2 * r
            cluster = torch.tensor([cx, cy, cz], device=device) + offsets
            adv_xyz_list.append(cluster)
        adv_xyz_init = torch.stack(adv_xyz_list)  # (N, 32, 3)

        # Fabricate intensities (medium reflectivity)
        adv_intensity = torch.full((N, 32, 1), 0.3, device=device)

        # ── Phase 3: Gradient-optimize cluster positions ──
        # Restore original data and add adversarial voxels
        data_dict['voxels'] = original_voxels.clone()
        data_dict['voxel_coords'] = original_coords.clone()
        data_dict['voxel_num_points'] = original_num_pts.clone()

        adv_xyz = adv_xyz_init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([adv_xyz], lr=self.lr)

        for iteration in range(self.iterations):
            optimizer.zero_grad()

            # Rebuild data_dict with current adversarial points
            data_dict['voxels'] = original_voxels.clone()
            data_dict['voxel_coords'] = original_coords.clone()
            data_dict['voxel_num_points'] = original_num_pts.clone()
            data_dict = self._add_voxels(data_dict, adv_xyz, adv_intensity)

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
                if adv_xyz.grad is not None:
                    adv_xyz -= self.lr * adv_xyz.grad.sign()
                    adv_xyz.grad.zero_()

        # Final reconstruction with optimized points
        data_dict['voxels'] = original_voxels.clone()
        data_dict['voxel_coords'] = original_coords.clone()
        data_dict['voxel_num_points'] = original_num_pts.clone()
        data_dict = self._add_voxels(data_dict, adv_xyz.detach(), adv_intensity)

        print(f"[Spawn] severity={self.severity:.2f}: "
              f"{N} clusters at {len(centers)} vulnerable regions, "
              f"r={r:.1f}m, {self.iterations} iters")

        return data_dict
