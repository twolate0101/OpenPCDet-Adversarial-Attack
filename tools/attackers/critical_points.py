"""
Critical Point Finder for PointPillar (adapted from CVPR 2019 paper Section 5.2-5.3).

Original paper: Uses PointNet's critical points (max-pool activators) + DBSCAN to find
"vulnerable regions" on target-class objects. Then places adversarial clusters there.

Adapted for PointPillar detection: Uses gradient magnitude w.r.t. voxel xyz coordinates
as a proxy for "criticality". Points whose positions most affect detection scores are
"critical". DBSCAN clusters their spatial locations → vulnerable region centers.

Reference: Xiang, Qi, Li — "Generating 3D Adversarial Point Clouds" (CVPR 2019)
"""

import numpy as np
import torch
from sklearn.cluster import DBSCAN


class CriticalPointFinder:
    """Find vulnerable spatial regions in a 3D detection scene via gradient analysis."""

    def __init__(self, model, eps=0.5, min_samples=3, top_k_pillars=64):
        """
        Args:
            model: PointPillar (or similar) detector with differentiable forward
            eps: DBSCAN radius in meters for clustering critical points
            min_samples: DBSCAN min_samples for a cluster
            top_k_pillars: Number of most-critical pillars to consider
        """
        self.model = model
        self.eps = eps
        self.min_samples = min_samples
        self.top_k_pillars = top_k_pillars

    def find(self, data_dict):
        """
        Find vulnerable region centers via gradient analysis + DBSCAN clustering.

        Args:
            data_dict: Post-collate-batch data on GPU with 'voxels', 'voxel_coords',
                       'voxel_num_points'. Must have requires_grad enabled on voxels.

        Returns:
            list of (x, y, z) tuples: cluster centers of vulnerable regions.
            Returns empty list if no clusters found.
        """
        voxels = data_dict['voxels']  # (M, max_pts, C)
        coords = data_dict['voxel_coords']  # (M, 4): (batch, z, y, x)
        num_points = data_dict['voxel_num_points']  # (M,)
        max_pts = voxels.shape[1]

        # 1. Compute per-point gradient magnitude w.r.t. detection scores
        xyz = voxels[:, :, :3].clone()
        intensity = voxels[:, :, 3:].clone()

        # Build valid mask
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

        # 2. Compute per-point gradient magnitude
        point_grad_norm = (grad * valid_mask).norm(dim=-1)  # (M, max_pts)

        # 3. Compute per-pillar "criticality" = max gradient across its points
        pillar_criticality = point_grad_norm.max(dim=-1).values  # (M,)

        # 4. Select top-K most critical pillars
        k = min(self.top_k_pillars, pillar_criticality.shape[0])
        top_k_indices = torch.topk(pillar_criticality, k).indices

        # 5. Get representative xyz for each critical pillar (use the max-gradient point)
        critical_pts_xyz = []
        for idx in top_k_indices:
            pgrad_norm = point_grad_norm[idx]  # (max_pts,)
            npts = int(num_points[idx].item())
            if npts > 0:
                best_pt_idx = torch.argmax(pgrad_norm[:npts])
                pt_xyz = xyz[idx, best_pt_idx].detach().cpu().numpy()
                critical_pts_xyz.append(pt_xyz)

        if len(critical_pts_xyz) < self.min_samples:
            # Not enough critical points for clustering — return their centroids directly
            if len(critical_pts_xyz) > 0:
                pts = np.array(critical_pts_xyz)
                return [tuple(pts.mean(axis=0))]
            return []

        critical_pts_xyz = np.array(critical_pts_xyz)

        # 6. DBSCAN clustering on critical point locations (xy plane only, since
        #    detection is primarily 2D for PointPillar)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(
            critical_pts_xyz[:, :2]  # cluster in xy (bird's eye view)
        )

        # 7. Compute cluster centers (in 3D: xy center + mean z)
        labels = clustering.labels_
        centers = []
        for label in set(labels):
            if label == -1:  # noise
                continue
            cluster_pts = critical_pts_xyz[labels == label]
            center = cluster_pts.mean(axis=0)
            centers.append(tuple(center))

        return centers
