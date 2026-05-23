import torch
from .base import BaseAttacker


class ObjectAttacker(BaseAttacker):
    """
    有意义物体点云插入攻击 (论文 Section 3.3.3, object.py).

    在场景中放置由基本 3D 几何体表面采样生成的"假目标"，模拟攻击者
    注入外形逼真的伪造物体以诱导检测器产生虚警。

    支持的几何体:
      - sphere  (球体)
      - box     (长方体, 模拟车辆)
      - cylinder (柱体, 模拟行人/电线杆)
      - cone    (锥体)

    severity 控制物体数量和尺寸:
      severity=0.2  → 2 个小物体 (~200 pts, 0.5m 尺寸)
      severity=0.5  → 5 个中物体 (~500 pts, 1.5m 尺寸)
      severity=1.0  → 10 个大物体 (~1000 pts, 3.0m 尺寸)

    放置范围: x∈[10,60] y∈[-30,30] z∈[-2,0]
    """

    SHAPES = ['sphere', 'box', 'cylinder', 'cone']

    def _sample_sphere(self, n, radius, device):
        """球面均匀采样"""
        u = torch.empty(n, device=device).uniform_(-1, 1)
        theta = torch.acos(u)
        phi = torch.empty(n, device=device).uniform_(0, 2 * 3.14159)
        x = radius * torch.sin(theta) * torch.cos(phi)
        y = radius * torch.sin(theta) * torch.sin(phi)
        z = radius * torch.cos(theta)
        return torch.stack([x, y, z], dim=1)

    def _sample_box(self, n, w, h, d, device):
        """长方体表面采样 (nx, ny, nz 各面等分)"""
        faces = 6
        n_per_face = n // faces
        points = []
        # 六个面: ±x, ±y, ±z
        # +x face (y∈[-h/2,h/2], z∈[-d/2,d/2])
        pw = w / 2
        ph = h / 2
        pd = d / 2
        for axis in range(3):
            for sign in [-1, 1]:
                nf = n_per_face
                u = torch.empty(nf, device=device).uniform_(-1, 1)
                v = torch.empty(nf, device=device).uniform_(-1, 1)
                p = torch.zeros(nf, 3, device=device)
                if axis == 0:  # ±x
                    p[:, 0] = sign * pw
                    p[:, 1] = u * ph
                    p[:, 2] = v * pd
                elif axis == 1:  # ±y
                    p[:, 0] = u * pw
                    p[:, 1] = sign * ph
                    p[:, 2] = v * pd
                else:  # ±z
                    p[:, 0] = u * pw
                    p[:, 1] = v * ph
                    p[:, 2] = sign * pd
                points.append(p)
        return torch.cat(points, dim=0)[:n]

    def _sample_cylinder(self, n, radius, height, device):
        """柱体表面采样 (侧面 + 上下底面)"""
        n_side = n * 2 // 3
        n_cap = (n - n_side) // 2
        # 侧面
        theta = torch.empty(n_side, device=device).uniform_(0, 2 * 3.14159)
        z_side = torch.empty(n_side, device=device).uniform_(-height / 2, height / 2)
        x_side = radius * torch.cos(theta)
        y_side = radius * torch.sin(theta)
        side = torch.stack([x_side, y_side, z_side], dim=1)
        # 上下底面
        r_cap = radius * torch.sqrt(torch.rand(n_cap, device=device))
        theta_cap = torch.empty(n_cap, device=device).uniform_(0, 2 * 3.14159)
        x_cap = r_cap * torch.cos(theta_cap)
        y_cap = r_cap * torch.sin(theta_cap)
        top = torch.stack([x_cap, y_cap, torch.full((n_cap,), height / 2, device=device)], dim=1)
        bottom = torch.stack([x_cap, y_cap, torch.full((n_cap,), -height / 2, device=device)], dim=1)
        return torch.cat([side, top, bottom], dim=0)[:n]

    def _sample_cone(self, n, radius, height, device):
        """锥体表面采样"""
        z = torch.empty(n, device=device).uniform_(0, height)
        r_at_z = radius * (1 - z / height)
        theta = torch.empty(n, device=device).uniform_(0, 2 * 3.14159)
        x = r_at_z * torch.cos(theta)
        y = r_at_z * torch.sin(theta)
        z = z - height / 2
        return torch.stack([x, y, z], dim=1)

    def forward(self, data_dict):
        points = data_dict['points']
        device = points.device
        severity = self.severity

        n_objects = max(1, int(severity * 10))           # 1~10 个物体
        base_size = 0.5 + severity * 2.5                  # 0.7~3.0m 基础尺寸
        n_pts_per_obj = max(100, int(severity * 1000))    # 100~1000 点/物体

        spawned_list = []
        for i in range(n_objects):
            shape = self.SHAPES[i % len(self.SHAPES)]
            cx = torch.empty(1, device=device).uniform_(10, 60)
            cy = torch.empty(1, device=device).uniform_(-30, 30)
            cz = torch.empty(1, device=device).uniform_(-2, 0)

            if shape == 'sphere':
                r = base_size / 2
                pts_local = self._sample_sphere(n_pts_per_obj, r, device)
            elif shape == 'box':
                w = base_size * torch.empty(1, device=device).uniform_(0.8, 1.2).item()
                h = base_size * torch.empty(1, device=device).uniform_(0.4, 0.8).item()
                d = base_size * torch.empty(1, device=device).uniform_(0.6, 1.0).item()
                pts_local = self._sample_box(n_pts_per_obj, w, d, h, device)
            elif shape == 'cylinder':
                r = base_size / 3
                h = base_size * torch.empty(1, device=device).uniform_(1.0, 2.0).item()
                pts_local = self._sample_cylinder(n_pts_per_obj, r, h, device)
            elif shape == 'cone':
                r = base_size / 3
                h = base_size * torch.empty(1, device=device).uniform_(1.0, 2.0).item()
                pts_local = self._sample_cone(n_pts_per_obj, r, h, device)
            else:
                continue

            pts_world = pts_local + torch.tensor([cx.item(), cy.item(), cz.item()], device=device)
            intensity = torch.zeros(pts_world.shape[0], 1, device=device)
            obj = torch.cat([pts_world, intensity], dim=1)
            spawned_list.append(obj)

        spawned = torch.cat(spawned_list, dim=0)
        data_dict['points'] = torch.cat([points, spawned], dim=0)

        print(f"[Object] {points.shape[0]} orig + {spawned.shape[0]} fake "
              f"({n_objects} objects, size~{base_size:.1f}m) → "
              f"{data_dict['points'].shape[0]} total")
        return data_dict
