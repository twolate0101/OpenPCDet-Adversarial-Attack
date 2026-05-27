"""
攻击器注册中心 — 统一管理所有对抗攻击方法。

攻击分类体系（基于 CVPR 2019 "Generating 3D Adversarial Point Clouds"）:

  BLACKBOX（黑盒攻击）: 不依赖模型梯度的攻击
    - noise:    点坐标高斯噪声（模拟传感器干扰）
    - drop:     随机点云删除（模拟遮挡/信号丢失）
    - geo_drop: 几何感知删除（优先删除距质心近的目标区域点）

  WHITEBOX（白盒扰动攻击）: 基于模型梯度的点坐标扰动
    - pgd:     PGD 迭代梯度攻击（投影梯度下降，L2 epsilon 球约束）
    - perturb: C&W 风格体素扰动（Adam + 软正则 λ，论文 Section 4 + Eq. 2）

  BLACKBOX_INSERTION（黑盒插入攻击）: 模板复制粘贴
    - ghost_template: 幽灵模板注入（真实目标模板 + 空旷区域放置，物理可实现）

  WHITEBOX_INSERTION（白盒插入攻击）: 基于梯度分析的对抗点/物体插入
    - spawn:   对抗点簇生成（论文 Section 5.2，梯度分析 + DBSCAN + 簇优化）
    - scatter: 对抗散点放置（论文 Section 5.1 扩展，独立散点 + sign-梯度优化）
    - object:  对抗物体放置（论文 Section 5.3，3D 几何物体 + 形状/姿态联合优化）

使用方式:
  from tools.attackers import get_attacker
  attacker = get_attacker('pgd', severity=0.5, model=model, iterations=10)
  data_dict = attacker(data_dict)
"""

from .base import BaseAttacker
import torch
import numpy as np


class TestAttacker(BaseAttacker):
    """测试攻击器: 将所有点云坐标清零，用于验证攻击管线是否正常拦截。"""
    def forward(self, data_dict):
        print(">>> 拦截器已触发！正在清空点云数据... <<<")
        data_dict['points'] = torch.zeros_like(data_dict['points'])
        return data_dict


class NoiseAttacker(BaseAttacker):
    """
    点坐标高斯噪声攻击（CVPR 2019 论文中的基础黑盒攻击基准）。

    给所有点的 XYZ 坐标加上高斯噪声，模拟 LiDAR 传感器受干扰或物理微振。
    severity 控制噪声标准差（米），例如 severity=0.2 ≈ 20cm 随机抖动。
    不修改反射率强度值。
    """
    def forward(self, data_dict):
        points = data_dict['points']
        is_numpy = isinstance(points, np.ndarray)
        if is_numpy:
            points = torch.from_numpy(points).float()
        noise = torch.randn_like(points[:, :3]) * self.severity
        points[:, :3] += noise
        data_dict['points'] = points.cpu().numpy() if is_numpy else points
        return data_dict


def get_attacker(attack_type, severity, **kwargs):
    """根据攻击类型名称创建对应的攻击器实例。

    Args:
        attack_type: 攻击类型字符串 ('none', 'noise', 'drop', 'spawn',
                     'pgd', 'perturb', 'scatter', 'object', 'test')
        severity: 攻击强度 (0.0 ~ 1.0)
        **kwargs: 传递给攻击器的额外参数（如 model, iterations, lr）

    Returns:
        BaseAttacker 实例，或 None（attack_type='none' 时）
    """
    if attack_type == 'none':
        return None
    elif attack_type == 'test':
        return TestAttacker(severity)
    elif attack_type == 'noise':
        return NoiseAttacker(severity)
    elif attack_type == 'drop':
        from .drop import DropAttacker
        return DropAttacker(severity)
    elif attack_type == 'geo_drop':
        from .geo_drop import GeoDropAttacker
        return GeoDropAttacker(severity)
    elif attack_type == 'spawn':
        from .spawn import SpawnAttacker
        return SpawnAttacker(severity, **kwargs)
    elif attack_type == 'pgd':
        from .pgd import PGDAttacker
        return PGDAttacker(severity, **kwargs)
    elif attack_type == 'perturb':
        from .perturbation import PerturbationAttacker
        return PerturbationAttacker(severity, **kwargs)
    elif attack_type == 'scatter':
        from .scatter import ScatterAttacker
        return ScatterAttacker(severity, **kwargs)
    elif attack_type == 'object':
        from .object import ObjectAttacker
        return ObjectAttacker(severity, **kwargs)
    elif attack_type == 'ghost_template':
        from .ghost_template import GhostTemplateAttacker
        return GhostTemplateAttacker(severity, **kwargs)
    else:
        raise ValueError(f"不支持的攻击类型: {attack_type}")
