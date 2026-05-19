# 提前把空壳写好，后续组员写完自己的类，在这里注册
from .base import BaseAttacker
import torch

class TestAttacker(BaseAttacker):
    def forward(self, data_dict):
        # 极简劫持测试：直接把所有的点云坐标清零
        print(">>> 拦截器已触发！正在清空点云数据... <<<")
        data_dict['points'] = torch.zeros_like(data_dict['points'])
        return data_dict

class NoiseAttacker(BaseAttacker):
    """
    点坐标微扰攻击 (CVPR 2019 论文中提到的基础攻击之一)
    给所有点的 XYZ 坐标加上高斯噪声，模拟传感器受干扰或物理微扰。
    """
    def forward(self, data_dict):
        points = data_dict['points']
        # severity 控制噪声的标准差（比如 severity=0.2 意味着大约 20cm 的抖动）
        # points 的前三列是 x, y, z；第四列是反射率强度（我们不破坏反射率）
        noise = torch.randn_like(points[:, :3]) * self.severity
        
        # 将噪声注入到物理坐标中
        points[:, :3] += noise
        
        data_dict['points'] = points
        return data_dict

def get_attacker(attack_type, severity):
    if attack_type == 'none':
        return None
    # 假设后续有 noise 和 drop 攻击
    # elif attack_type == 'noise':
    #     from .noise import NoiseAttacker
    #     return NoiseAttacker(severity)
    elif attack_type == 'test':
        return TestAttacker(severity)
    elif attack_type == 'noise':
        return NoiseAttacker(severity)
    else:
        raise ValueError(f"不支持的攻击类型: {attack_type}")