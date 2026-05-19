# 提前把空壳写好，后续组员写完自己的类，在这里注册
from .base import BaseAttacker
import torch

class TestAttacker(BaseAttacker):
    def forward(self, data_dict):
        # 极简劫持测试：直接把所有的点云坐标清零
        print(">>> 拦截器已触发！正在清空点云数据... <<<")
        data_dict['points'] = torch.zeros_like(data_dict['points'])
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
    else:
        raise ValueError(f"不支持的攻击类型: {attack_type}")