import torch

class BaseAttacker:
    def __init__(self, severity=1.0, **kwargs):
        # severity 控制攻击强度，例如扰动的方差、删除点的比例
        self.severity = severity

    def forward(self, data_dict):
        """
        核心拦截接口。
        组员必须重写这个方法，且只能使用 torch 原生算子！
        
        参数:
            data_dict: OpenPCDet 原生的字典。
                       里面最重要的是 data_dict['points']，
                       形状通常是 (N, 5)，即 [batch_idx, x, y, z, intensity]。
        返回:
            修改后的 data_dict
        """
        raise NotImplementedError("每个攻击方法必须实现 forward 逻辑")