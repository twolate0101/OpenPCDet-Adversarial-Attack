# tools/visual_utils/bev_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import os

class BEVVisualizer:
    """2D 鸟瞰图可视化器 (用于快速验证检测结果)"""
    
    @staticmethod
    def draw(points, boxes, save_path, title="BEV View", labels=None):
        """
        :param points: (N, 3) 或 (N, 4) numpy array, 已经是在 CPU 上的 xyz (或 xyzi)
        :param boxes: (M, 7) numpy array [x, y, z, dx, dy, dz, heading]
        :param save_path: 保存路径
        :param title: 图片标题
        :param labels: (M,) numpy array, 类别标签 (可选, 1=Car, 2=Ped, 3=Cyc)
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 1. 创建画布（纯黑背景，更有自动驾驶的科技感）
        fig, ax = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # 2. 画点云（根据高度 z 赋予渐变色，每隔5个点采样一次防止过密）
        pts = np.asarray(points)
        if pts.shape[1] >= 3:
            # 使用 z 轴高度着色 (viridis 色带在黑底上非常清晰)
            ax.scatter(pts[::5, 0], pts[::5, 1], s=0.1, c=pts[::5, 2], cmap='viridis', alpha=0.8)
        else:
            ax.scatter(pts[::5, 0], pts[::5, 1], s=0.1, c='gray', alpha=0.8)

        # 3. 画边界框（投影到 2D 鸟瞰图）
        for idx, box in enumerate(boxes):
            x, y, z, dx, dy, dz, heading = box
            # 计算 2D 旋转矩阵
            cos_a, sin_a = np.cos(heading), np.sin(heading)
            # 矩形的四个角点 + 闭合点
            corners = np.array([
                [-dx/2, -dy/2], [dx/2, -dy/2], [dx/2, dy/2], [-dx/2, dy/2], [-dx/2, -dy/2]
            ])
            # 旋转并平移到真实坐标
            rot_corners = np.zeros_like(corners)
            rot_corners[:, 0] = corners[:, 0] * cos_a - corners[:, 1] * sin_a + x
            rot_corners[:, 1] = corners[:, 0] * sin_a + corners[:, 1] * cos_a + y

            # 根据类别上色 (1=Car:红, 2=Pedestrian:绿, 3=Cyclist:青)
            if labels is not None and len(labels) > idx:
                label = labels[idx]
                color = 'r' if label == 1 else ('g' if label == 2 else 'c')
            else:
                color = 'r' # 默认红色
                
            ax.plot(rot_corners[:, 0], rot_corners[:, 1], c=color, linewidth=2)
            
            # 画车头方向箭头
            ax.arrow(x, y, (dx/2) * cos_a, (dx/2) * sin_a, 
                     head_width=0.5, head_length=0.3, fc=color, ec=color)

        # 4. 设置视角 (雷达坐标系：x向正前方，y向左侧)
        ax.set_xlim(0, 70)   # 重点看前方 70 米
        ax.set_ylim(-40, 40) # 左右各 40 米
        ax.set_aspect('equal')
        ax.axis('off')       # 隐藏坐标轴
        ax.set_title(title, color='white', fontsize=15) # 标题设为白色

        # 5. 保存高清图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"✅ BEV 图片已保存: {save_path}")