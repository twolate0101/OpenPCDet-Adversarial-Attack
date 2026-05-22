# tools/visual_utils/ply_exporter.py
import numpy as np
import os
import matplotlib.cm as cm

class PLYExporter:
    """点云 PLY 导出器 (材质显影版：截断拉伸 + Turbo伪彩色，完美区分车/树/车道线)"""

    @staticmethod
    def export(points, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        points = np.asarray(points, dtype=np.float32)
        xyz = points[:, :3]
        
        if points.shape[1] >= 4:
            intensity = points[:, 3]
            
            # 🌟 核心升级 1：百分位数截断 (Percentile Clipping)
            # 丢弃最高 1% 的极亮镜面反射噪点，将 99% 的有效材质对比度强行拉伸到最大！
            i_min = np.min(intensity)
            i_max = np.percentile(intensity, 99) 
            
            intensity_clipped = np.clip(intensity, i_min, i_max)
            if i_max - i_min > 1e-5:
                norm_i = (intensity_clipped - i_min) / (i_max - i_min)
            else:
                norm_i = np.zeros_like(intensity)
                
            # 🌟 核心升级 2：Turbo 伪彩色映射 (材质断层显影)
            # 深蓝/紫 = 树木/行人衣物 (吸光)
            # 绿/黄 = 金属车身/沥青路面 (中等反射)
            # 亮红/白 = 车道线/标志牌 (高反光玻璃微珠)
            colors_rgba = cm.turbo(norm_i)
            rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)
        else:
            # 兜底：如果没有 intensity，按 Z 轴高度着色
            z_vals = xyz[:, 2]
            z_min, z_max = np.min(z_vals), np.max(z_vals)
            z_norm = (z_vals - z_min) / (z_max - z_min) if (z_max - z_min) > 1e-5 else np.zeros_like(z_vals)
            colors_rgba = cm.viridis(z_norm) 
            rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)
        
        # 构造 Header
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(xyz)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        
        # 内存布局打包
        dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        data = np.empty(len(xyz), dtype=dtype)
        data['x'] = xyz[:, 0]
        data['y'] = xyz[:, 1]
        data['z'] = xyz[:, 2]
        data['red'] = rgb[:, 0]
        data['green'] = rgb[:, 1]
        data['blue'] = rgb[:, 2]
        
        with open(save_path, 'wb') as f:
            f.write(header.encode('ascii'))
            data.tofile(f)
            
        print(f"✅ 材质显影 PLY (Turbo色带) 已保存: {save_path} ({len(xyz)} 个点)")