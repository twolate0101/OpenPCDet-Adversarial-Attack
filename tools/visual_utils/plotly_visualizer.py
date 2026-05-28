# tools/visual_utils/plotly_visualizer.py
import plotly.graph_objects as go
import numpy as np
import os
from scipy.spatial import cKDTree

class PlotlyVisualizer:
    """3D 交互式热力图可视化器 (学术白底版 + 防眩晕网格 + 独立车头线)"""

    @staticmethod
    def export_attack_analysis(pts_clean, pts_adv, boxes, save_path, title="Attack Analysis", labels=None):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig = go.Figure()

        # 1. 降采样 + 扰动计算（处理 clean/adv 点数不一致的情况）
        #    - 扰动/删除攻击: 点数相等或 adv 更少，按索引对应计算
        #    - 插入攻击: adv 更多，共同部分按索引对应，注入部分用最近邻距离
        n_clean, n_adv = len(pts_clean), len(pts_adv)
        n_common = min(n_clean, n_adv)
        n_sample = min(25000, n_adv)

        if n_clean == n_adv:
            # 点数相等：直接按索引对应
            sample_idx = np.random.choice(n_adv, n_sample, replace=False)
            pts_c = pts_clean[sample_idx, :3]
            pts_a = pts_adv[sample_idx, :3]
            perturbation_delta = np.linalg.norm(pts_a - pts_c, axis=1)
        else:
            # 点数不等（插入/删除攻击）
            sample_common = min(n_sample, n_common)
            sample_extra = n_sample - sample_common

            idx_common = np.random.choice(n_common, sample_common, replace=False)
            pts_a_common = pts_adv[idx_common, :3]
            pts_c_common = pts_clean[idx_common, :3]
            delta_common = np.linalg.norm(pts_a_common - pts_c_common, axis=1)

            if n_adv > n_common and sample_extra > 0:
                # 插入攻击：注入点用最近邻距离作为扰动度量
                idx_extra = np.random.choice(
                    np.arange(n_common, n_adv), min(sample_extra, n_adv - n_common), replace=False
                )
                pts_a_extra = pts_adv[idx_extra, :3]
                tree = cKDTree(pts_clean[:, :3])
                delta_extra, _ = tree.query(pts_a_extra, k=1)
                pts_a = np.vstack([pts_a_common, pts_a_extra])
                perturbation_delta = np.concatenate([delta_common, delta_extra])
            elif n_clean > n_common and sample_extra > 0:
                # 删除攻击：clean 中被删的点不参与可视化
                pts_a = pts_a_common
                perturbation_delta = delta_common
            else:
                pts_a = pts_a_common
                perturbation_delta = delta_common
        
        # 🌟 绘制毒化点云 (学术白底专属配置)
        fig.add_trace(go.Scatter3d(
            x=pts_a[:, 0], y=pts_a[:, 1], z=pts_a[:, 2],
            mode='markers',
            marker=dict(
                size=1.2,                  # 🌟 极小点径：防止点云糊成一团，保留车辆边缘细节
                color=perturbation_delta, 
                colorscale='Reds',         # 🌟 白底神器：低扰动隐身(浅粉)，高扰动刺眼(深红)
                colorbar=dict(
                    title="Delta (m)", 
                    thickness=15, 
                    x=1.0, 
                    tickfont=dict(color='black')
                ), 
                opacity=0.8                # 稍微透明，增加 3D 层次感
            ),
            hovertemplate='<b>Adv Point</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>Delta: %{customdata:.4f}m<extra></extra>',
            customdata=perturbation_delta,
            name='Adversarial Points'
        ))

        # 3. 绘制 3D Bounding Box (多类别变色)
        color_map = {1: 'red', 2: 'green', 3: 'cyan'} # 1=Car, 2=Ped, 3=Cyc
        edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]

        for i, box in enumerate(boxes):
            x, y, z, l, w, h, yaw = box
            corners_local = np.array([
                [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2], [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
                [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2], [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2]
            ])
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            corners = np.dot(corners_local, R.T) + np.array([x, y, z])
            
            # 构造 12 条边
            box_x, box_y, box_z = [], [], []
            for edge_start, edge_end in edges:
                box_x.extend([corners[edge_start, 0], corners[edge_end, 0], None])
                box_y.extend([corners[edge_start, 1], corners[edge_end, 1], None])
                box_z.extend([corners[edge_start, 2], corners[edge_end, 2], None])

            label = labels[i] if labels is not None else 1
            box_color = color_map.get(label, 'black') # 白底下默认用黑色

            # 画框
            fig.add_trace(go.Scatter3d(
                x=box_x, y=box_y, z=box_z, mode='lines',
                line=dict(color=box_color, width=4), 
                name=f'Class {label}',
                showlegend=False 
            ))
            
            # 🌟 画车头方向指示线 (小棍子：从底面中心指向前面中心)
            bottom_center = np.mean(corners[0:4], axis=0)
            front_center = np.mean([corners[1], corners[2], corners[5], corners[6]], axis=0)
            
            fig.add_trace(go.Scatter3d(
                x=[bottom_center[0], front_center[0]], 
                y=[bottom_center[1], front_center[1]], 
                z=[bottom_center[2], front_center[2]],
                mode='lines',
                line=dict(color='darkblue', width=7), # 🌟 深蓝色粗线，与框形成强烈对比
                showlegend=False
            ))

        # 4. 🌟 设置场景布局 (学术白底 + 防眩晕网格)
        fig.update_layout(
            title=dict(text=title, font=dict(color='black')),
            scene=dict(
                aspectmode='data', # 强制真实物理比例 (扁长方形街道)
                xaxis=dict(
                    title='X (Forward)', 
                    gridcolor='lightgray',       # 🌟 浅灰网格，建立空间参考系
                    zerolinecolor='lightgray',
                    color='black'                # 坐标轴字体黑色
                ),
                yaxis=dict(
                    title='Y (Left)', 
                    gridcolor='lightgray', 
                    zerolinecolor='lightgray',
                    color='black'
                ),
                zaxis=dict(
                    title='Z (Up)', 
                    gridcolor='lightgray', 
                    zerolinecolor='lightgray',
                    color='black'
                ),
                bgcolor='white',                 # 🌟 纯白背景
                camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.8))
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig.write_html(save_path, include_plotlyjs='cdn')
        print(f"✅ Plotly 3D 交互网页(学术白底版)已保存: {save_path}")