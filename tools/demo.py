import argparse
import glob
from pathlib import Path

#try:
#    import open3d
#    from visual_utils import open3d_vis_utils as V
#    OPEN3D_FLAG = True
#except:
#    import mayavi.mlab as mlab
#    from visual_utils import visualize_utils as V
#   OPEN3D_FLAG = False

import numpy as np
import torch
import sys

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--attack', type=str, default='none', help='none,noise,drop 等')
    parser.add_argument('--severity', type=float, default=1.0, help='攻击强度')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    # === 动态加载攻击器 ===
    sys.path.append('..')
    try:
        from attackers import get_attacker
        attacker = get_attacker(args.attack,args.severity)
        if attacker is not None:
            logger.info(f"✅ 已启用攻击模式: {args.attack}，强度: {args.severity}")
    except ImportError:
        attacker = None
        logger.warning("⚠️ 未找到attacker模块,将以正常模式运行.")


    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    # === 【终极拦截点前移：动态劫持体素化流程】 ===
    if attacker is not None:
        import types
        original_prepare_data = demo_dataset.prepare_data
        
        def hooked_prepare_data(self, data_dict):
            """
            在数据被切分成体素前，强行拉到显存进行攻击，然后再放回内存进行体素化。
            """
            # 1. 此时 input_dict 里只有最原始的 numpy 点云
            pt_tensor = torch.from_numpy(data_dict['points']).cuda()
            
            # 2. 包装成你的 Attacker 认识的格式
            tmp_dict = {'points': pt_tensor}
            tmp_dict = attacker.forward(tmp_dict)
            
            # 3. 攻击完成后，将带毒的点云转回 CPU numpy，覆盖回去
            data_dict['points'] = tmp_dict['points'].cpu().numpy()
            
            # 4. 把带毒的数据扔给 OpenPCDet 原生的体素化算子
            return original_prepare_data(data_dict=data_dict)
        
        # 将劫持后的函数强制绑定给 demo_dataset
        demo_dataset.prepare_data = types.MethodType(hooked_prepare_data, demo_dataset)
    # ===============================================


    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)
            print(f"\n✅ 检测成功！总共找到 {len(pred_dicts[0]['pred_labels'])} 个目标。")
            print(f"坐标框 (x, y, z, dx, dy, dz, heading):\n {pred_dicts[0]['pred_boxes']}")
            print(f"置信度得分:\n {pred_dicts[0]['pred_scores']}")
            print(f"类别标签 (1=Car, 2=Pedestrian, 3=Cyclist):\n {pred_dicts[0]['pred_labels']}\n")

            # === 在这里插入可视化的代码 ===
            import matplotlib.pyplot as plt
            import numpy as np
            import os

            # 1. 把张量从 GPU 拿回到 CPU，并转成 Numpy 数组
            points = data_dict['points'][:, 1:].cpu().numpy()
            boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            # 2. 创建画布（纯黑背景，更有自动驾驶的科技感）
            fig, ax = plt.subplots(figsize=(12, 12))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

            # 3. 画点云（根据高度 z 赋予渐变色，每隔5个点画一个提速）
            ax.scatter(points[::5, 0], points[::5, 1], s=0.1, c=points[::5, 2], cmap='viridis', alpha=0.8)

            # 4. 画边界框（投影到 2D 鸟瞰图）
            for idx, box in enumerate(boxes):
                x, y, z, dx, dy, dz, heading = box
                # 计算 2D 旋转矩阵
                cos_a, sin_a = np.cos(heading), np.sin(heading)
                # 矩形的四个角点
                corners = np.array([[-dx/2, -dy/2], [dx/2, -dy/2], [dx/2, dy/2], [-dx/2, dy/2], [-dx/2, -dy/2]])
                # 旋转并平移到真实坐标
                rot_corners = np.zeros_like(corners)
                rot_corners[:, 0] = corners[:, 0] * cos_a - corners[:, 1] * sin_a + x
                rot_corners[:, 1] = corners[:, 0] * sin_a + corners[:, 1] * cos_a + y

                # 根据类别上色 (1=Car:红, 2=Pedestrian:绿, 3=Cyclist:青)
                color = 'r' if labels[idx] == 1 else ('g' if labels[idx] == 2 else 'c')
                ax.plot(rot_corners[:, 0], rot_corners[:, 1], c=color, linewidth=2)

            # 5. 设置视角 (雷达坐标系：x向正前方，y向左侧)
            ax.set_xlim(0, 70)   # 重点看前方 70 米
            ax.set_ylim(-40, 40) # 左右各 40 米
            ax.set_aspect('equal')
            ax.axis('off')       # 隐藏坐标轴

            # 6. 保存高清图片
            save_path = 'result_bev.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"📸 鸟瞰图已成功渲染并保存至: {os.path.abspath(save_path)}")
            # ==============================

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
