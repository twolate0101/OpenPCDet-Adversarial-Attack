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

from visual_utils.bev_visualizer import BEVVisualizer
from visual_utils.plotly_visualizer import PlotlyVisualizer
from visual_utils.ply_exporter import PLYExporter


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
    parser.add_argument('--attack', type=str, default='none', help='none,noise,drop,pgd,perturb,spawn 等')
    parser.add_argument('--severity', type=float, default=1.0, help='攻击强度')
    parser.add_argument('--iterations', type=int, default=10, help='白盒攻击迭代次数')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    # === 攻击器加载（延迟到 model 创建后） ===
    sys.path.append('..')
    BLACKBOX_ATTACKS = {'noise', 'drop', 'geo_drop', 'ghost_template'}
    whitebox_attacker = None


    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    # === 黑盒攻击：通过 dataset._attacker 在体素化前注入 ===
    blackbox_attacker = None
    try:
        from attackers import get_attacker
        if args.attack in BLACKBOX_ATTACKS:
            blackbox_attacker = get_attacker(args.attack, args.severity)
            if blackbox_attacker is not None:
                demo_dataset._attacker = blackbox_attacker
                logger.info(f'✅ 黑盒攻击: {args.attack}, 强度: {args.severity}')
    except ImportError:
        logger.warning('⚠️ 未找到attacker模块,将以正常模式运行.')


    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # === 白盒攻击：需要 model 引用，在 GPU 推理前注入 ===
    if args.attack not in BLACKBOX_ATTACKS and args.attack != 'none':
        try:
            from attackers import get_attacker
            whitebox_attacker = get_attacker(args.attack, args.severity, model=model,
                                             iterations=args.iterations)
            if whitebox_attacker is not None:
                logger.info(f"✅ 白盒攻击: {args.attack}, 强度: {args.severity}")
        except ImportError:
            pass

    # 创建统一的可视化输出目录，
    out_dir = Path('output/viz_results')
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 可视化结果将统一保存至: {out_dir.absolute()}")

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            # 保存干净点云（用于可视化扰动热力图）
            clean_points = data_dict['points'].copy()

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            # 白盒攻击：在模型推理前对 voxels 执行梯度攻击
            if whitebox_attacker is not None:
                with torch.enable_grad():
                    data_dict = whitebox_attacker.forward(data_dict)

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

                       # ==========================================
            # 🌟 多维可视化链路 
            # ==========================================
            
            # 1. 提取数据 (注意切片 1:4 避开 batch_idx)
            adv_points_xyz = data_dict['points'][:, 1:4].cpu().numpy()
            adv_points_full = data_dict['points'][:, 1:].cpu().numpy() 
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy() # 🌟 新增：提取标签
            
            # 获取干净点云 (用于 Plotly 算扰动)
            clean_points_xyz = clean_points[:, :3]

            # 2. 调用三大可视化模块
            # [A] BEV 鸟瞰图 (传入 labels 让框变色)
            bev_path = out_dir / f"{idx:04d}_attacked_bev.png"
            BEVVisualizer.draw(
                adv_points_xyz, 
                pred_boxes, 
                bev_path, 
                title=f"BEV (Idx: {idx})", 
                labels=pred_labels  # 🌟 新增参数
            )
            
            # [B] Plotly 3D 交互热力图
            plotly_path = out_dir / f"{idx:04d}_3d_heatmap.html"
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy() # 提取标签
            PlotlyVisualizer.export_attack_analysis(
                pts_clean=clean_points_xyz, 
                pts_adv=adv_points_xyz, 
                boxes=pred_boxes, 
                labels=pred_labels, 
                save_path=plotly_path,
                title=f"Attack Analysis (Idx: {idx})"
            )
            
            # [C] PLY 导出
            ply_path = out_dir / f"{idx:04d}_adv_cloud.ply"
            PLYExporter.export(adv_points_full, ply_path)
            
            logger.info(f"🎨 多维可视化完成，文件已保存至 {out_dir}")


    logger.info('Demo done.')


if __name__ == '__main__':
    main()
