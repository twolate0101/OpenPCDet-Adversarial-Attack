"""
C&W Perturbation 攻击独立测试脚本。

与 PGD 不同，C&W 使用 Adam 优化器 + 软正则 λ，更精细的扰动。
攻击完成后自动输出对比报告并保存并排 BEV 图。

用法:
    cd /root/OpenPCDet/tools
    python test_perturb.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt ../ckpts/pointpillar_7728.pth \
        --data_path ../data/kitti/testing/velodyne/000008.bin \
        --severity 0.5 --iterations 50
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import sys

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

CLS_NAMES = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
CLS_COLORS = {1: 'r', 2: 'g', 3: 'c'}


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
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
        input_dict = {'points': points, 'frame_id': index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='C&W Perturbation Attack Test')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml')
    parser.add_argument('--data_path', type=str, default='../data/kitti/testing/velodyne/000008.bin')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ext', type=str, default='.bin')
    parser.add_argument('--severity', type=float, default=0.5, help='controls lambda_reg (higher = stronger attack)')
    parser.add_argument('--iterations', type=int, default=50, help='Adam optimization steps')
    parser.add_argument('--lr', type=float, default=0.01, help='Adam learning rate')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def clone_data_dict(data_dict):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}


def run_inference(model, data_dict):
    d = clone_data_dict(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(d)
    return pred_dicts[0]


def draw_bev(ax, points, boxes, labels, title):
    ax.set_facecolor('black')
    ax.scatter(points[::5, 0], points[::5, 1], s=0.1, c=points[::5, 2], cmap='viridis', alpha=0.8)
    for box, label in zip(boxes, labels):
        x, y, z, dx, dy, dz, heading = box
        cos_a, sin_a = np.cos(heading), np.sin(heading)
        corners = np.array([[-dx / 2, -dy / 2], [dx / 2, -dy / 2], [dx / 2, dy / 2], [-dx / 2, dy / 2], [-dx / 2, -dy / 2]])
        rot_corners = np.zeros_like(corners)
        rot_corners[:, 0] = corners[:, 0] * cos_a - corners[:, 1] * sin_a + x
        rot_corners[:, 1] = corners[:, 0] * sin_a + corners[:, 1] * cos_a + y
        ax.plot(rot_corners[:, 0], rot_corners[:, 1], c=CLS_COLORS.get(int(label), 'w'), linewidth=1.5)
    ax.set_xlim(0, 70)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=13)


def print_comparison(before, after):
    print(f'\n{"="*62}')
    print(f'  C&W Perturbation Attack Comparison Report')
    print(f'{"="*62}')
    print(f'  {"":<30} {"Before":>12} {"After":>12}')
    print(f'  {"-"*30} {"-"*12} {"-"*12}')
    print(f'  {"Total detections":<30} {before["num"]:>12} {after["num"]:>12}')
    for c in CLS_NAMES.values():
        print(f'  {c:<30} {before["cls_counts"][c]:>12} {after["cls_counts"][c]:>12}')
    asr = (before['num'] - after['num']) / max(before['num'], 1) * 100
    print(f'  {"-"*30} {"-"*12} {"-"*12}')
    print(f'  {"ASR (count-based)":<30} {"":>12} {asr:>11.1f}%')
    print(f'\n  --- Confidence Scores ---')
    print(f'  {"Mean":<30} {before["score_mean"]:>12.4f} {after["score_mean"]:>12.4f}')
    print(f'  {"Median":<30} {before["score_median"]:>12.4f} {after["score_median"]:>12.4f}')
    print(f'  {"Min":<30} {before["score_min"]:>12.4f} {after["score_min"]:>12.4f}')
    if before['score_mean'] > 0:
        drop = (before['score_mean'] - after['score_mean']) / before['score_mean'] * 100
        print(f'  {"ASR (score drop)":<30} {"":>12} {drop:>11.1f}%')
    print(f'\n  --- Top-5 Scores ---')
    print(f'  Before: {list(before["scores"][:5])}')
    print(f'  After:  {list(after["scores"][:5])}')
    print(f'{"="*62}\n')


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------C&W Perturbation Attack Test-------------------------')
    logger.info(f'Severity={args.severity}  Iterations={args.iterations}  LR={args.lr}')
    logger.info(f'Data: {args.data_path}')

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total samples: {len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    sys.path.append('..')
    from attackers.perturbation import PerturbationAttacker
    attacker = PerturbationAttacker(severity=args.severity, model=model,
                                    iterations=args.iterations, lr=args.lr)

    import matplotlib.pyplot as plt

    for idx, data_dict in enumerate(demo_dataset):
        logger.info(f'Sample index: {idx + 1}')

        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        # ============ 攻击前推理 ============
        logger.info('Inference BEFORE attack...')
        pred_clean = run_inference(model, data_dict)
        labels_clean = pred_clean['pred_labels']
        scores_clean = pred_clean['pred_scores']
        stats_before = {
            'num': len(labels_clean),
            'cls_counts': {CLS_NAMES[c]: int((labels_clean == c).sum().item()) for c in CLS_NAMES},
            'scores': scores_clean.cpu().numpy(),
            'score_mean': float(scores_clean.mean().item()) if len(labels_clean) > 0 else 0.0,
            'score_median': float(scores_clean.median().item()) if len(labels_clean) > 0 else 0.0,
            'score_min': float(scores_clean.min().item()) if len(labels_clean) > 0 else 0.0,
        }

        # ============ C&W Perturbation 攻击 ============
        logger.info('Running C&W Perturbation attack...')
        data_dict = attacker.forward(data_dict)

        # ============ 攻击后推理 ============
        logger.info('Inference AFTER attack...')
        with torch.no_grad():
            pred_dicts, _ = model.forward(data_dict)
        pred_adv = pred_dicts[0]
        labels_adv = pred_adv['pred_labels']
        scores_adv = pred_adv['pred_scores']
        stats_after = {
            'num': len(labels_adv),
            'cls_counts': {CLS_NAMES[c]: int((labels_adv == c).sum().item()) for c in CLS_NAMES},
            'scores': scores_adv.cpu().numpy(),
            'score_mean': float(scores_adv.mean().item()) if len(labels_adv) > 0 else 0.0,
            'score_median': float(scores_adv.median().item()) if len(labels_adv) > 0 else 0.0,
            'score_min': float(scores_adv.min().item()) if len(labels_adv) > 0 else 0.0,
        }

        print_comparison(stats_before, stats_after)

        # ============ 并排 BEV 对比图 ============
        pts = data_dict['points'][:, 1:].cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(22, 10))
        fig.patch.set_facecolor('black')
        draw_bev(axes[0], pts, pred_clean['pred_boxes'].cpu().numpy(),
                 labels_clean.cpu().numpy(), f'ORIGINAL  ({stats_before["num"]} objects)')
        draw_bev(axes[1], pts, pred_adv['pred_boxes'].cpu().numpy(),
                 labels_adv.cpu().numpy(), f'C&W PERTURBATION  ({stats_after["num"]} objects)')
        plt.tight_layout(pad=2)
        save_path = 'compare_perturb_bev.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        logger.info(f'Comparison BEV saved to: {os.path.abspath(save_path)}')

    logger.info('C&W Perturbation attack test done.')


if __name__ == '__main__':
    main()
