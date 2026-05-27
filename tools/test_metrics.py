"""
对抗攻击度量综合测试脚本。

对指定点云执行 C&W Perturbation 攻击，然后计算全套度量指标。
支持与 PGD 攻击的并排对比。

用法:
    cd /root/OpenPCDet/tools
    python test_metrics.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt ../ckpts/pointpillar_7728.pth \
        --data_path ../data/kitti/testing/velodyne/000008.bin \
        --severity 0.5 --iterations 50
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import sys

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

CLS_NAMES = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}


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
    parser = argparse.ArgumentParser(description='Attack Metrics Test')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml')
    parser.add_argument('--data_path', type=str, default='../data/kitti/testing/velodyne/000008.bin')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ext', type=str, default='.bin')
    parser.add_argument('--severity', type=float, default=0.5)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--attack', type=str, default='perturb',
                        help='Attack type: perturb, pgd')
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


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info(f'--- Attack Metrics Test: {args.attack} severity={args.severity} ---')

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    sys.path.append('..')
    from attackers.metrics import PerturbationMetrics

    if args.attack == 'perturb':
        from attackers.perturbation import PerturbationAttacker
        attacker = PerturbationAttacker(severity=args.severity, model=model,
                                        iterations=args.iterations, lr=args.lr)
    elif args.attack == 'pgd':
        from attackers.pgd import PGDAttacker
        attacker = PGDAttacker(severity=args.severity, model=model, iterations=args.iterations)
    else:
        raise ValueError(f"Unknown attack: {args.attack}")

    for idx, data_dict in enumerate(demo_dataset):
        logger.info(f'Sample {idx + 1}/{len(demo_dataset)}')

        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        # 记录攻击前原始坐标 (来自 voxels)
        original_xyz = data_dict['voxels'][:, :, :3].clone()
        num_points = data_dict['voxel_num_points']
        max_pts = original_xyz.shape[1]
        point_idx = torch.arange(max_pts, device=original_xyz.device).unsqueeze(0)
        valid_mask = (point_idx < num_points.unsqueeze(1)).float()

        # 攻击前推理
        logger.info('Clean inference...')
        pred_clean = run_inference(model, data_dict)

        # 攻击
        logger.info(f'Running {args.attack} attack...')
        data_dict = attacker.forward(data_dict)

        # 攻击后坐标
        perturbed_xyz = data_dict['voxels'][:, :, :3]

        # 攻击后推理
        logger.info('Adversarial inference...')
        with torch.no_grad():
            pred_dicts, _ = model.forward(data_dict)
        pred_adv = pred_dicts[0]

        # 计算全套度量
        metrics = PerturbationMetrics.compute_all(
            original_xyz=original_xyz, perturbed_xyz=perturbed_xyz, valid_mask=valid_mask,
            clean_preds=pred_clean, adv_preds=pred_adv, class_names=CLS_NAMES
        )
        PerturbationMetrics.pretty_print(metrics)

        # 逐类 ASR
        print(f"  ── Per-Class ASR ──")
        for cls_id, cls_name in CLS_NAMES.items():
            n_c = int((pred_clean['pred_labels'] == cls_id).sum().item())
            n_a = int((pred_adv['pred_labels'] == cls_id).sum().item())
            asr_cls = (n_c - n_a) / max(n_c, 1) * 100
            print(f"  {cls_name:<14} {n_c:>3} → {n_a:>3}  (ASR {asr_cls:>5.1f}%)")

    logger.info('Metrics test done.')


if __name__ == '__main__':
    main()
