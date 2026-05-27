"""
对抗攻击综合实验管线 (论文 Section 4, experiment.py)

批量运行多种攻击，多 severity 消融实验，自动收集度量并生成对比报告。

支持的攻击类型:
  黑盒 (速度快, 适合全量): noise, drop
  白盒 (需梯度, 扰动):    pgd, perturb
  插入 (需梯度, 加点):    spawn, scatter, object

用法:
    # 快速消融: 所有攻击 × 3 个 severity × 单帧
    python experiment.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt ../ckpts/pointpillar_7728.pth \
        --data_path ../data/kitti/testing/velodyne/ \
        --severities 0.2,0.5,0.8 --max_frames 3 --skip_whitebox

    # 完整对比 (含白盒):
    python experiment.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt ../ckpts/pointpillar_7728.pth \
        --data_path ../data/kitti/testing/velodyne/ \
        --severities 0.5 --max_frames 5
"""
import argparse
import glob
import os
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

CLS_NAMES = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
BLACKBOX_ATTACKS = ['noise', 'drop']
WHITEBOX_ATTACKS = ['pgd', 'perturb']
INSERTION_ATTACKS = ['spawn', 'scatter', 'object']  # gradient-based point insertion
ALL_ATTACKS = BLACKBOX_ATTACKS + WHITEBOX_ATTACKS + INSERTION_ATTACKS


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
    parser = argparse.ArgumentParser(description='Adversarial Attack Experiment Pipeline')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml')
    parser.add_argument('--data_path', type=str, default='../data/kitti/testing/velodyne/')
    parser.add_argument('--ckpt', type=str, default='../ckpts/pointpillar_7728.pth')
    parser.add_argument('--ext', type=str, default='.bin')
    parser.add_argument('--severities', type=str, default='0.2,0.5,0.8',
                        help='Comma-separated severity values')
    parser.add_argument('--max_frames', type=int, default=5,
                        help='Max frames to evaluate (limit for speed)')
    parser.add_argument('--skip_whitebox', action='store_true',
                        help='Skip gradient-based attacks (faster)')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Iterations for white-box attacks')
    parser.add_argument('--viz_frames', type=int, default=1,
                        help='Number of frames to generate per-attack BEV/Plotly/PLY visualizations')
    args = parser.parse_args()
    args.severities = [float(s) for s in args.severities.split(',')]
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def clone_data_dict(data_dict):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}


def run_clean_inference(model, data_dict):
    d = clone_data_dict(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(d)
    return pred_dicts[0]


def compute_pred_stats(pred, cls_names):
    labels = pred['pred_labels']
    scores = pred['pred_scores']
    box_count = len(labels)
    cls_counts = {cls_names[c]: int((labels == c).sum().item()) for c in cls_names}
    score_stats = dict(
        mean=float(scores.mean().item()) if box_count > 0 else 0.0,
        median=float(scores.median().item()) if box_count > 0 else 0.0,
        max=float(scores.max().item()) if box_count > 0 else 0.0,
    )
    return dict(count=box_count, cls_counts=cls_counts, score_stats=score_stats)


def run_blackbox_attack(attacker, raw_points, demo_dataset):
    """黑盒攻击: 在原始点云上攻击, 然后走 prepare_data → collate → gpu 完整管线.
    Returns: (data_dict_gpu, attacked_points_numpy)
    """
    pts_tensor = torch.from_numpy(raw_points).cuda()
    tmp = {'points': pts_tensor}
    tmp = attacker.forward(tmp)
    attacked_points = tmp['points'].cpu().numpy()

    input_dict = {'points': attacked_points, 'frame_id': 0}
    data_dict = demo_dataset.prepare_data(data_dict=input_dict)
    data_dict_gpu = demo_dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict_gpu)
    return data_dict_gpu, attacked_points


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info(f'{"="*64}')
    logger.info(f'  Adversarial Attack Experiment Pipeline (Sec 4)')
    logger.info(f'{"="*64}')
    logger.info(f'Attacks: {ALL_ATTACKS if not args.skip_whitebox else BLACKBOX_ATTACKS}')
    logger.info(f'Severities: {args.severities}')
    logger.info(f'Max frames: {args.max_frames}')
    logger.info(f'Data: {args.data_path}')

    sys.path.append('..')
    from attackers import get_attacker
    from attackers.metrics import PerturbationMetrics

    # ======== 加载数据 ========
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    total_frames = min(len(demo_dataset), args.max_frames)
    logger.info(f'Total frames available: {len(demo_dataset)}, using: {total_frames}')

    # ======== 加载模型 ========
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # ======== 预加载所有帧的原始点云 (numpy) ========
    raw_frame_points = []
    raw_frame_ids = []
    for frame_idx in range(total_frames):
        sample_path = demo_dataset.sample_file_list[frame_idx]
        raw_pts = np.fromfile(sample_path, dtype=np.float32).reshape(-1, 4)
        raw_frame_points.append(raw_pts)
        raw_frame_ids.append(Path(sample_path).stem)

    # ======== 实验主循环 ========
    all_results = defaultdict(lambda: defaultdict(list))
    clean_stats_list = []

    attack_types = BLACKBOX_ATTACKS
    if not args.skip_whitebox:
        attack_types += WHITEBOX_ATTACKS + INSERTION_ATTACKS

    for frame_idx in range(total_frames):
        raw_pts = raw_frame_points[frame_idx]
        frame_id = raw_frame_ids[frame_idx]
        logger.info(f'\n{"─"*50}')
        logger.info(f'Frame {frame_idx + 1}/{total_frames}: {frame_id}')

        # clean inference: 原始点云走完整管线
        input_dict = {'points': raw_pts.copy(), 'frame_id': 0}
        clean_dict = demo_dataset.prepare_data(data_dict=input_dict)
        data_dict_gpu = demo_dataset.collate_batch([clean_dict])
        load_data_to_gpu(data_dict_gpu)
        pred_clean = run_clean_inference(model, data_dict_gpu)
        clean_stats = compute_pred_stats(pred_clean, CLS_NAMES)
        clean_stats_list.append(clean_stats)

        # ── 单帧可视化：干净帧 BEV + PLY ──
        do_viz = frame_idx < args.viz_frames
        if do_viz:
            viz_frame_dir = Path('result') / 'frames' / frame_id
            viz_frame_dir.mkdir(parents=True, exist_ok=True)

            clean_boxes = pred_clean['pred_boxes'].cpu().numpy()
            clean_labels = pred_clean['pred_labels'].cpu().numpy()

            BEVVisualizer.draw(
                raw_pts, clean_boxes,
                save_path=str(viz_frame_dir / 'clean_bev.png'),
                title=f'{frame_id} — Clean ({clean_stats["count"]} detections)',
                labels=clean_labels)
            PLYExporter.export(raw_pts, str(viz_frame_dir / 'clean_cloud.ply'))
            logger.info(f'  [viz] Clean BEV + PLY saved to {viz_frame_dir}')

        for atk_type in attack_types:
            for sev in args.severities:
                t0 = time.time()
                attacked_pts_np = None  # adversarial raw points (only for black-box)

                if atk_type in WHITEBOX_ATTACKS:
                    # 白盒攻击: 在 collated GPU tensors 上操作
                    attacker = get_attacker(atk_type, severity=sev, model=model,
                                            iterations=args.iterations)
                    d = clone_data_dict(data_dict_gpu)
                    d = attacker.forward(d)
                    with torch.no_grad():
                        preds, _ = model.forward(d)
                    pred_adv = preds[0]

                    # 度量 (白盒有原始/扰动对应点)
                    original_xyz = data_dict_gpu['voxels'][:, :, :3]
                    perturbed_xyz = d['voxels'][:, :, :3]
                    num_pts = data_dict_gpu['voxel_num_points']
                    max_pts_val = original_xyz.shape[1]
                    pt_idx = torch.arange(max_pts_val, device=original_xyz.device).unsqueeze(0)
                    vmask = (pt_idx < num_pts.unsqueeze(1)).float()
                    pert_metrics = PerturbationMetrics.l2_summary(original_xyz, perturbed_xyz, vmask)

                elif atk_type in INSERTION_ATTACKS:
                    # 插入型白盒攻击: 在 GPU data_dict 上添加对抗体素（论文 Section 5.1-5.3）
                    attacker = get_attacker(atk_type, severity=sev, model=model,
                                            iterations=args.iterations)
                    d = clone_data_dict(data_dict_gpu)
                    d = attacker.forward(d)
                    with torch.no_grad():
                        preds, _ = model.forward(d)
                    pred_adv = preds[0]

                    # 度量: 新增的体素数量及它们与原始点云的 Chamfer 距离
                    n_orig = data_dict_gpu['voxel_num_points'].sum().item()
                    n_adv = d['voxel_num_points'].sum().item()
                    n_added = n_adv - n_orig
                    pert_metrics = dict(
                        mean=n_added,  # number of points added
                        max=d['voxels'].shape[0] - data_dict_gpu['voxels'].shape[0],  # voxels added
                    )

                else:
                    # 黑盒攻击: 在原始点云上攻击, 走完整 prepare_data 管线
                    attacker = get_attacker(atk_type, severity=sev)
                    d_gpu, attacked_pts_np = run_blackbox_attack(attacker, raw_pts.copy(), demo_dataset)
                    with torch.no_grad():
                        preds, _ = model.forward(d_gpu)
                    pred_adv = preds[0]
                    pert_metrics = dict(mean=0.0, max=0.0)

                elapsed = time.time() - t0

                adv_stats = compute_pred_stats(pred_adv, CLS_NAMES)
                asr = (clean_stats['count'] - adv_stats['count']) / max(clean_stats['count'], 1) * 100

                # ── 单帧可视化：攻击后 BEV + Plotly + PLY (仅黑盒有完整点云) ──
                if do_viz:
                    atk_viz_dir = viz_frame_dir / atk_type / f'sev_{sev:.2f}'
                    atk_viz_dir.mkdir(parents=True, exist_ok=True)

                    adv_boxes = pred_adv['pred_boxes'].cpu().numpy()
                    adv_labels = pred_adv['pred_labels'].cpu().numpy()
                    pts_for_bev = attacked_pts_np if attacked_pts_np is not None else raw_pts

                    # BEV (攻击后)
                    BEVVisualizer.draw(
                        pts_for_bev, adv_boxes,
                        save_path=str(atk_viz_dir / 'adv_bev.png'),
                        title=f'{frame_id} [{atk_type}] sev={sev:.2f} — ASR={asr:.1f}%',
                        labels=adv_labels)

                    # PLY (攻击后点云)
                    PLYExporter.export(pts_for_bev, str(atk_viz_dir / 'adv_cloud.ply'))

                    # Plotly 3D 交互热力图 (仅黑盒有干净/攻击点云对)
                    # 注意: 插入类攻击(生成新点)会使 adv > clean, 需对齐尺寸
                    if attacked_pts_np is not None:
                        n_common = min(len(raw_pts), len(attacked_pts_np))
                        plot_sample = min(n_common, 30000)
                        idx_c = np.random.choice(len(raw_pts), min(plot_sample, len(raw_pts)), replace=False)
                        idx_a = np.random.choice(len(attacked_pts_np), min(plot_sample, len(attacked_pts_np)), replace=False)
                        PlotlyVisualizer.export_attack_analysis(
                            pts_clean=raw_pts[idx_c], pts_adv=attacked_pts_np[idx_a],
                            boxes=adv_boxes, labels=adv_labels,
                            save_path=str(atk_viz_dir / '3d_heatmap.html'),
                            title=f'{frame_id} [{atk_type}] sev={sev:.2f} (ASR={asr:.1f}%)')

                result = dict(
                    frame=frame_id, severity=sev, asr=asr,
                    clean_count=clean_stats['count'], adv_count=adv_stats['count'],
                    clean_score=clean_stats['score_stats']['mean'],
                    adv_score=adv_stats['score_stats']['mean'],
                    pert_mean=pert_metrics.get('mean', 0),
                    pert_max=pert_metrics.get('max', 0),
                    cls_clean=clean_stats['cls_counts'],
                    cls_adv=adv_stats['cls_counts'],
                )
                all_results[atk_type][sev].append(result)

                if atk_type in INSERTION_ATTACKS:
                    logger.info(f'  [{atk_type}] sev={sev:.2f}: '
                                f'{clean_stats["count"]}→{adv_stats["count"]} boxes '
                                f'(ASR={asr:.1f}%) '
                                f'+{pert_metrics.get("mean", 0):.0f}pts '
                                f'({elapsed:.1f}s)')
                else:
                    logger.info(f'  [{atk_type}] sev={sev:.2f}: '
                                f'{clean_stats["count"]}→{adv_stats["count"]} boxes '
                                f'(ASR={asr:.1f}%) '
                                f'pert={pert_metrics.get("mean", 0):.4f}m '
                                f'({elapsed:.1f}s)')

    # ======== 汇总报告 ========
    print_report(all_results, clean_stats_list, args.severities, attack_types, logger)
    save_visualizations(all_results, clean_stats_list, args.severities, attack_types)


def print_report(all_results, clean_stats_list, severities, attack_types, logger):
    """生成论文风格的汇总表 (Table 1/2 格式)"""
    n_frames = len(clean_stats_list)
    avg_clean = np.mean([s['count'] for s in clean_stats_list])
    avg_clean_score = np.mean([s['score_stats']['mean'] for s in clean_stats_list])

    print(f'\n{"="*80}')
    print(f'  EXPERIMENT SUMMARY REPORT')
    print(f'  Frames: {n_frames}  |  Avg clean detections: {avg_clean:.1f}  '
          f'|  Avg clean score: {avg_clean_score:.4f}')
    print(f'{"="*80}')

    # ─── Table 1: ASR by attack type × severity ───
    print(f'\n  Table 1 — Attack Success Rate (ASR %) by Method × Severity')
    print(f'  {"Attack":<12}', end='')
    for sev in severities:
        print(f'{"sev=" + str(sev):>12}', end='')
    print(f'  {"Avg":>8}')
    print(f'  {"─"*12}', end='')
    print(f'{"─"*12}' * len(severities), end='')
    print(f'{"─"*8}')

    for atk_type in attack_types:
        print(f'  {atk_type:<12}', end='')
        asr_list = []
        for sev in severities:
            results = all_results[atk_type][sev]
            if results:
                avg_asr = np.mean([r['asr'] for r in results])
                asr_list.append(avg_asr)
                print(f'{avg_asr:>11.1f}%', end='')
            else:
                print(f'{"N/A":>12}', end='')
        if asr_list:
            print(f'  {np.mean(asr_list):>7.1f}%', end='')
        print()

    # ─── Table 2: Perturbation Magnitude (white-box + insertion) ───
    wb_types = [a for a in attack_types if a in WHITEBOX_ATTACKS]
    if wb_types:
        print(f'\n  Table 2 — Perturbation Magnitude (mean L2, meters)')
        print(f'  {"Attack":<12}', end='')
        for sev in severities:
            print(f'{"sev=" + str(sev):>12}', end='')
        print()
        print(f'  {"─"*12}' + f'{"─"*12}' * len(severities))

        for atk_type in wb_types:
            print(f'  {atk_type:<12}', end='')
            for sev in severities:
                results = all_results[atk_type][sev]
                if results:
                    avg_pert = np.mean([r['pert_mean'] for r in results])
                    print(f'{avg_pert:>11.4f}m', end='')
                else:
                    print(f'{"N/A":>12}', end='')
            print()

    # ─── Table 2b: Insertion Attack Overhead ───
    ins_types = [a for a in attack_types if a in INSERTION_ATTACKS]
    if ins_types:
        print(f'\n  Table 2b — Insertion Attack Overhead (avg points / voxels added)')
        print(f'  {"Attack":<12} {"pts_added":>12} {"vox_added":>12} {"pts/frame":>12}')
        print(f'  {"─"*12} {"─"*12} {"─"*12} {"─"*12}')

        for atk_type in ins_types:
            print(f'  {atk_type:<12}', end='')
            all_pts = []
            all_vox = []
            for sev in severities:
                for r in all_results[atk_type][sev]:
                    all_pts.append(r['pert_mean'])  # points added
                    all_vox.append(r['pert_max'])   # voxels added
            if all_pts:
                print(f'{np.mean(all_pts):>11.0f}', end='')
                print(f'{np.mean(all_vox):>11.0f}', end='')
                print(f'{np.mean(all_pts):>11.0f}', end='')
            print()

    # ─── Table 3: Per-Class ASR (averaged across severities) ───
    print(f'\n  Table 3 — Per-Class Detection Change (avg across severities)')
    print(f'  {"Attack":<12} {"Car":>12} {"Pedestrian":>12} {"Cyclist":>12}')
    print(f'  {"─"*12} {"─"*12} {"─"*12} {"─"*12}')

    for atk_type in attack_types:
        print(f'  {atk_type:<12}', end='')
        for cls_name in ['Car', 'Pedestrian', 'Cyclist']:
            deltas = []
            for sev in severities:
                for r in all_results[atk_type][sev]:
                    c_clean = r['cls_clean'].get(cls_name, 0)
                    c_adv = r['cls_adv'].get(cls_name, 0)
                    if c_clean > 0:
                        deltas.append((c_clean - c_adv) / c_clean * 100)
            if deltas:
                avg_d = np.mean(deltas)
                print(f'{avg_d:>11.1f}%', end='')
            else:
                print(f'{"N/A":>12}', end='')
        print()

    # ─── Summary ───
    ins_types = [a for a in attack_types if a in INSERTION_ATTACKS]
    print(f'\n  ── Key Takeaways ──')
    # Best attack (highest ASR)
    best_asr = -999
    best_pair = ('', 0.0)
    for atk_type in attack_types:
        for sev in severities:
            results = all_results[atk_type][sev]
            if results:
                avg = np.mean([r['asr'] for r in results])
                if avg > best_asr:
                    best_asr = avg
                    best_pair = (atk_type, sev)
    print(f'  Best attack: {best_pair[0]} @ severity={best_pair[1]:.1f} (ASR={best_asr:.1f}%)')

    # Most stealthy white-box (best ASR per perturbation)
    if wb_types:
        best_ratio = -999
        best_stealth = ('', 0.0)
        for atk_type in wb_types:
            for sev in severities:
                results = all_results[atk_type][sev]
                if results:
                    asr = np.mean([r['asr'] for r in results])
                    pert = max(np.mean([r['pert_mean'] for r in results]), 1e-6)
                    ratio = asr / pert
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_stealth = (atk_type, sev)
        print(f'  Best ASR/perturbation: {best_stealth[0]} @ severity={best_stealth[1]:.1f} '
              f'(ASR/m = {best_ratio:.1f})')

    # Most efficient insertion (best ASR per point added)
    if ins_types:
        best_eff = -999
        best_ins = ('', 0.0)
        for atk_type in ins_types:
            for sev in severities:
                results = all_results[atk_type][sev]
                if results:
                    asr = np.mean([r['asr'] for r in results])
                    pts = max(np.mean([r['pert_mean'] for r in results]), 1)
                    eff = asr / pts * 100  # ASR per 100 points
                    if eff > best_eff:
                        best_eff = eff
                        best_ins = (atk_type, sev)
        print(f'  Best ASR/point: {best_ins[0]} @ severity={best_ins[1]:.1f} '
              f'(ASR/100pts = {best_eff:.1f})')

    print(f'{"="*80}\n')

    # Save CSV
    result_dir = Path('result')
    result_dir.mkdir(exist_ok=True)
    csv_path = result_dir / 'experiment_results.csv'
    with open(csv_path, 'w') as f:
        f.write('attack,severity,frame,clean_count,adv_count,asr,clean_score,adv_score,pert_mean,pert_max\n')
        for atk_type in attack_types:
            for sev in severities:
                for r in all_results[atk_type][sev]:
                    f.write(f'{atk_type},{sev},{r["frame"]},'
                            f'{r["clean_count"]},{r["adv_count"]},{r["asr"]:.2f},'
                            f'{r["clean_score"]:.4f},{r["adv_score"]:.4f},'
                            f'{r["pert_mean"]:.4f},{r["pert_max"]:.4f}\n')
    logger.info(f'Results saved to: {os.path.abspath(csv_path)}')


def save_visualizations(all_results, clean_stats_list, severities, attack_types):
    """生成论文风格的可视化图表并保存到 result/ 目录。"""
    result_dir = Path('result')
    result_dir.mkdir(exist_ok=True)
    n_frames = len(clean_stats_list)
    avg_clean = np.mean([s['count'] for s in clean_stats_list])

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 200,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    })

    # ---- Build aggregate data matrix: [n_attacks, n_severities] ----
    atk_labels = list(attack_types)
    sev_labels = [f'{s:.1f}' for s in severities]
    asr_matrix = np.zeros((len(atk_labels), len(severities)))
    asr_std_matrix = np.zeros((len(atk_labels), len(severities)))
    pert_matrix = np.zeros((len(atk_labels), len(severities)))

    for i, atk in enumerate(atk_labels):
        for j, sev in enumerate(severities):
            results = all_results[atk][sev]
            if results:
                asr_vals = [r['asr'] for r in results]
                pert_vals = [r['pert_mean'] for r in results]
                asr_matrix[i, j] = np.mean(asr_vals)
                asr_std_matrix[i, j] = np.std(asr_vals)
                pert_matrix[i, j] = np.mean(pert_vals)
            else:
                asr_matrix[i, j] = np.nan
                asr_std_matrix[i, j] = 0
                pert_matrix[i, j] = 0

    # ---- Color palette ----
    cmap_sev = plt.cm.viridis  # severity colormap
    colors_sev = [cmap_sev(0.2 + 0.6 * i / max(len(severities) - 1, 1)) for i in range(len(severities))]
    colors_atk = plt.cm.tab10(np.linspace(0, 0.9, len(atk_labels)))

    # ================================================================
    # Figure 1: ASR Grouped Bar Chart
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(max(8, 1.8 * len(atk_labels)), 5.5))
    x = np.arange(len(atk_labels))
    width = 0.75 / len(severities)

    for j in range(len(severities)):
        bars = ax1.bar(x + j * width, asr_matrix[:, j], width,
                       color=colors_sev[j], edgecolor='white', linewidth=0.5,
                       label=f'severity={sev_labels[j]}')
        for bar, val in zip(bars, asr_matrix[:, j]):
            if not np.isnan(val):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + (2 if val >= 0 else -8),
                         f'{val:.0f}%', ha='center', va='bottom' if val >= 0 else 'top',
                         fontsize=7, fontweight='bold')

    ax1.axhline(y=0, color='gray', linewidth=0.8, linestyle='-')
    ax1.set_xticks(x + width * (len(severities) - 1) / 2)
    ax1.set_xticklabels(atk_labels, fontweight='bold')
    ax1.set_ylabel('Attack Success Rate (%)')
    ax1.set_title(f'Fig 1: ASR by Attack Method × Severity\n'
                  f'({n_frames} frames, avg clean detections={avg_clean:.1f})')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    fig1.tight_layout()
    fig1.savefig(result_dir / 'fig1_asr_bars.png')
    plt.close(fig1)

    # ================================================================
    # Figure 2: Severity Ablation Curves (line plot)
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for i, atk in enumerate(atk_labels):
        ax2.plot(severities, asr_matrix[i, :], 'o-', color=colors_atk[i],
                 linewidth=2, markersize=8, label=atk, markeredgecolor='white',
                 markeredgewidth=0.5)

    ax2.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
    ax2.set_xlabel('Severity')
    ax2.set_ylabel('Attack Success Rate (%)')
    ax2.set_title(f'Fig 2: ASR vs Severity Ablation\n({n_frames} frames)')
    ax2.legend(framealpha=0.9, ncol=2 if len(atk_labels) > 5 else 1)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xlim(min(severities) - 0.05, max(severities) + 0.05)
    fig2.tight_layout()
    fig2.savefig(result_dir / 'fig2_severity_curves.png')
    plt.close(fig2)

    # ================================================================
    # Figure 3: Per-Class ASR Heatmap
    # ================================================================
    cls_names = ['Car', 'Pedestrian', 'Cyclist']
    cls_matrix = np.zeros((len(atk_labels), len(cls_names)))

    for i, atk in enumerate(atk_labels):
        for k, cls_name in enumerate(cls_names):
            deltas = []
            for sev in severities:
                for r in all_results[atk][sev]:
                    c_clean = r['cls_clean'].get(cls_name, 0)
                    c_adv = r['cls_adv'].get(cls_name, 0)
                    if c_clean > 0:
                        deltas.append((c_clean - c_adv) / c_clean * 100)
            cls_matrix[i, k] = np.mean(deltas) if deltas else 0

    fig3, ax3 = plt.subplots(figsize=(7, max(3.5, 0.5 * len(atk_labels))))
    vmax = max(abs(cls_matrix).max(), 1)
    im = ax3.imshow(cls_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    cbar = fig3.colorbar(im, ax=ax3, shrink=0.85)
    cbar.set_label('ASR % (+ = suppression, − = FP generation)')

    for i in range(len(atk_labels)):
        for k in range(len(cls_names)):
            ax3.text(k, i, f'{cls_matrix[i, k]:.0f}%', ha='center', va='center',
                     fontweight='bold', fontsize=10,
                     color='white' if abs(cls_matrix[i, k]) > vmax * 0.5 else 'black')

    ax3.set_xticks(range(len(cls_names)))
    ax3.set_xticklabels(cls_names, fontweight='bold')
    ax3.set_yticks(range(len(atk_labels)))
    ax3.set_yticklabels(atk_labels, fontweight='bold')
    ax3.set_title(f'Fig 3: Per-Class Detection Change\n({n_frames} frames, avg across severities)')
    fig3.tight_layout()
    fig3.savefig(result_dir / 'fig3_perclass_heatmap.png')
    plt.close(fig3)

    # ================================================================
    # Figure 4: Perturbation vs ASR Trade-off (white-box + insertion)
    # ================================================================
    wb_types = [a for a in attack_types if a in WHITEBOX_ATTACKS]
    ins_types = [a for a in attack_types if a in INSERTION_ATTACKS]
    if wb_types or ins_types:
        fig4, ax4 = plt.subplots(figsize=(7, 5.5))

        # White-box: L2 perturbation
        for i, atk in enumerate(wb_types):
            sev_vals = []
            asr_vals = []
            pert_vals = []
            for j, sev in enumerate(severities):
                results = all_results[atk][sev]
                if results:
                    sev_vals.append(sev)
                    asr_vals.append(np.mean([r['asr'] for r in results]))
                    pert_vals.append(np.mean([r['pert_mean'] for r in results]))
            if sev_vals:
                ax4.plot(pert_vals, asr_vals, 'o-', color=colors_atk[atk_labels.index(atk)],
                         linewidth=2, markersize=10, label=f'{atk} (L2)', markeredgecolor='white')
                for s, a, p in zip(sev_vals, asr_vals, pert_vals):
                    ax4.annotate(f'sev={s:.1f}', (p, a), textcoords='offset points',
                                 xytext=(8, 4), fontsize=8, alpha=0.85,
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # Insertion: points added
        for i, atk in enumerate(ins_types):
            sev_vals = []
            asr_vals = []
            pert_vals = []
            for j, sev in enumerate(severities):
                results = all_results[atk][sev]
                if results:
                    sev_vals.append(sev)
                    asr_vals.append(np.mean([r['asr'] for r in results]))
                    pert_vals.append(np.mean([r['pert_mean'] for r in results]))
            if sev_vals:
                ax4.plot(pert_vals, asr_vals, 's--', color=colors_atk[atk_labels.index(atk)],
                         linewidth=2, markersize=10, label=f'{atk} (pts)', markeredgecolor='white')
                for s, a, p in zip(sev_vals, asr_vals, pert_vals):
                    ax4.annotate(f'sev={s:.1f}', (p, a), textcoords='offset points',
                                 xytext=(8, 4), fontsize=8, alpha=0.85,
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        ax4.set_xlabel('Attack Magnitude (L2 meters / points added)')
        ax4.set_ylabel('Attack Success Rate (%)')
        ax4.set_title(f'Fig 4: ASR vs Attack Magnitude Trade-off\n({n_frames} frames)')
        ax4.legend(framealpha=0.9)
        ax4.grid(alpha=0.3, linestyle='--')
        fig4.tight_layout()
        fig4.savefig(result_dir / 'fig4_perturbation_tradeoff.png')
        plt.close(fig4)

    # ================================================================
    # Figure 5: Summary Dashboard (2×2 or 2×3 layout)
    # ================================================================
    fig5 = plt.figure(figsize=(14, 10))
    fig5.suptitle(f'Adversarial Attack on PointPillar — Experiment Summary\n'
                  f'({n_frames} frames, avg clean detections={avg_clean:.1f})',
                  fontsize=15, fontweight='bold', y=0.98)

    # Subplot 1: ASR bars (top-left)
    ax5a = fig5.add_subplot(2, 2, 1)
    x = np.arange(len(atk_labels))
    width = 0.75 / len(severities)
    for j in range(len(severities)):
        ax5a.bar(x + j * width, asr_matrix[:, j], width,
                 color=colors_sev[j], edgecolor='white', linewidth=0.5,
                 label=f's={sev_labels[j]}')
    ax5a.axhline(y=0, color='gray', linewidth=0.8)
    ax5a.set_xticks(x + width * (len(severities) - 1) / 2)
    ax5a.set_xticklabels(atk_labels, fontsize=9)
    ax5a.set_ylabel('ASR (%)')
    ax5a.set_title('ASR by Method × Severity', fontweight='bold')
    ax5a.legend(fontsize=7, loc='upper right')
    ax5a.grid(axis='y', alpha=0.3, linestyle='--')

    # Subplot 2: Severity curves (top-right)
    ax5b = fig5.add_subplot(2, 2, 2)
    for i, atk in enumerate(atk_labels):
        ax5b.plot(severities, asr_matrix[i, :], 'o-', color=colors_atk[i],
                  linewidth=2, markersize=7, label=atk, markeredgecolor='white')
    ax5b.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
    ax5b.set_xlabel('Severity')
    ax5b.set_ylabel('ASR (%)')
    ax5b.set_title('Severity Ablation Curves', fontweight='bold')
    ax5b.legend(fontsize=7, ncol=2 if len(atk_labels) > 5 else 1)
    ax5b.grid(alpha=0.3, linestyle='--')

    # Subplot 3: Per-class heatmap (bottom-left)
    ax5c = fig5.add_subplot(2, 2, 3)
    vmax_c = max(abs(cls_matrix).max(), 1)
    im2 = ax5c.imshow(cls_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-vmax_c, vmax=vmax_c)
    for i in range(len(atk_labels)):
        for k in range(len(cls_names)):
            ax5c.text(k, i, f'{cls_matrix[i, k]:.0f}%', ha='center', va='center',
                      fontweight='bold', fontsize=9,
                      color='white' if abs(cls_matrix[i, k]) > vmax_c * 0.5 else 'black')
    ax5c.set_xticks(range(len(cls_names)))
    ax5c.set_xticklabels(cls_names, fontsize=10)
    ax5c.set_yticks(range(len(atk_labels)))
    ax5c.set_yticklabels(atk_labels, fontsize=10)
    ax5c.set_title('Per-Class Detection Change (%)', fontweight='bold')
    fig5.colorbar(im2, ax=ax5c, shrink=0.85)

    # Subplot 4: Box count comparison (bottom-right) — all attacks
    ax5d = fig5.add_subplot(2, 2, 4)
    all_atk_colors = plt.cm.Set2(np.linspace(0, 1, len(atk_labels)))
    avg_adv_counts = []
    for i, atk in enumerate(atk_labels):
        all_adv = []
        for sev in severities:
            for r in all_results[atk][sev]:
                all_adv.append(r['adv_count'])
        avg_adv_counts.append(np.mean(all_adv) if all_adv else 0)
    bars2 = ax5d.barh(atk_labels, avg_adv_counts, color=all_atk_colors, edgecolor='white')
    ax5d.axvline(x=avg_clean, color='red', linewidth=2, linestyle='--',
                 label=f'Clean avg ({avg_clean:.1f})')
    for bar, val in zip(bars2, avg_adv_counts):
        ax5d.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                  f'{val:.1f}', va='center', fontsize=9, fontweight='bold')
    ax5d.set_xlabel('Avg Detections per Frame')
    ax5d.set_title('Detection Count After Attack (all attacks)', fontweight='bold')
    ax5d.legend(fontsize=8)
    ax5d.grid(axis='x', alpha=0.3, linestyle='--')

    fig5.tight_layout(rect=[0, 0, 1, 0.94])
    fig5.savefig(result_dir / 'fig5_dashboard.png')
    plt.close(fig5)

    print(f'\n  Visualization results saved to: {result_dir.resolve()}/')
    for fname in sorted(result_dir.glob('fig*.png')):
        print(f'    {fname.name}')


if __name__ == '__main__':
    main()
