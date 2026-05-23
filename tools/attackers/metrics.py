"""
对抗扰动度量工具 (论文 Section 3.4, metrics.py)

实现三维点云对抗攻击的量化评估体系，包含三类度量:
  1. 扰动幅度 (L2, Chamfer, Hausdorff, outlier ratio)
  2. 攻击有效性 (ASR, 置信度下降, 各类别检出变化)
  3. 感知质量 (分布偏移, 异常点比例)

可用于 Perturbation 攻击 (有原始/扰动对应点对) 和 Generation 攻击 (新注入点)。

参考: Xiang, Qi, Li - "Generating 3D Adversarial Point Clouds" (CVPR 2019, Sec 3.4)
"""

import torch
import numpy as np


class PerturbationMetrics:
    """
    对抗攻击效果量化评估器。

    使用方法:
        metrics = PerturbationMetrics.compute_all(
            original_xyz=orig, perturbed_xyz=adv,
            clean_preds=pred_clean, adv_preds=pred_adv,
            valid_mask=mask
        )
        print(metrics)
    """

    @staticmethod
    def l2_per_point(original_xyz, perturbed_xyz, valid_mask=None):
        """
        逐点 L2 距离 (论文 Eq.4)。
        Args:
            original_xyz: (N, 3) 或 (M, max_pts, 3)
            perturbed_xyz: 同上
            valid_mask: (N,) 或 (M, max_pts), bool 或 float
        Returns:
            per_point_dist: (N,) 或 (M, max_pts) 每个有效点的位移
        """
        dist = torch.norm(perturbed_xyz - original_xyz, p=2, dim=-1)
        if valid_mask is not None:
            dist = dist * valid_mask
        return dist

    @staticmethod
    def l2_summary(original_xyz, perturbed_xyz, valid_mask=None):
        """L2 扰动统计摘要"""
        dist = PerturbationMetrics.l2_per_point(original_xyz, perturbed_xyz, valid_mask)
        if valid_mask is not None:
            mask_bool = valid_mask.bool() if valid_mask.dtype != torch.bool else valid_mask
            d = dist[mask_bool]
        else:
            d = dist.flatten()

        if d.numel() == 0:
            return dict(mean=0, median=0, max=0, std=0, p90=0, p95=0, p99=0)

        d_sorted = d.sort().values
        n = d_sorted.numel()
        return dict(
            mean=d.mean().item(),
            median=d_sorted[n // 2].item(),
            max=d_sorted[-1].item(),
            std=d.std().item(),
            p90=d_sorted[int(0.90 * n)].item(),
            p95=d_sorted[int(0.95 * n)].item(),
            p99=d_sorted[int(0.99 * n)].item(),
        )

    @staticmethod
    def outlier_ratio(original_xyz, perturbed_xyz, threshold=0.3, valid_mask=None):
        """
        大幅扰动点比例 (论文 Sec 3.4, outlier ratio)。
        统计位移超过 threshold (m) 的点占比。
        """
        dist = PerturbationMetrics.l2_per_point(original_xyz, perturbed_xyz, valid_mask)
        if valid_mask is not None:
            mask_bool = valid_mask.bool() if valid_mask.dtype != torch.bool else valid_mask
            d = dist[mask_bool]
        else:
            d = dist.flatten()

        if d.numel() == 0:
            return 0.0
        return (d > threshold).float().mean().item()

    @staticmethod
    def chamfer_distance(pc1, pc2, n_samples=5000):
        """
        Chamfer distance (论文 Sec 3.4 Eq.5)。
        CD(A,B) = mean_a min_b ||a-b||^2 + mean_b min_a ||b-a||^2

        对 10万+ 点云自动下采样以控制计算量。
        """
        if pc1.shape[0] > n_samples:
            idx = torch.randperm(pc1.shape[0], device=pc1.device)[:n_samples]
            pc1 = pc1[idx]
        if pc2.shape[0] > n_samples:
            idx = torch.randperm(pc2.shape[0], device=pc2.device)[:n_samples]
            pc2 = pc2[idx]

        # pairwise L2^2: (n1, n2)
        diff = pc1.unsqueeze(1) - pc2.unsqueeze(0)  # (n1, n2, 3)
        dist2 = (diff ** 2).sum(dim=-1)               # (n1, n2)

        d1 = dist2.min(dim=1).values.mean()  # mean_a min_b
        d2 = dist2.min(dim=0).values.mean()  # mean_b min_a
        return (d1 + d2).item()

    @staticmethod
    def hausdorff_distance(pc1, pc2, n_samples=5000):
        """
        Hausdorff distance (论文 Sec 3.4 Eq.6)。
        H(A,B) = max{ max_a min_b ||a-b||, max_b min_a ||b-a|| }
        """
        if pc1.shape[0] > n_samples:
            idx = torch.randperm(pc1.shape[0], device=pc1.device)[:n_samples]
            pc1 = pc1[idx]
        if pc2.shape[0] > n_samples:
            idx = torch.randperm(pc2.shape[0], device=pc2.device)[:n_samples]
            pc2 = pc2[idx]

        diff = pc1.unsqueeze(1) - pc2.unsqueeze(0)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)

        d_ab = dist.min(dim=1).values.max()
        d_ba = dist.min(dim=0).values.max()
        return max(d_ab.item(), d_ba.item())

    # ──── 攻击有效性度量 ────

    @staticmethod
    def attack_success_rate(clean_preds, adv_preds, mode='count'):
        """
        攻击成功率 (论文 Sec 3.4, ASR)。

        Args:
            clean_preds: dict with 'pred_scores', 'pred_labels', 'pred_boxes'
            adv_preds: 同上
            mode: 'count' — 基于检出框数量
                  'score' — 基于平均置信度
        Returns:
            ASR ∈ [0, 100] (%)
        """
        scores_clean = clean_preds.get('pred_scores', torch.tensor([]))
        scores_adv = adv_preds.get('pred_scores', torch.tensor([]))

        if mode == 'count':
            n_clean = len(scores_clean)
            n_adv = len(scores_adv)
            if n_clean == 0:
                return 0.0
            return (n_clean - n_adv) / n_clean * 100

        elif mode == 'score':
            if scores_clean.numel() == 0 or scores_clean.mean() == 0:
                return 0.0
            drop = (scores_clean.mean() - scores_adv.mean()) / scores_clean.mean()
            return drop.item() * 100

        else:
            raise ValueError(f"Unknown ASR mode: {mode}")

    @staticmethod
    def class_breakdown(clean_preds, adv_preds, class_names=None):
        """
        逐类检出变化。
        Returns: dict of {cls_name: {'before': N, 'after': M, 'delta': D}}
        """
        if class_names is None:
            class_names = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}

        labels_clean = clean_preds.get('pred_labels', torch.tensor([]))
        labels_adv = adv_preds.get('pred_labels', torch.tensor([]))

        breakdown = {}
        for cls_id, cls_name in class_names.items():
            n_before = int((labels_clean == cls_id).sum().item())
            n_after = int((labels_adv == cls_id).sum().item())
            breakdown[cls_name] = dict(before=n_before, after=n_after,
                                       delta=n_before - n_after)

        breakdown['Total'] = dict(
            before=len(labels_clean), after=len(labels_adv),
            delta=len(labels_clean) - len(labels_adv)
        )
        return breakdown

    @staticmethod
    def confidence_stats(clean_preds, adv_preds):
        """置信度统计对比"""
        sc = clean_preds.get('pred_scores', torch.tensor([]))
        sa = adv_preds.get('pred_scores', torch.tensor([]))

        def _stats(s):
            if s.numel() == 0:
                return dict(mean=0, median=0, min=0, max=0, std=0)
            return dict(
                mean=s.mean().item(), median=s.median().item(),
                min=s.min().item(), max=s.max().item(), std=s.std().item()
            )

        return dict(clean=_stats(sc), adversarial=_stats(sa))

    # ──── 综合接口 ────

    @staticmethod
    def compute_perturbation(original_xyz, perturbed_xyz, valid_mask=None):
        """针对 Perturbation 攻击的全套度量 (论文 Sec 3.2 评估)"""
        return dict(
            l2=PerturbationMetrics.l2_summary(original_xyz, perturbed_xyz, valid_mask),
            outlier_ratio_30cm=PerturbationMetrics.outlier_ratio(
                original_xyz, perturbed_xyz, threshold=0.3, valid_mask=valid_mask),
            chamfer=PerturbationMetrics.chamfer_distance(
                original_xyz.reshape(-1, 3), perturbed_xyz.reshape(-1, 3)),
            hausdorff=PerturbationMetrics.hausdorff_distance(
                original_xyz.reshape(-1, 3), perturbed_xyz.reshape(-1, 3)),
        )

    @staticmethod
    def compute_effectiveness(clean_preds, adv_preds, class_names=None):
        """针对任意攻击的有效性度量 (论文 Sec 3.4 评估)"""
        return dict(
            asr_count=PerturbationMetrics.attack_success_rate(clean_preds, adv_preds, 'count'),
            asr_score=PerturbationMetrics.attack_success_rate(clean_preds, adv_preds, 'score'),
            class_breakdown=PerturbationMetrics.class_breakdown(clean_preds, adv_preds, class_names),
            confidence=PerturbationMetrics.confidence_stats(clean_preds, adv_preds),
        )

    @staticmethod
    def compute_all(original_xyz=None, perturbed_xyz=None, valid_mask=None,
                    clean_preds=None, adv_preds=None, class_names=None):
        """一键计算所有度量"""
        result = {}
        if original_xyz is not None and perturbed_xyz is not None:
            result['perturbation'] = PerturbationMetrics.compute_perturbation(
                original_xyz, perturbed_xyz, valid_mask)
        if clean_preds is not None and adv_preds is not None:
            result['effectiveness'] = PerturbationMetrics.compute_effectiveness(
                clean_preds, adv_preds, class_names)
        return result

    @staticmethod
    def pretty_print(metrics):
        """格式化打印度量报告 (类似论文 Table 1)"""
        print(f"\n{'='*64}")
        print(f"  Perturbation Measurement Report (Sec 3.4)")
        print(f"{'='*64}")

        if 'perturbation' in metrics:
            p = metrics['perturbation']
            l2 = p['l2']
            print(f"\n  ── Perturbation Magnitude ──")
            print(f"  L2 mean/median/max:  {l2['mean']:.4f} / {l2['median']:.4f} / {l2['max']:.4f} m")
            print(f"  L2 std/p90/p95/p99:  {l2['std']:.4f} / {l2['p90']:.4f} / {l2['p95']:.4f} / {l2['p99']:.4f} m")
            print(f"  Outlier ratio (>0.3m): {p['outlier_ratio_30cm']:.2%}")
            print(f"  Chamfer distance:     {p['chamfer']:.4f}")
            print(f"  Hausdorff distance:   {p['hausdorff']:.4f}")

        if 'effectiveness' in metrics:
            e = metrics['effectiveness']
            print(f"\n  ── Attack Effectiveness ──")
            print(f"  ASR (count):  {e['asr_count']:.1f}%")
            print(f"  ASR (score):  {e['asr_score']:.1f}%")
            cb = e['class_breakdown']
            print(f"  {'Class':<14} {'Before':>8} {'After':>8} {'Delta':>8}")
            print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}")
            for cls_name, counts in cb.items():
                sign = '+' if counts['delta'] < 0 else ''
                print(f"  {cls_name:<14} {counts['before']:>8} {counts['after']:>8} "
                      f"{sign}{-counts['delta']:>7}")
            conf = e['confidence']
            print(f"\n  ── Confidence Statistics ──")
            print(f"  {'':<10} {'Clean':>12} {'Adversarial':>12}")
            for stat in ['mean', 'median', 'min', 'max']:
                print(f"  {stat:<10} {conf['clean'][stat]:>12.4f} {conf['adversarial'][stat]:>12.4f}")

        print(f"{'='*64}\n")
