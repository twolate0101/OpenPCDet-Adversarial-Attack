#!/usr/bin/env python3
"""
Generate paper-quality figures from full mAP evaluation results.
KITTI 3D AP_R40 (3769 validation samples, PointPillars)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# Data: Full mAP results (AP_R40 3D, Easy/Moderate/Hard)
# ============================================================
# Format: {attack: {severity: {'Car': (E,M,H), 'Ped': (E,M,H), 'Cyc': (E,M,H)}}}
DATA = {
    'baseline': {0.0: {'Car': (87.75,78.40,75.19), 'Ped': (57.20,51.43,46.87), 'Cyc': (81.56,62.92,59.03)}},
    'noise': {
        0.1: {'Car': (83.45,69.85,66.63), 'Ped': (20.02,19.56,19.81), 'Cyc': (61.51,44.02,41.24)},
        0.3: {'Car': (8.09,6.90,6.18), 'Ped': (0.004,0.006,0.008), 'Cyc': (0.08,0.07,0.07)},
        0.5: {'Car': (0.04,0.00,0.00), 'Ped': (0.00,0.001,0.001), 'Cyc': (0.001,0.001,0.001)},
    },
    'drop': {
        0.1: {'Car': (87.06,76.36,73.26), 'Ped': (55.76,50.31,46.77), 'Cyc': (81.86,62.87,59.93)},
        0.3: {'Car': (86.94,75.62,72.09), 'Ped': (54.31,49.39,45.38), 'Cyc': (76.52,57.91,54.71)},
        0.5: {'Car': (83.97,70.04,66.42), 'Ped': (43.80,39.89,36.40), 'Cyc': (61.67,44.60,41.57)},
    },
    'geo_drop': {
        0.1: {'Car': (69.21,68.17,65.86), 'Ped': (46.51,42.54,39.07), 'Cyc': (60.92,50.42,47.90)},
        0.3: {'Car': (36.69,47.71,45.42), 'Ped': (25.78,24.63,22.72), 'Cyc': (32.99,34.13,32.46)},
        0.5: {'Car': (20.94,35.56,34.36), 'Ped': (12.37,12.40,11.73), 'Cyc': (19.73,25.67,24.69)},
    },
}

# Also include white-box results (Moderate only) for the summary
WHITEBOX_MOD = {
    'pgd':           {0.3: {'Car': 0.14, 'Ped': 0.09, 'Cyc': 0.12}},
    'perturb':       {0.5: {'Car': 0.60, 'Ped': 0.78, 'Cyc': 1.39}},
    'saliency_mask': {0.3: {'Car': 37.06, 'Ped': 0.004, 'Cyc': 0.015}},
    'spawn':         {0.5: {'Car': 78.32, 'Ped': 50.55, 'Cyc': 62.12}},
    'scatter':       {0.5: {'Car': 78.02, 'Ped': 49.58, 'Cyc': 61.53}},
    'object':        {0.5: {'Car': 78.31, 'Ped': 51.09, 'Cyc': 62.44}},
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def fig6_map_grouped_bar():
    """Fig 6: Grouped bar chart — Moderate AP by attack × severity."""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    attacks = ['noise', 'drop', 'geo_drop']
    severities = [0.1, 0.3, 0.5]
    classes = ['Car', 'Ped', 'Cyc']
    class_labels = ['Car (IoU=0.70)', 'Pedestrian (IoU=0.50)', 'Cyclist (IoU=0.50)']
    baseline_mod = {'Car': 78.40, 'Ped': 51.43, 'Cyc': 62.92}

    colors = {'Car': '#2196F3', 'Ped': '#FF9800', 'Cyc': '#4CAF50'}
    sev_hatches = {0.1: '', 0.3: '//', 0.5: 'xx'}

    x = np.arange(len(attacks) * len(severities))
    width = 0.25
    positions = []

    for si, sev in enumerate(severities):
        offsets = []
        for ai, atk in enumerate(attacks):
            base_idx = ai * len(severities) + si
            offsets.append(base_idx)
        positions.append(offsets)

    # Rearrange: group by attack, sub-group by severity
    x_groups = np.arange(len(attacks))
    sub_width = 0.08
    bar_gap = 0.02

    for ci, cls in enumerate(classes):
        for si, sev in enumerate(severities):
            bars_x = []
            bars_h = []
            for ai, atk in enumerate(attacks):
                val = DATA[atk][sev][cls][1]  # Moderate
                bars_x.append(ai + (si - 1) * (sub_width + bar_gap) + ci * (sub_width * 3 + bar_gap * 3 + 0.04) - 0.35)
                bars_h.append(val)

    # Simpler approach: one group per (attack, severity), bars for each class
    labels = []
    for atk in attacks:
        for sev in severities:
            labels.append(f'{atk}\ns={sev}')

    n_groups = len(labels)
    x = np.arange(n_groups)
    width = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(13, 5.5))

    for ci, (cls, clabel) in enumerate(zip(classes, class_labels)):
        vals = []
        for atk in attacks:
            for sev in severities:
                vals.append(DATA[atk][sev][cls][1])
        bars = ax.bar(x + offsets[ci], vals, width, label=clabel, color=colors[cls],
                      edgecolor='white', linewidth=0.5, alpha=0.9)
        # Add value labels on bars
        for bar in bars:
            h = bar.get_height()
            if h > 3:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}',
                        ha='center', va='bottom', fontsize=6.5, rotation=90)

    # Baseline lines
    for ci, (cls, clabel) in enumerate(zip(classes, class_labels)):
        ax.axhline(y=baseline_mod[cls], color=colors[cls], linestyle='--', alpha=0.5, linewidth=1)
        ax.text(n_groups - 0.5, baseline_mod[cls] + 1, f'Baseline {cls}: {baseline_mod[cls]:.1f}',
                fontsize=7, color=colors[cls], alpha=0.7, ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('3D AP (AP_R40, Moderate)')
    ax.set_title('Fig 6: Full Evaluation mAP — Black-box Attacks (KITTI val, 3769 samples)')
    ax.set_ylim(0, 95)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig6_map_grouped_bar.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def fig7_severity_curves():
    """Fig 7: Line plot — AP vs severity for noise/drop/geo_drop."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    attacks = ['noise', 'drop', 'geo_drop']
    severities = [0.0, 0.1, 0.3, 0.5]
    classes = ['Car', 'Ped', 'Cyc']
    class_labels = ['Car (IoU=0.70)', 'Pedestrian (IoU=0.50)', 'Cyclist (IoU=0.50)']
    colors = {'noise': '#E53935', 'drop': '#1E88E5', 'geo_drop': '#43A047'}
    markers = {'noise': 'o', 'drop': 's', 'geo_drop': '^'}
    baseline_mod = {'Car': 78.40, 'Ped': 51.43, 'Cyc': 62.92}

    for ci, (cls, clabel) in enumerate(zip(classes, class_labels)):
        ax = axes[ci]
        for atk in attacks:
            vals = [baseline_mod[cls]]
            for sev in [0.1, 0.3, 0.5]:
                vals.append(DATA[atk][sev][cls][1])
            ax.plot(severities, vals, '-', color=colors[atk], marker=markers[atk],
                    markersize=7, linewidth=2, label=atk, alpha=0.9)
            # Annotate points
            for s, v in zip(severities, vals):
                if v > 2:
                    ax.annotate(f'{v:.1f}', (s, v), textcoords="offset points",
                                xytext=(0, 8), ha='center', fontsize=7)

        ax.set_xlabel('Severity')
        ax.set_title(clabel)
        ax.set_xlim(-0.05, 0.55)
        ax.set_ylim(-2, 95)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    axes[0].set_ylabel('3D AP (AP_R40, Moderate)')
    axes[0].legend(loc='upper right', fontsize=9)

    fig.suptitle('Fig 7: Severity Gradient — Black-box Attack Impact on mAP (KITTI val, 3769 samples)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig7_severity_curves.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def fig8_heatmap():
    """Fig 8: Heatmap — AP drop rate across attack × class."""
    attacks_order = ['noise', 'drop', 'geo_drop']
    severities = [0.1, 0.3, 0.5]
    classes = ['Car', 'Ped', 'Cyc']
    class_labels = ['Car\n(IoU=0.70)', 'Pedestrian\n(IoU=0.50)', 'Cyclist\n(IoU=0.50)']
    baseline = {'Car': 78.40, 'Ped': 51.43, 'Cyc': 62.92}

    # Build matrix: rows = attack×severity, cols = class
    row_labels = []
    matrix = []
    for atk in attacks_order:
        for sev in severities:
            row_labels.append(f'{atk} s={sev}')
            row = []
            for cls in classes:
                mod_ap = DATA[atk][sev][cls][1]
                drop_pct = (1 - mod_ap / baseline[cls]) * 100
                row.append(drop_pct)
            matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(class_labels)):
            val = matrix[i, j]
            color = 'white' if val > 60 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('AP Drop Rate (%)', fontsize=11)

    ax.set_title('Fig 8: AP Drop Rate Heatmap — Attack × Severity × Class\n(Moderate, relative to baseline)',
                 fontsize=12)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig8_map_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def fig9_difficulty_comparison():
    """Fig 9: Easy/Mod/Hard comparison grouped bars."""
    attacks = ['noise', 'drop', 'geo_drop']
    severities = [0.1, 0.3, 0.5]
    difficulties = ['Easy', 'Moderate', 'Hard']
    diff_colors = {'Easy': '#66BB6A', 'Moderate': '#FFA726', 'Hard': '#EF5350'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ai, atk in enumerate(attacks):
        ax = axes[ai]
        x = np.arange(len(severities))
        width = 0.22

        for di, diff in enumerate(difficulties):
            # Average across 3 classes for this difficulty
            vals = []
            for sev in severities:
                avg = np.mean([DATA[atk][sev][cls][di] for cls in ['Car', 'Ped', 'Cyc']])
                vals.append(avg)
            bars = ax.bar(x + (di - 1) * width, vals, width, label=diff,
                         color=diff_colors[diff], edgecolor='white', linewidth=0.5, alpha=0.9)
            for bar in bars:
                h = bar.get_height()
                if h > 3:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                            f'{h:.1f}', ha='center', va='bottom', fontsize=7, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([f's={s}' for s in severities])
        ax.set_xlabel('Severity')
        ax.set_title(atk.replace('_', ' ').title())
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_ylim(0, 95)

        if ai == 0:
            ax.set_ylabel('3D AP (AP_R40, avg over Car/Ped/Cyc)')
            ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Fig 9: Difficulty Level Comparison — Easy / Moderate / Hard (KITTI val, 3769 samples)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig9_difficulty_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def fig10_summary_all_attacks():
    """Fig 10: Summary bar chart — ALL attacks Moderate AP comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))

    baseline = {'Car': 78.40, 'Ped': 51.43, 'Cyc': 62.92}

    # All attack entries (best severity for each)
    entries = [
        ('Baseline', 78.40, 51.43, 62.92),
        ('noise s=0.1', 69.85, 19.56, 44.02),
        ('noise s=0.3', 6.90, 0.006, 0.07),
        ('noise s=0.5', 0.00, 0.001, 0.001),
        ('drop s=0.1', 76.36, 50.31, 62.87),
        ('drop s=0.3', 75.62, 49.39, 57.91),
        ('drop s=0.5', 70.04, 39.89, 44.60),
        ('geo_drop s=0.1', 68.17, 42.54, 50.42),
        ('geo_drop s=0.3', 47.71, 24.63, 34.13),
        ('geo_drop s=0.5', 35.56, 12.40, 25.67),
        ('pgd s=0.3', 0.14, 0.09, 0.12),
        ('perturb s=0.5', 0.60, 0.78, 1.39),
        ('saliency s=0.3', 37.06, 0.004, 0.015),
        ('spawn s=0.5', 78.32, 50.55, 62.12),
        ('scatter s=0.5', 78.02, 49.58, 61.53),
        ('object s=0.5', 78.31, 51.09, 62.44),
    ]

    labels = [e[0] for e in entries]
    car_vals = [e[1] for e in entries]
    ped_vals = [e[2] for e in entries]
    cyc_vals = [e[3] for e in entries]

    n = len(entries)
    x = np.arange(n)
    width = 0.25

    ax.bar(x - width, car_vals, width, label='Car (IoU=0.70)', color='#2196F3', alpha=0.9)
    ax.bar(x, ped_vals, width, label='Pedestrian (IoU=0.50)', color='#FF9800', alpha=0.9)
    ax.bar(x + width, cyc_vals, width, label='Cyclist (IoU=0.50)', color='#4CAF50', alpha=0.9)

    # Baseline lines
    for val, color, lbl in [(78.40, '#2196F3', 'Car'), (51.43, '#FF9800', 'Ped'), (62.92, '#4CAF50', 'Cyc')]:
        ax.axhline(y=val, color=color, linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('3D AP (AP_R40, Moderate)')
    ax.set_title('Fig 10: All Attacks Summary — Moderate AP Comparison (KITTI val, 3769 samples)')
    ax.set_ylim(0, 95)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig10_all_attacks_summary.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


if __name__ == '__main__':
    fig6_map_grouped_bar()
    fig7_severity_curves()
    fig8_heatmap()
    fig9_difficulty_comparison()
    fig10_summary_all_attacks()
    print('\nAll figures generated successfully!')
