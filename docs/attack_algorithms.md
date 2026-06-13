# 对抗攻击算法手册

> 基于 CVPR 2019 "Generating 3D Adversarial Point Clouds" (Xiang, Qi, Li) 框架，
> 针对 PointPillars (KITTI) 检测器实现的 10 种对抗攻击算法。

---

## 一、分类总览

| 类型 | 攻击方式 | 操作对象 | 需要梯度 | 算法 |
|------|---------|---------|---------|------|
| 黑盒扰动 | 点坐标加高斯噪声 | points (N,4) | 否 | `noise` |
| 黑盒删除 | 随机/几何删除点云 | points (N,4) | 否 | `drop` `geo_drop` |
| 白盒扰动 | 迭代微调 voxel 坐标 | voxels (M,P,4) | 是 | `pgd` `perturb` |
| 白盒删除 | 梯度指导移除关键 voxels | voxels + coords | 是 | `saliency_mask` |
| 白盒插入 | 梯度优化生成对抗点/物体 | voxels | 是 | `spawn` `scatter` `object` |
| 黑盒插入 | 真实模板复制粘贴 | points (N,4) | 否 | `ghost_template` |

**注入架构（双通道）**：

```
黑盒:  dataset.__getitem__()  →  _attacker.forward(points)  →  体素化  →  模型推理
白盒:  体素化完成  →  eval_utils 调用 attacker.forward(data_dict)  →  模型推理
```

- 黑盒攻击在 `pcdet/datasets/dataset.py` 的体素化前拦截（修改 `points`）
- 白盒攻击在 `tools/eval_utils/eval_utils.py` 的模型推理前注入（修改 `voxels`）

---

## 二、通用参数

所有攻击器继承 `BaseAttacker`，统一使用 `severity ∈ [0.0, 1.0]` 控制强度。

| 参数 | 类型 | 说明 |
|------|------|------|
| `severity` | float | 攻击强度，0.0=无攻击，1.0=最强。各算法的具体映射不同 |
| `model` | nn.Module | 模型实例，白盒攻击必需，黑盒攻击不需要 |
| `iterations` | int | 迭代次数，仅白盒攻击使用（默认值因算法而异）|
| `lr` | float | 学习率，仅 Adam 优化的白盒攻击使用 |

---

## 三、黑盒删除攻击

### 3.1 noise — 高斯噪声

**文件**: `attackers/__init__.py` (NoiseAttacker 类)

**原理**: 给所有点的 XYZ 坐标加上独立高斯噪声，模拟 LiDAR 传感器受干扰或物理微振。不修改反射率强度值。

**severity 映射**: `severity` = 噪声标准差（米）。`severity=0.2` 表示约 20cm 随机抖动。

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| severity | 1.0 | 高斯噪声标准差 σ（米）|

**测试命令**:
```bash
# 单帧 demo
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth \
  --data_path ../data/kitti/testing/velodyne/000008.bin \
  --attack noise --severity 0.5

# 全量 mAP
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth --batch_size 4 \
  --attack noise --severity 0.5
```

**已有结果**:
| 类别 | severity | Car 3D AP (Mod) | Ped 3D AP (Mod) | Cyc 3D AP (Mod) |
|------|----------|----------------|----------------|----------------|
| baseline | — | 78.40 | 51.43 | 62.92 |
| noise | 0.1 | 69.85 | 19.56 | 44.02 |
| noise | 0.3 | 6.90 | 0.006 | 0.07 |
| noise | 0.5 | 0.04 | 0.00 | 0.00 |

**详细结果** (AP_R40, Easy/Moderate/Hard):
| severity | Car (E/M/H) | Ped (E/M/H) | Cyc (E/M/H) |
|----------|-------------|-------------|-------------|
| 0.1 | 83.45/69.85/66.63 | 20.02/19.56/19.81 | 61.51/44.02/41.24 |
| 0.3 | 8.09/6.90/6.18 | 0.004/0.006/0.008 | 0.08/0.07/0.07 |
| 0.5 | 0.04/0.00/0.00 | 0.00/0.001/0.001 | 0.001/0.001/0.001 |

---

### 3.2 drop — 随机点云删除

**文件**: `attackers/drop.py`

**原理**: 按 severity 比例随机丢弃点云中的点，模拟 LiDAR 传感器遮挡/信号丢失。保留至少 1% 或 100 个点确保体素化不崩溃。

**severity 映射**: `severity` = 删除比例。`severity=0.5` 表示随机丢弃 50% 的点。

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| severity | 1.0 | 删除比例 (0.0~1.0)|

**测试命令**:
```bash
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth \
  --data_path ../data/kitti/testing/velodyne/000008.bin \
  --attack drop --severity 0.5

python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth --batch_size 4 \
  --attack drop --severity 0.5
```

**已有结果**:
| 类别 | severity | Car 3D AP (Mod) | Ped 3D AP (Mod) | Cyc 3D AP (Mod) |
|------|----------|----------------|----------------|----------------|
| baseline | — | 78.40 | 51.43 | 62.92 |
| drop | 0.1 | 76.36 | 50.31 | 62.87 |
| drop | 0.3 | 75.62 | 49.39 | 57.91 |
| drop | 0.5 | 70.04 | 39.89 | 44.60 |

**详细结果** (AP_R40, Easy/Moderate/Hard):
| severity | Car (E/M/H) | Ped (E/M/H) | Cyc (E/M/H) |
|----------|-------------|-------------|-------------|
| 0.1 | 87.06/76.36/73.26 | 55.76/50.31/46.77 | 81.86/62.87/59.93 |
| 0.3 | 86.94/75.62/72.09 | 54.31/49.39/45.38 | 76.52/57.91/54.71 |
| 0.5 | 83.97/70.04/66.42 | 43.80/39.89/36.40 | 61.67/44.60/41.57 |

---

### 3.3 geo_drop — 几何感知删除

**文件**: `attackers/geo_drop.py`

**原理**: 计算点云质心，优先删除距质心近的点（目标表面高密度区域），比随机删除更有效地破坏目标几何特征。

**severity 映射**: `severity` = 删除比例（距质心最近的 N% 的点被删除）。

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| severity | 1.0 | 删除比例 (0.0~1.0)，最多删 90% |

**测试命令**:
```bash
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth \
  --data_path ../data/kitti/testing/velodyne/000008.bin \
  --attack geo_drop --severity 0.5

python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth --batch_size 4 \
  --attack geo_drop --severity 0.5
```

**已有结果**:
| 类别 | severity | Car 3D AP (Mod) | Ped 3D AP (Mod) | Cyc 3D AP (Mod) |
|------|----------|----------------|----------------|----------------|
| baseline | — | 78.40 | 51.43 | 62.92 |
| geo_drop | 0.1 | 68.17 | 42.54 | 50.42 |
| geo_drop | 0.3 | 47.71 | 24.63 | 34.13 |
| geo_drop | 0.5 | 35.56 | 12.40 | 25.67 |

**详细结果** (AP_R40, Easy/Moderate/Hard):
| severity | Car (E/M/H) | Ped (E/M/H) | Cyc (E/M/H) |
|----------|-------------|-------------|-------------|
| 0.1 | 69.21/68.17/65.86 | 46.51/42.54/39.07 | 60.92/50.42/47.90 |
| 0.3 | 36.69/47.71/45.42 | 25.78/24.63/22.72 | 32.99/34.13/32.46 |
| 0.5 | 20.94/35.56/34.36 | 12.37/12.40/11.73 | 19.73/25.67/24.69 |

---

## 四、白盒扰动攻击

### 4.1 pgd — 投影梯度下降

**文件**: `attackers/pgd.py`

**原理**: 对 voxels 的 xyz 坐标执行多轮 sign-梯度上升，每轮扰动后投影回 L2 epsilon 球内。目标：最大化检测分数和的负值 → 检测器失效。

**对应论文**: Section 4 (Adversarial Point Perturbation) — Lp 范数约束下的迭代扰动。

**与 perturb 的区别**: PGD 用 sign 梯度 + 硬投影（epsilon 球约束），perturb 用完整梯度 + Adam + 软正则。

**severity 映射**: `severity` = L2 epsilon 扰动预算。每步步长 `alpha = severity / iterations`。

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| severity | 1.0 | L2 扰动预算 ε |
| model | None | 模型实例（必需）|
| iterations | 10 | 迭代次数 |
| alpha | auto | 每步步长 = severity / iterations |

**测试命令**:
```bash
# 单帧 demo
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth \
  --data_path ../data/kitti/testing/velodyne/000008.bin \
  --attack pgd --severity 0.3 --iterations 10

# 全量 mAP
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth --batch_size 4 \
  --attack pgd --severity 0.3 --iterations 10
```

**已有结果** (severity=0.3):
| 类别 | Car 3D AP (Mod) | Ped 3D AP (Mod) | Cyc 3D AP (Mod) |
|------|----------------|----------------|----------------|
| pgd 0.3 | 0.14 | 0.09 | 0.12 |

---

### 4.2 perturb — C&W 风格体素扰动

**文件**: `attackers/perturbation.py`

**原理**: 使用 Adam 优化器迭代调整 voxels 的 xyz 坐标，优化目标为 `min_δ  -sum(scores) + λ·mean(||δ||₂²)`。λ 与 severity 反比：severity 越大 → λ 越小 → 允许更大扰动。

**对应论文**: Section 3 Eq.2 (C&W 优化目标) + Section 4 (点扰动框架)。

**与 pgd 的区别**: 使用完整梯度（非 sign），Adam 优化器，软正则 λ（非硬投影）→ 扰动更精细。

**severity 映射**: `severity` 控制 λ 的倒数。`severity=0.1` → λ≈10（隐蔽），`severity=1.0` → λ≈1（强力）。

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| severity | 1.0 | 控制 λ（扰动正则的反比权重）|
| model | None | 模型实例（必需）|
| iterations | 50 | Adam 迭代次数 |
| lr | 0.01 | Adam 学习率 |
| lambda_reg | auto | `max(0.5, 1/(severity+1e-6))` |

**测试命令**:
```bash
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth \
  --data_path ../data/kitti/testing/velodyne/000008.bin \
  --attack perturb --severity 0.5 --iterations 20

python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth --batch_size 4 \
  --attack perturb --severity 0.5 --iterations 20
```

**已有结果** (severity=0.5):
| 类别 | Car 3D AP (Mod) | Ped 3D AP (Mod) | Cyc 3D AP (Mod) |
|------|----------------|----------------|----------------|
| perturb 0.5 | 0.60 | 0.78 | 1.39 |

---

## 五、白盒删除攻击

### 5.1 saliency_mask — 显著性体素掩码

**文件**: `attackers/saliency_mask.py`

**原理**: 借鉴 JSMA 梯度显著性思想，通过一次前向+反向传播获取 voxels 梯度，按梯度绝对值排序，将对检测贡献最大的 Top-K% voxels **真正移除**（同步过滤 voxels/voxel_coords/voxel_num_points）。

**为什么不置零 xyz**: PillarVFE 计算 `f_center = xyz - voxel_center`，xyz 置零后 f_center 产生很大的虚假偏移，效果更像扰动而非删除。真正移除后，对应 BEV 位置保持为零。

**severity 映射**: `severity` = 移除比例。`severity=0.3` 表示移除梯度最大的 30% voxels。

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| severity | 0.1 | 移除 voxels 的比例 (0.0~1.0) |
| model | None | 模型实例（必需）|

**边界保护**:
- `severity <= 0`: 直接返回，不做修改
- `num_voxels <= 1`: 直接返回，无法安全移除
- `k = min(max(1, ...), num_voxels - 1)`: 至少保留 1 个 voxel

**测试命令**:
```bash
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth \
  --data_path ../data/kitti/testing/velodyne/000008.bin \
  --attack saliency_mask --severity 0.3

python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth --batch_size 4 \
  --attack saliency_mask --severity 0.3
```

**已有结果**:
| severity | Car 3D AP (Mod) | Ped 3D AP (Mod) | Cyc 3D AP (Mod) | Recall@0.3 | Recall@0.5 | Recall@0.7 |
|----------|----------------|----------------|----------------|-----------|-----------|-----------|
| 0.3 | 37.06 | 0.004 | 0.015 | 1.16 | 0.59 | 0.20 |
| 0.5 | 11.89 | 0.00 | 0.00 | 12.72 | 10.84 | 7.71 |

---

## 六、白盒插入攻击

三种白盒插入攻击共享 `CriticalPointFinder`（梯度分析 + DBSCAN 聚类）定位脆弱区域。

### CriticalPointFinder（共用模块）

**文件**: `attackers/critical_points.py`

**流程**:
1. 前向+反向传播，计算 voxels xyz 梯度
2. 逐点梯度幅值 → 逐 pillar 关键性得分（取 max）
3. Top-K 最关键的 pillar
4. 提取每个关键 pillar 中梯度最大的点的 xyz 坐标
5. DBSCAN 在 XY（BEV）平面聚类 → 脆弱区域中心

---

### 6.1 spawn — 对抗点簇生成

**文件**: `attackers/spawn.py`

**原理**: 在 CriticalPointFinder 找到的脆弱区域中心附近生成点云簇（每簇 32 个表面点），通过 Adam 优化器迭代调整簇的 XYZ 位置，最小化检测置信度。每个簇对应一个独立的 pillar 体素。

**对应论文**: Section 5.2 (Generating Adversarial Point Clouds)。

**severity 映射**: `severity` 控制点簇数量。`severity=0.5` → ~5 簇，`severity=1.0` → ~10 簇。

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| severity | 1.0 | 控制簇数量（max(1, severity×10)）|
| model | None | 模型实例（必需）|
| iterations | 30 | Adam 迭代次数 |
| lr | 0.05 | Adam 学习率 |
| num_clusters | auto | max(1, severity×10) |
| pts_per_cluster | 32 | 每簇表面点数 |
| cluster_radius | auto | 0.8 × severity |

**测试命令**:
```bash
# 单帧 demo
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth \
  --data_path ../data/kitti/testing/velodyne/000008.bin \
  --attack spawn --severity 0.5 --iterations 20

# 全量 mAP
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth --batch_size 4 \
  --attack spawn --severity 0.5 --iterations 20
```

**已有结果** (severity=0.5):
| 类别 | Car 3D AP (Mod) | Ped 3D AP (Mod) | Cyc 3D AP (Mod) | Recall@0.3 | Recall@0.5 | Recall@0.7 |
|------|----------------|----------------|----------------|-----------|-----------|-----------|
| baseline | 78.40 | 51.43 | 62.92 | 94.11 | 88.71 | 63.90 |
| spawn 0.5 | 78.32 | 50.55 | 62.12 | 93.72 | 88.22 | 63.70 |

---

### 6.2 scatter — 对抗散点放置

**文件**: `attackers/scatter.py`

**原理**: 在脆弱区域附近以高斯散布生成独立散点（每个散点 = 1 个独立 pillar 体素），通过 sign-梯度迭代优化散点 XYZ 坐标。与 spawn 的区别：spawn 是簇（32 点/簇），scatter 是独立散点（1 点/散点）。

**对应论文**: Section 5.1 (Generating Adversarial Independent Points)。

**severity 映射**: `severity` 控制散点数量。`severity=0.3` → ~300 散点，`severity=1.0` → ~1000 散点。

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| severity | 1.0 | 控制散点数量（max(100, severity×1000)）|
| model | None | 模型实例（必需）|
| iterations | 30 | sign-梯度迭代次数 |
| lr | 0.05 | 学习率 |
| num_points | auto | max(100, severity×1000) |
| scatter_sigma | auto | severity × 2.0 |

**测试命令**:
```bash
# 单帧 demo
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth \
  --data_path ../data/kitti/testing/velodyne/000008.bin \
  --attack scatter --severity 0.5 --iterations 20

# 全量 mAP
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth --batch_size 4 \
  --attack scatter --severity 0.5 --iterations 20
```

**已有结果** (severity=0.5):
| 类别 | Car 3D AP (Mod) | Ped 3D AP (Mod) | Cyc 3D AP (Mod) | Recall@0.3 | Recall@0.5 | Recall@0.7 |
|------|----------------|----------------|----------------|-----------|-----------|-----------|
| baseline | 78.40 | 51.43 | 62.92 | 94.11 | 88.71 | 63.90 |
| scatter 0.5 | 78.02 | 49.58 | 61.53 | 93.36 | 87.95 | 63.45 |

---

### 6.3 object — 对抗物体放置

**文件**: `attackers/object.py`

**原理**: 在脆弱区域放置小型 3D 立方体（表面 32 点），通过 Adam 优化器联合调整物体的位置偏移 (center_delta)、三轴尺寸 (log_size) 和绕 Z 轴旋转角度 (theta)，最小化检测分数。是最"物理可实现"的白盒插入攻击。

**对应论文**: Section 5.3 (Generating 3D Adversarial Objects)。

**severity 映射**: `severity` 控制物体数量。`severity=0.3` → 2 个物体，`severity=1.0` → 8 个物体。

**可学习参数**:
| 参数 | 形状 | 说明 |
|------|------|------|
| center_delta | (N, 3) | 物体中心相对脆弱区域中心的偏移 |
| log_size | (N, 3) | 物体三轴尺寸的对数（exp 后 clamp 到 [0.3, 3.0]m）|
| theta | (N,) | 绕 Z 轴旋转角度 |

**其他参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| severity | 1.0 | 控制物体数量（max(1, severity×8)）|
| model | None | 模型实例（必需）|
| iterations | 30 | Adam 迭代次数 |
| lr | 0.05 | Adam 学习率 |

**测试命令**:
```bash
# 单帧 demo
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth \
  --data_path ../data/kitti/testing/velodyne/000008.bin \
  --attack object --severity 0.5 --iterations 20

# 全量 mAP
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth --batch_size 4 \
  --attack object --severity 0.5 --iterations 20
```

**已有结果** (severity=0.5):
| 类别 | Car 3D AP (Mod) | Ped 3D AP (Mod) | Cyc 3D AP (Mod) | Recall@0.3 | Recall@0.5 | Recall@0.7 |
|------|----------------|----------------|----------------|-----------|-----------|-----------|
| baseline | 78.40 | 51.43 | 62.92 | 94.11 | 88.71 | 63.90 |
| object 0.5 | 78.31 | 51.09 | 62.44 | 93.89 | 88.50 | 63.73 |

---

## 七、黑盒插入攻击

### 7.1 ghost_template — 幽灵模板注入

**文件**: `attackers/ghost_template.py`

**原理**: 借鉴 KITTI GT-Sampling（复制粘贴真实目标）技术，将其反转用于攻击：在点云空旷区域（远处路面 x>30m）注入预存的目标模板（Car/Ped/Cyc 的长方体表面点云），制造假阳性检测。模板来自真实目标几何形状，可在物理世界中用 3D 打印或投影仪复现。

**与 spawn/scatter/object 的区别**: spawn/scatter/object 用梯度优化生成对抗点簇（数字幻觉），ghost_template 用真实目标模板复制粘贴（物理可实现）。

**severity 映射**: `severity` 控制注入模板数量。`severity=0.1` → 1 个模板，`severity=1.0` → 10 个模板。

**模板规格**:
| 类型 | 尺寸 (m) | 表面点数 |
|------|---------|---------|
| Car | 4.0 × 1.6 × 1.5 | 500 |
| Pedestrian | 0.6 × 0.6 × 1.7 | 200 |
| Cyclist | 1.7 × 0.6 × 1.7 | 300 |

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| severity | 0.3 | 注入模板数量 = max(1, severity×10)|

**测试命令**:
```bash
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth \
  --data_path ../data/kitti/testing/velodyne/000008.bin \
  --attack ghost_template --severity 0.5

python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt ../ckpts/pointpillar_7728.pth --batch_size 4 \
  --attack ghost_template --severity 0.5
```

**已有结果** (severity=0.5):
| 类别 | Car 3D AP (Mod) | Ped 3D AP (Mod) | Cyc 3D AP (Mod) | Recall@0.3 | Recall@0.5 | Recall@0.7 |
|------|----------------|----------------|----------------|-----------|-----------|-----------|
| baseline | 78.40 | 51.43 | 62.92 | 94.11 | 88.71 | 63.90 |
| ghost_template 0.5 | 77.45 | 51.29 | 61.96 | 93.39 | 87.92 | 63.00 |

---

## 八、实验结果汇总

### KITTI 3D AP 测试结果 (AP_R40, Moderate 难度)

> KITTI 评估按 Easy / Moderate / Hard 三个难度等级分别计算 AP，不做平均。
> 下表每列为一个类别（Car / Ped / Cyc）的 **Moderate 难度 3D AP_R40**（KITTI 排行榜主指标）。

| 攻击 | 类型 | severity | Car 3D | Ped 3D | Cyc 3D |
|------|------|----------|--------|--------|--------|
| baseline | — | — | 78.40 | 51.43 | 62.92 |
| noise | 黑盒扰动 | 0.1 | 69.85 | 19.56 | 44.02 |
| noise | 黑盒扰动 | 0.3 | 6.90 | 0.006 | 0.07 |
| noise | 黑盒扰动 | 0.5 | 0.04 | 0.00 | 0.00 |
| drop | 黑盒删除 | 0.1 | 76.36 | 50.31 | 62.87 |
| drop | 黑盒删除 | 0.3 | 75.62 | 49.39 | 57.91 |
| drop | 黑盒删除 | 0.5 | 70.04 | 39.89 | 44.60 |
| geo_drop | 黑盒删除 | 0.1 | 68.17 | 42.54 | 50.42 |
| geo_drop | 黑盒删除 | 0.3 | 47.71 | 24.63 | 34.13 |
| geo_drop | 黑盒删除 | 0.5 | 35.56 | 12.40 | 25.67 |
| pgd | 白盒扰动 | 0.3 | 0.14 | 0.09 | 0.12 |
| perturb | 白盒扰动 | 0.5 | 0.60 | 0.78 | 1.39 |
| saliency_mask | 白盒删除 | 0.3 | 37.06 | 0.004 | 0.015 |
| saliency_mask | 白盒删除 | 0.5 | 11.89 | 0.00 | 0.00 |
| spawn | 白盒插入 | 0.5 | 78.32 | 50.55 | 62.12 |
| scatter | 白盒插入 | 0.5 | 78.02 | 49.58 | 61.53 |
| object | 白盒插入 | 0.5 | 78.31 | 51.09 | 62.44 |
| ghost_template | 黑盒插入 | 0.5 | 77.45 | 51.29 | 61.96 |

**结果分析**:

| 攻击类型 | 效果 | 原因 |
|---------|------|------|
| 扰动（noise/pgd/perturb） | AP 接近 0 | 坐标扰动破坏了所有点的几何结构，检测器完全失效 |
| 黑盒删除（drop/geo_drop） | AP 下降 10~80% | 随机/几何删除减少点云密度，geo_drop 定向删除更有效 |
| 白盒删除（saliency_mask） | AP 下降 50~100% | 梯度指导精准定位最关键 voxels，0.3 即可大幅降低 Car AP |
| 插入（spawn/scatter/object/ghost_template） | AP ≈ baseline | 插入攻击目标是制造假阳性，而 AP 衡量的是漏检率 |

**Severity 梯度分析（黑盒攻击）**:

| 攻击 | sev=0.1 (Car Mod) | sev=0.3 (Car Mod) | sev=0.5 (Car Mod) | 趋势 |
|------|---------|---------|---------|------|
| noise | 69.85 (-11%) | 6.90 (-91%) | 0.04 (-100%) | 极陡峭，0.3即崩溃 |
| drop | 76.36 (-3%) | 75.62 (-4%) | 70.04 (-11%) | 平缓，随机删除影响小 |
| geo_drop | 68.17 (-13%) | 47.71 (-39%) | 35.56 (-55%) | 线性下降，几何删除持续有效 |

> 关键发现：noise 攻击在 severity≥0.3 时即导致检测器完全崩溃（AP接近0），而 drop 即使在 0.5 仍保持较高性能。
> geo_drop 的删除效果介于两者之间，且 severity 与 AP 下降呈近似线性关系。

> 注：插入攻击的效果需要用“平均预测目标数”或“假阳性率”来评估，KITTI 标准 AP 不完全反映其影响。

### Recall 测试结果

| 攻击 | severity | Recall@0.3 | Recall@0.5 | Recall@0.7 |
|------|----------|-----------|-----------|-----------|
| baseline | — | 94.11 | 88.71 | 63.90 |
| drop | 0.5 | 90.93 | 82.94 | 56.31 |
| geo_drop | 0.5 | 63.71 | 58.19 | 36.88 |
| noise | 0.5 | 32.50 | 14.56 | 1.46 |
| pgd | 0.3 | 18.07 | 12.06 | 3.53 |
| perturb | 0.5 | 12.38 | 8.12 | 2.97 |
| saliency_mask | 0.3 | 1.16 | 0.59 | 0.20 |
| saliency_mask | 0.5 | 12.72 | 10.84 | 7.71 |
| spawn | 0.5 | 93.72 | 88.22 | 63.70 |
| scatter | 0.5 | 93.36 | 87.95 | 63.45 |
| object | 0.5 | 93.89 | 88.50 | 63.73 |
| ghost_template | 0.5 | 93.39 | 87.92 | 63.00 |

---

## 九、附录

### A. 文件清单

| 文件 | 说明 |
|------|------|
| `attackers/__init__.py` | 注册中心 + TestAttacker + NoiseAttacker |
| `attackers/base.py` | BaseAttacker 基类 |
| `attackers/drop.py` | DropAttacker（随机删除）|
| `attackers/geo_drop.py` | GeoDropAttacker（几何感知删除）|
| `attackers/pgd.py` | PGDAttacker（投影梯度下降）|
| `attackers/perturbation.py` | PerturbationAttacker（C&W 扰动）|
| `attackers/saliency_mask.py` | SaliencyMaskAttacker（显著性掩码删除）|
| `attackers/spawn.py` | SpawnAttacker（对抗点簇）|
| `attackers/scatter.py` | ScatterAttacker（对抗散点）|
| `attackers/object.py` | ObjectAttacker（对抗物体）|
| `attackers/ghost_template.py` | GhostTemplateAttacker（幽灵模板）|
| `attackers/critical_points.py` | CriticalPointFinder（共用梯度分析模块）|
| `attackers/metrics.py` | PerturbationMetrics（度量评估）|

### B. 度量工具

**文件**: `attackers/metrics.py`

| 度量 | 说明 | 来源 |
|------|------|------|
| L2 扰动 | 所有点的平均 L2 位移 | 论文 Section 4 |
| Chamfer Distance | 两组点云的双向最近点距离均值 | 论文 Section 5.1.1 |
| Hausdorff Distance | 两组点云的最大最近点距离 | 论文 Section 5.1.1 |
| ASR | 攻击成功率（检测分数下降超过阈值的比例）| 自定义 |
| 置信度变化 | 攻击前后平均检测置信度差 | 自定义 |

### C. 可视化工具

| 工具 | 文件 | 输出 |
|------|------|------|
| BEV 鸟瞰图 | `visual_utils/bev_visualizer.py` | PNG（黑底 viridis，多类别变色框）|
| Plotly 3D | `visual_utils/plotly_visualizer.py` | HTML（交互式扰动热力图，Reds 色标）|
| PLY 导出 | `visual_utils/ply_exporter.py` | PLY（Turbo 伪彩色，CloudCompare 可打开）|

### D. 实验管线

| 脚本 | 说明 |
|------|------|
| `demo.py` | 单帧 demo + 多维可视化 |
| `test.py` | 全量 mAP 评估（支持 `--attack` 参数）|
| `experiment.py` | 批量实验：多攻击×多severity×多帧，输出 CSV + 论文图 |
| `test_pgd.py` | PGD 单帧测试 + 并排 BEV 对比 |
| `test_perturb.py` | C&W 单帧测试 + 并排 BEV 对比 |
| `test_metrics.py` | 度量综合测试 |
