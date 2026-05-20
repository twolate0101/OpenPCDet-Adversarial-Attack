# 快速开始

## clone新的仓库
```bash
# 1. 将老镜像里旧的代码文件夹改名备份
mv OpenPCDet OpenPCDet_bak

# 2. 克隆 Fork 后的新仓库
git clone https://github.com/twolate0101/OpenPCDet-Adversarial-Attack.git OpenPCDet
cd OpenPCDet

# 3. 本地编译
python setup.py develop
```
## 一些环境准备
```bash
# 1. 把老镜像的模型的ckpts(模型权重 .pth)复制到新文件夹
mkdir -p ~/OpenPCDet/ckpts

cp ~/OpenPCDet_bak/ckpts/pointpillar_7728.pth ~/OpenPCDet/ckpts/

# 2. 检查数据盘是否有 kitti 数据包，(查看autodl控制台,如果数据盘占用 53% ~ 55% 那么大概是解压过了)如果没有就按照下面的步骤重新解压一份到数据盘
mkdir -p /root/autodl-tmp/kitti

unzip -q /root/autodl-pub/KITTI/object/data_object_calib.zip -d /root/autodl-tmp/kitti/
unzip -q /root/autodl-pub/KITTI/object/data_object_label_2.zip -d /root/autodl-tmp/kitti/
unzip -q /root/autodl-pub/KITTI/object/data_object_velodyne.zip -d /root/autodl-tmp/kitti/

# 3.创建新的软链接,链接数据盘的kitti数据包
cd ~/OpenPCDet/data/kitti

ln -s /root/autodl-tmp/kitti/training .
ln -s /root/autodl-tmp/kitti/testing .
# 这时在 ~/OpenPCDet/data/kitti 目录下会有 training 和 testing 两个软链接，指向数据盘的 kitti 数据包,以及一个ImageSets 目录

# 4. 重新生成数据索引文件 (这步需要一两分钟)
cd ~/OpenPCDet
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

```
## 测试demo.py
```bash
# 进入/tools
cd ~/OpenPCDet/tools

# 1.无损空跑,不带攻击参数 (应检测到 58 个目标,图片正常)
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../ckpts/pointpillar_7728.pth --data_path ../data/kitti/testing/velodyne/000008.bin

# 2. 测试攻击 (会输出INFO:  已启用攻击模式: test，强度: 1.0 .并且无法正常检查,报告 IndexError: too many indices for tensor of dimension 1)
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../ckpts/pointpillar_7728.pth --data_path ../data/kitti/testing/velodyne/000008.bin --attack test

# 3.测试微扰攻击 (应检测到更少(如48,24)的目标,图片有明显的噪点且检测框减少)
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../ckpts/pointpillar_7728.pth --data_path ../data/kitti/testing/velodyne/000008.bin --attack noise --severity 0.5
```

# 协作建议

## 建立个人特征分支
```bash
# 1. 确保当前在最新的主分支，且代码是最新的
git checkout main
git pull origin main

# 2. 建立并切换到自己的攻击算法分支（例如：,noise,drop,attack-pgd, attack-jsma）
git checkout -b 你的算法名
```
## 协作工作流
### 开发规范
1. 在 在自己的 分支上编写算法。
2. 在`tools/attackers/`下创建一个新的 Python 文件，例如`drop.py`，并实现你的攻击算法。(遵守`tools/attackers/base.py`里定义的接口规范，确保你的攻击类继承自`BaseAttacker`并实现`forward`方法) ,编写完后在`tools/attackers/__init__.py`里注册你的攻击类。
开发规范（实现与注册）

* 位置：把每个攻击器放到 tools/attackers/ 下（示例文件名：drop.py, noise.py）。
* 基类：所有攻击器继承 BaseAttacker，必须实现 forward(self, data_dict) 并返回与原流程兼容的 data_dict。不要修改 data_dict 的关键字段结构（例如保持 data_dict['points'] 为形状正确的张量或 numpy）。
* 推荐文件结构（示例）：
    * base.py — BaseAttacker（已存在）
    * tools/attackers/noise.py — NoiseAttacker
    * tools/attackers/drop.py — DropAttacker
    * `__init__.py` — 工厂函数 get_attacker 注册和导出
* 工厂注册：在 get_attacker 中新增分支并返回相应类的实例，签名形如 get_attacker(attack_type, severity, **kwargs)。
3. 示例
```python
# drop.py 示例
from .base import BaseAttacker
import torch

class DropAttacker(BaseAttacker):
    def forward(self, data_dict):
        points = data_dict['points']
        # 纯 PyTorch 张量操作：根据 severity 随机丢弃一定比例的点
        drop_mask = torch.rand(points.shape[0], device=points.device) > self.severity
        data_dict['points'] = points[drop_mask]
        return data_dict
```

注入点与运行入口

* demo.py：在体素化前通过劫持 dataset.prepare_data 注入（已经实现）。
* test.py：通过 register_forward_pre_hook 把攻击注入到模型前向（已经实现）。
* 两个入口均使用 CLI 参数 --attack 和 --severity 来控制，故在大多数情况下无需新增 CLI 参数。

4. 单帧验毒: 改变attack 或severity 参数  运行 
```
cd ~/OpenPCDet/tools

python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../ckpts/pointpillar_7728.pth --data_path ../data/kitti/testing/velodyne/000008.bin --attack noise --severity 0.5
```
查看输出检测到的目标数量,生成的 `result_bev.png` 里的检测框及雷达图变化.
说明：攻击效果依赖于帧、模型和 severity，不保证固定的目标数量变化。

### 提交流程（建议）
工作流示例：
```bash
git checkout master
git pull origin master
git checkout -b feat/<your-attack-name>
# 开发、测试
git add tools/attackers/<your_file>.py
git commit -m "Add <Your> attacker: <brief desc>"
git push -u origin feat/<your-attack-name>
# 在 GitHub 上打开 PR，指向 master
```
在 PR 描述里附上最小复现命令（copy 上面的 demo 命令），并标注会改变哪些字段或返回值格式（若有）.

注意事项

* 提交前请确保你的改动至少能在一帧上跑通 demo（避免因接口不兼容导致 CI/测试失败）。
*若攻击器需要额外超参数（除了 severity），请：
    * 在 demo.py/test.py 的 argparse 中添加参数，或让 get_attacker 接收 **kwargs，
    * 在 README/PR 中说明该参数用途与默认值，
    * 在 eval 输出目录中把参数信息写入目录名或日志文件以区分实验结果.


