# CompanyNuScenes 类别语义合并实验说明

## 背景说明

公司自建 nuScenes 格式点云数据集保留了 26 个细粒度类别。这种标注体系能够较完整地描述真实道路环境中的目标类型，例如不同年龄或状态的行人、不同用途的车辆、交通设施以及可移动障碍物等。细粒度标签对数据资产沉淀和业务分析很有价值，但直接用于 3D 检测训练时会带来明显的长尾问题。

在当前 26 类验证集评估中，整体 mAP 为 0.1546。部分类别的验证集 GT 数量很少，例如 `human_pedestrian_wheelchair` 仅 7 个，`vehicle_emergency_ambulance` 仅 6 个，`movable_object_pushable_pullable` 仅 8 个，`other` 仅 13 个。这类低频类别会导致两个问题：一方面模型很难从少量样本中学习稳定的外观和几何模式；另一方面宏平均 mAP 会受到小样本类别的显著影响，使整体指标同时混合了检测能力、类别分布和样本稀疏性。

因此，本实验新增一套 10 类主语义类别任务。它不替代原始 26 类任务，而是在保持相同点云、相同 train/val 划分和相同评估协议的基础上，将细粒度类别按语义和业务用途合并，用于观察类别重组后检测稳定性和主类别性能变化。

## 为什么不直接只用官方数据集

nuScenes、Waymo 和 Argoverse2 等公开数据集是通用自动驾驶检测基准，具有成熟的类别体系和评估协议。但公司自建数据集反映的是公司实际采集设备、点云格式、标注规范、道路场景和业务部署目标。直接只使用官方数据集无法覆盖这些工程和业务差异，也无法回答模型在公司真实数据分布下的表现。

本项目中的类别合并并不是为了简单模仿官方 nuScenes 的 10 类设置，而是为了把公司原始 26 类细粒度标注重组为更适合模型学习和部署分析的主语义类别体系。合并后的类别仍然来自公司数据本身，数据划分、点云读取方式和评估样本均保持不变，因此可以更直接地比较细粒度检测和主类别检测之间的差异。

## 为什么不是简单删除小类

长尾类别样本少，并不意味着它们没有价值。若直接删除小类样本，训练集会丢失一部分真实道路目标，评估也无法反映这些目标对主类别检测的贡献。更合理的做法是根据语义相近性、业务需求和样本分布，将细粒度类别合并到更高层级的语义类别中。

例如，`human_pedestrian_child`、`human_pedestrian_wheelchair` 和 `human_pedestrian_stroller` 在细粒度语义上不同，但对于 3D 检测模型而言，它们都属于行人相关目标，几何中心和空间占据模式也更接近行人主类。将它们合并到 `pedestrian` 后，长尾样本仍然参与训练和评估，同时可以缓解单个小类样本过少造成的指标波动。

## 类别合并原则

类别合并遵循以下原则：

- 语义一致性：合并后的类别应保持明确的语义边界，避免把业务含义差异过大的目标放入同一类。
- 几何形态相近性：3D 检测依赖目标的空间位置、尺寸和朝向，合并类别应尽量具有相近的几何形态或检测模式。
- 检测任务可学习性：合并后的类别需要有足够样本支撑模型学习，减少单一类别样本过少导致的过拟合和召回不稳定。
- 实际部署需求：主类别应对应工程部署中更常用的目标分组，例如行人、车辆、两轮车、交通锥和障碍物等。
- 缓解长尾分布：将低频细粒度类别并入主语义类别，使长尾样本继续贡献监督信号。
- 评估稳定性：避免连续帧相关性和类别稀疏性导致单类 AP 大幅波动，使宏平均指标更能反映主类别检测能力。

## 26 类到 10 类映射表

| 合并后类别 | 原始类别 |
| --- | --- |
| pedestrian | human_pedestrian_adult, human_pedestrian_child, human_pedestrian_wheelchair, human_pedestrian_stroller, human_pedestrian_personal_mobility, group_human_pedestrian |
| car | vehicle_car |
| bus | vehicle_bus_bendy, vehicle_bus_rigid |
| truck | vehicle_truck, vehicle_construction, vehicle_trailer |
| emergency_vehicle | vehicle_emergency_ambulance, vehicle_emergency_police, vehicle_emergency_other |
| two_wheeler | vehicle_motorcycle, vehicle_bicycle, group_vehicle_bicycle, vehicle_tricycle, bicycle |
| barrier | movable_object_barrier |
| traffic_cone | movable_object_trafficcone |
| movable_object | movable_object_pushable_pullable, movable_object_debris |
| other | animal, other |

上述方案保留了用户给出的 10 类语义设计。`vehicle_construction` 和 `vehicle_trailer` 被放入 `truck`，主要考虑它们与大型车辆在几何尺度和检测形态上更接近；`vehicle_emergency_*` 被单独保留为 `emergency_vehicle`，是为了保留应急车辆的业务语义；两轮和轻型骑行相关类别统一到 `two_wheeler`，用于减少细粒度骑行类样本不足带来的不稳定。

## 新增文件和使用流程

类别映射定义在：

```text
pcdet/datasets/company_nuscenes/company_class_mapping.py
```

10 类 info 由原始 26 类 info 转换生成，脚本为：

```text
tools/company_nuscenes/create_company_10cls_infos.py
```

10 类数据集配置为：

```text
tools/cfgs/dataset_configs/company_nuscenes_10cls_trainval_dataset.yaml
```

10 类 FSHNet 训练配置为：

```text
tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml
```

原始 26 类配置 `tools/cfgs/nuscenes_models/company_fshnet_26cls_trainval.yaml` 不需要修改，仍然用于原始细粒度类别实验。

## 实验命令

生成 10 类 info：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl
/home/ubuntu/anaconda3/envs/fshnet/bin/python tools/company_nuscenes/create_company_10cls_infos.py --data_path data/nuscenes --version v1.0-trainval
```

该命令读取：

```text
data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

并生成：

```text
data/nuscenes/v1.0-trainval/company_nuscenes_10cls_infos_train.pkl
data/nuscenes/v1.0-trainval/company_nuscenes_10cls_infos_val.pkl
```

训练 10 类 FSHNet：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl
CUDA_VISIBLE_DEVICES=0,1 /home/ubuntu/anaconda3/envs/fshnet/bin/python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --launcher pytorch --cfg_file tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml --batch_size 2 --workers 4 --extra_tag company_fshnet_10cls_2gpu_bs2_run1
```

测试 10 类 FSHNet：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/anaconda3/envs/fshnet/bin/python tools/test.py --cfg_file tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml --batch_size 1 --ckpt output/nuscenes_models/company_fshnet_10cls_trainval/company_fshnet_10cls_2gpu_bs2_run1/ckpt/checkpoint_epoch_36.pth
```

## 实验设计建议

论文实验中建议同时报告原始 26 类 baseline 和合并 10 类 baseline。26 类结果用于说明细粒度类别检测在长尾分布下的困难，10 类结果用于说明类别语义重组后主类别检测的稳定性。二者关注的问题不同，不能只报告合并后更好看的指标。

建议至少报告以下内容：

- 原始 26 类 baseline 的整体 mAP 和分类别 AP。
- 合并 10 类 baseline 的主类别 mAP 和分类别 AP。
- 不同距离区间的 mAP、precision、recall 和 F1。
- 评估时的 TP、FP、FN 和平均每帧预测数量。
- 部分典型类别的对比分析，例如行人、车辆、两轮车、交通锥和可移动物体。

如果合并后指标提高，应解释为主语义类别任务降低了细粒度长尾分类难度，而不能简单表述为模型整体能力必然增强。若某些主类仍然表现较弱，应进一步分析其样本数量、空间尺度、遮挡、点云稀疏程度和标注一致性。

## 论文表述建议

可以在论文的数据集部分使用如下表述：

> 公司自建点云数据集采用 nuScenes 格式组织，但其类别体系包含 26 个细粒度目标类别。该类别设计能够较完整地覆盖真实采集场景中的行人、车辆、交通设施和其他道路目标，但也带来了明显的长尾分布。部分类别在验证集中样本数量很少，导致单类 AP 容易受少量样本和连续帧相关性的影响，进而影响宏平均 mAP 的稳定性。

可以在方法或实验设置部分使用如下表述：

> 为了分析细粒度类别长尾问题对检测性能的影响，本文在保留原始 26 类任务的同时，构建了一套主语义类别合并方案。该方案不删除低频类别，而是依据语义一致性、几何形态相近性和实际部署需求，将原始 26 类重组为 10 个主类别。这样既保留了长尾样本的监督信息，也降低了细粒度类别之间的混淆难度。

可以在与公开数据集差异部分使用如下表述：

> 与 nuScenes、Waymo 和 Argoverse2 等公开基准不同，公司数据集反映的是特定采集设备、标注体系和业务场景下的数据分布。本文的类别合并并非直接复用公开数据集类别，而是在公司原始标注体系内部进行语义重组，以便更准确地评估模型在公司应用场景中的主类别检测能力。

可以在实验结果分析部分使用如下表述：

> 26 类实验主要用于揭示细粒度类别检测中的长尾问题和类别混淆现象，10 类实验则用于评估语义合并后主类别检测的稳定性。若合并后主类别 mAP 和召回率提升，说明类别重组能够缓解低频小类对宏平均指标的影响，并为后续实际部署提供更稳定的类别粒度。但该结果不能替代原始 26 类实验，因为二者对应不同的任务定义和评价目标。

## 注意事项

- 10 类合并实验不能直接替代原始 26 类实验，两者应作为不同任务分别报告。
- 论文中使用 10 类结果时，必须明确列出 26 类到 10 类的映射关系。
- 所有模型比较必须使用相同的合并规则、相同的 train/val/test 划分和相同的评估协议。
- 10 类 info 文件由 26 类 info 文件派生，不应覆盖原始 26 类 info 文件。
- 如果未来单独划分 test set，应按 scene 划分，避免连续帧泄漏导致训练集和测试集高度相关。
- 合并后的 `other` 类语义较宽，论文中应谨慎解释该类 AP，不宜将其作为核心类别性能代表。
