# T23 2026 CompanyNuScenes 数据集适配与训练指南

本文档记录 `FSHNet_ljl` 对 `NuScenes-develop_t23_2026` 数据集的适配方式，以及在 Ubuntu 服务器上训练 10 类 FSHNet 模型的完整流程。

## 1. 适用范围

本指南适用于下面的服务器目录约定：

- 项目目录：`~/WXY/pointcloud_Projects/FSHNet_ljl`
- 数据集目录：`~/WXY/data_ljl/NuScenes-develop_t23_2026`
- Python 环境：`/home/ubuntu/anaconda3/envs/fshnet/bin/python`
- GitHub 仓库：`https://github.com/ljl71/FSHNet_ljl`
- 分支：`main`

建议先确认代码版本至少包含以下提交：

- `a7bdd6b Adapt CompanyNuScenes to T23 2026 dataset`
- `543e0f4 Align CompanyNuScenes 10-class tools with T23 dataset`
- `ba1622f Support compressed T23 PCD headers`

## 2. 数据集结构

T23 2026 数据集是 nuScenes-like 自定义数据集，不是官方 nuScenes 原包。当前适配默认使用：

- `VERSION: v1.0-develop`
- `DATA_PATH: data/NuScenes-develop_t23_2026`
- `MAX_SWEEPS: 1`
- `LIDAR_POINT_FORMAT: pcd`
- `LIDAR_POINT_FIELDS: ['x', 'y', 'z', 'intensity']`
- `PRED_VELOCITY: False`

典型目录如下：

```text
NuScenes-develop_t23_2026/
  v1.0-develop/
    sample.json
    sample_data.json
    sample_annotation.json
    scene.json
    ...
  samples/
    2026-01-22/
      2026-01-22-18-14-36_uid1/
        lidar_fusion/
          *.pcd
        camera_cam*_compressed/
          *.jpg
        ...
```

点云文件为 PCD。当前读取器已经支持常见的 `DATA binary`、`DATA ascii`、`DATA binary_compressed`，并兼容仅提供 `WIDTH/HEIGHT` 而没有 `POINTS` 的 PCD header。

## 3. 本项目的主要适配点

本项目新增或修改了以下能力：

1. `CompanyNuScenesDataset` 支持 T23 2026 的 nuScenes-like JSON 表结构。
2. 支持从 `sample_data.json` 查找 `LIDAR_TOP`，解决部分 `sample.json` 缺少 `data` 字段的问题。
3. 支持 T23 数据中的 PCD 点云读取。
4. 支持旧绝对路径重定向：如果 info 中保存的是旧机器路径，dataloader 会根据 `samples/...` 后缀映射到当前 `DATA_PATH`。
5. 支持 26 类 CompanyNuScenes 训练与评估。
6. 支持将 26 类标注合并为 10 类，生成 10 类训练/验证 info。
7. 支持 26 类模型预测结果的 10 类后验合并评估。
8. `Argo2Dataset` 改为可选导入，不再因为服务器没有 `av2` 影响公司数据集训练。
9. 修复 `TransFusionHead` 在公司数据集评估阶段依赖原始 nuScenes 多 task `self.tasks` 的问题。

## 4. 10 类类别定义

10 类训练使用下面的类别顺序：

```python
[
    'pedestrian',
    'car',
    'bus',
    'truck',
    'emergency_vehicle',
    'two_wheeler',
    'barrier',
    'traffic_cone',
    'movable_object',
    'other'
]
```

10 类配置文件：

```text
tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml
```

10 类数据配置：

```text
tools/cfgs/dataset_configs/company_nuscenes_10cls_trainval_dataset.yaml
```

正式 train/val/test 场景级划分配置：

```text
tools/cfgs/nuscenes_models/company_fshnet_10cls_trainvaltest.yaml
tools/cfgs/dataset_configs/company_nuscenes_10cls_trainvaltest_dataset.yaml
```

## 5. Ubuntu 服务器完整训练流程

下面所有命令都是单行命令，适合通过 ToDesk 复制到服务器终端执行。

### 5.1 拉取 GitHub 最新代码

如果服务器已有旧代码目录，直接强制同步到 GitHub `main`：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && git fetch origin main && git reset --hard origin/main && git log -1 --oneline
```

如果远程地址不正确，先修正 remote：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && git remote set-url origin https://github.com/ljl71/FSHNet_ljl.git && git fetch origin main && git reset --hard origin/main && git log -1 --oneline
```

### 5.2 创建数据集软链接

项目配置默认读取 `data/NuScenes-develop_t23_2026`。服务器真实数据在 `~/WXY/data_ljl/NuScenes-develop_t23_2026`，因此需要创建软链接：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && mkdir -p data && ln -sfnT ~/WXY/data_ljl/NuScenes-develop_t23_2026 data/NuScenes-develop_t23_2026 && ls -ld data/NuScenes-develop_t23_2026
```

检查数据集版本目录：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && ls data/NuScenes-develop_t23_2026/v1.0-develop && ls data/NuScenes-develop_t23_2026/samples | head
```

### 5.3 生成 26 类源 info

10 类 info 是从 26 类源 info 合并得到的，所以必须先生成 26 类 info：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && /home/ubuntu/anaconda3/envs/fshnet/bin/python tools/company_nuscenes/create_company_infos.py --data_path data/NuScenes-develop_t23_2026 --save_path data/NuScenes-develop_t23_2026 --version v1.0-develop --max_sweeps 1 --min_lidar_points 1
```

成功后应看到类似输出：

```text
Company nuScenes train infos: 53645
Company nuScenes val infos: 13511
Saved: data/NuScenes-develop_t23_2026/v1.0-develop/company_nuscenes_infos_train.pkl
Saved: data/NuScenes-develop_t23_2026/v1.0-develop/company_nuscenes_infos_val.pkl
```

### 5.4 生成 10 类 train/val info

当前推荐先使用 train/val 方案。该方案会把已有 26 类 train/val info 映射为 10 类 train/val info：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && /home/ubuntu/anaconda3/envs/fshnet/bin/python tools/company_nuscenes/create_company_10cls_infos.py --data_path data/NuScenes-develop_t23_2026 --version v1.0-develop --split_mode trainval
```

成功后应看到每个 10 类的 GT 数量，并显示：

```text
[train] all original classes were mapped successfully.
[val] all original classes were mapped successfully.
[total] all original 26 classes were mapped to the 10 merged classes.
```

检查生成文件：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && ls data/NuScenes-develop_t23_2026/v1.0-develop/company_nuscenes_infos_train.pkl data/NuScenes-develop_t23_2026/v1.0-develop/company_nuscenes_infos_val.pkl data/NuScenes-develop_t23_2026/v1.0-develop/company_nuscenes_10cls_infos_train.pkl data/NuScenes-develop_t23_2026/v1.0-develop/company_nuscenes_10cls_infos_val.pkl
```

### 5.5 可选：快速检查 dataloader

正式训练前可以先跑 dataloader smoke test：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && /home/ubuntu/anaconda3/envs/fshnet/bin/python tools/company_nuscenes/smoke_test_company_dataloader.py --cfg_file tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml --data_path data/NuScenes-develop_t23_2026 --version v1.0-develop
```

如果这里能正常读出样本、点云 shape 和 GT boxes，说明数据链路基本可用。

## 6. 开始 10 类训练

双卡训练命令：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && CUDA_VISIBLE_DEVICES=0,1 /home/ubuntu/anaconda3/envs/fshnet/bin/python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --launcher pytorch --cfg_file tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml --batch_size 2 --extra_tag t23_2026_10cls
```

如果显存不足，可以把 batch size 降到 1：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && CUDA_VISIBLE_DEVICES=0,1 /home/ubuntu/anaconda3/envs/fshnet/bin/python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --launcher pytorch --cfg_file tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml --batch_size 1 --extra_tag t23_2026_10cls_bs1
```

训练输出默认位于：

```text
output/nuscenes_models/company_fshnet_10cls_trainval/<extra_tag>/
```

其中 checkpoint 通常在：

```text
output/nuscenes_models/company_fshnet_10cls_trainval/<extra_tag>/ckpt/
```

## 7. 使用 tmux 后台训练

推荐用 tmux 跑长时间训练，断开 ToDesk 或 SSH 后训练仍会继续。

新建 tmux session 并直接启动双卡 10 类训练：

```bash
tmux new -s fshnet10 'cd ~/WXY/pointcloud_Projects/FSHNet_ljl && CUDA_VISIBLE_DEVICES=0,1 /home/ubuntu/anaconda3/envs/fshnet/bin/python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --launcher pytorch --cfg_file tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml --batch_size 2 --extra_tag t23_2026_10cls'
```

在 tmux 中临时退出但不中断训练：

```text
Ctrl-b 然后按 d
```

重新进入训练窗口：

```bash
tmux attach -t fshnet10
```

查看已有 tmux session：

```bash
tmux ls
```

如果需要停止训练，进入 tmux 后按 `Ctrl-c`，或在外部关闭 session：

```bash
tmux kill-session -t fshnet10
```

## 8. 训练后评估

训练完成后，选择某个 checkpoint 进行单卡评估。下面以 epoch 36 为例：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && CUDA_VISIBLE_DEVICES=0 /home/ubuntu/anaconda3/envs/fshnet/bin/python tools/test.py --cfg_file tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml --batch_size 1 --ckpt output/nuscenes_models/company_fshnet_10cls_trainval/t23_2026_10cls/ckpt/checkpoint_epoch_36.pth
```

双卡评估命令：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && CUDA_VISIBLE_DEVICES=0,1 /home/ubuntu/anaconda3/envs/fshnet/bin/python -m torch.distributed.launch --nproc_per_node=2 tools/test.py --launcher pytorch --cfg_file tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml --batch_size 2 --ckpt output/nuscenes_models/company_fshnet_10cls_trainval/t23_2026_10cls/ckpt/checkpoint_epoch_36.pth
```

评估成功时应进入：

```text
CompanyNuScenes distance evaluation
```

并输出每类 AP、mAP、P/R/F1 @2m 和距离分段结果。

## 9. 26 类结果后验合并为 10 类评估

如果已经有 26 类模型的 `result.pkl`，可以不重新训练模型，先做 10 类后验合并评估：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && /home/ubuntu/anaconda3/envs/fshnet/bin/python tools/company_nuscenes/evaluate_26cls_predictions_as_10cls.py --pred_pkl output/nuscenes_models/company_fshnet_26cls_trainval/default/eval/epoch_36/val/default/result.pkl --data_path data/NuScenes-develop_t23_2026 --version v1.0-develop --info_pkl data/NuScenes-develop_t23_2026/v1.0-develop/company_nuscenes_10cls_infos_val.pkl --output_dir output/company_10cls_merged_eval_from_26cls
```

注意：26 类模型的 10 类后验合并评估不等价于真正的 10 类训练模型。正式 10 类 baseline 仍然需要用 10 类 info 和 10 类配置重新训练。

## 10. 常见问题

### 10.1 `No module named 'av2'`

原因：Argo2 依赖 `av2`，但公司数据集不需要 Argo2。当前代码已将 `Argo2Dataset` 改为可选导入。确认代码已更新到 GitHub 最新：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && git fetch origin main && git reset --hard origin/main && git log -1 --oneline
```

### 10.2 `company_nuscenes_infos_train.pkl` 不存在

原因：还没生成 26 类源 info。先执行：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && /home/ubuntu/anaconda3/envs/fshnet/bin/python tools/company_nuscenes/create_company_infos.py --data_path data/NuScenes-develop_t23_2026 --save_path data/NuScenes-develop_t23_2026 --version v1.0-develop --max_sweeps 1 --min_lidar_points 1
```

然后再生成 10 类 info。

### 10.3 `Unsupported PCD header`

原因：旧版 PCD 读取器不支持 T23 数据中的某些 PCD header 或压缩格式。当前代码已修复。确认最新提交至少包含：

```text
ba1622f Support compressed T23 PCD headers
```

### 10.4 CUDA 显存不足

优先降低 batch size：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && CUDA_VISIBLE_DEVICES=0,1 /home/ubuntu/anaconda3/envs/fshnet/bin/python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --launcher pytorch --cfg_file tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml --batch_size 1 --extra_tag t23_2026_10cls_bs1
```

### 10.5 训练中断后继续训练

找到最新 checkpoint 后使用 `--ckpt` 继续：

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && CUDA_VISIBLE_DEVICES=0,1 /home/ubuntu/anaconda3/envs/fshnet/bin/python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --launcher pytorch --cfg_file tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml --batch_size 2 --extra_tag t23_2026_10cls --ckpt output/nuscenes_models/company_fshnet_10cls_trainval/t23_2026_10cls/ckpt/latest_model.pth
```

如果没有 `latest_model.pth`，把路径替换为实际存在的 `checkpoint_epoch_*.pth`。

## 11. 推荐执行顺序总结

第一次完整训练建议按下面顺序执行：

1. 拉取 GitHub 最新代码。
2. 创建 `data/NuScenes-develop_t23_2026` 软链接。
3. 生成 26 类源 info。
4. 生成 10 类 train/val info。
5. 检查 info 文件是否存在。
6. 可选跑 dataloader smoke test。
7. 用 tmux 启动 10 类训练。
8. 训练完成后用 `tools/test.py` 评估。
