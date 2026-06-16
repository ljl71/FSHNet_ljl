# T23 Ubuntu one-line commands

These commands assume:

- project: `~/WXY/pointcloud_Projects/FSHNet_ljl`
- dataset: `~/WXY/data_ljl/NuScenes-develop_t23_2026`
- conda env python: `/home/ubuntu/anaconda3/envs/fshnet/bin/python`

Update the repository to GitHub `main`:

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && git fetch origin main && git reset --hard origin/main && git log -1 --oneline
```

Link the dataset to the default path used by the T23 configs:

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && mkdir -p data && ln -sfnT ~/WXY/data_ljl/NuScenes-develop_t23_2026 data/NuScenes-develop_t23_2026 && ls -ld data/NuScenes-develop_t23_2026
```

Generate the source 26-class info files:

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && /home/ubuntu/anaconda3/envs/fshnet/bin/python tools/company_nuscenes/create_company_infos.py --data_path data/NuScenes-develop_t23_2026 --save_path data/NuScenes-develop_t23_2026 --version v1.0-develop --max_sweeps 1 --min_lidar_points 1
```

Generate the merged 10-class train/val info files:

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && /home/ubuntu/anaconda3/envs/fshnet/bin/python tools/company_nuscenes/create_company_10cls_infos.py --data_path data/NuScenes-develop_t23_2026 --version v1.0-develop --split_mode trainval
```

Check the generated 10-class info files:

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && ls data/NuScenes-develop_t23_2026/v1.0-develop/company_nuscenes_10cls_infos_train.pkl data/NuScenes-develop_t23_2026/v1.0-develop/company_nuscenes_10cls_infos_val.pkl
```

Run 10-class FSHNet training:

```bash
cd ~/WXY/pointcloud_Projects/FSHNet_ljl && CUDA_VISIBLE_DEVICES=0,1 /home/ubuntu/anaconda3/envs/fshnet/bin/python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --launcher pytorch --cfg_file tools/cfgs/nuscenes_models/company_fshnet_10cls_trainval.yaml --batch_size 2 --extra_tag t23_2026_10cls
```
