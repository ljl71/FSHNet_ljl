# Company NuScenes FSHNet Adapter

This directory contains the first-stage company 26-class nuScenes-style data adapter for FSHNet.

Key boundaries:

- Dataset: `CompanyNuScenesDataset`
- Model config: `tools/cfgs/nuscenes_models/company_fshnet_26cls_trainval.yaml`
- LiDAR only, single frame: `MAX_SWEEPS: 1`
- Point format: `x, y, z, intensity`
- `gt_sampling` is disabled for the company config
- FSHNet keeps its original `vel` head; GT velocity is zero-filled only for shape compatibility
- Evaluation uses custom company 26-class distance metrics and does not report official nuScenes NDS

Useful commands from the repository root:

```bash
python tools/company_nuscenes/create_company_infos.py

python tools/company_nuscenes/check_company_infos.py \
  --cfg_file tools/cfgs/nuscenes_models/company_fshnet_26cls_trainval.yaml

python tools/company_nuscenes/smoke_test_company_dataloader.py \
  --cfg_file tools/cfgs/nuscenes_models/company_fshnet_26cls_trainval.yaml

python tools/company_nuscenes/smoke_test_company_evaluation.py
```

Training smoke test:

```bash
python tools/train.py \
  --cfg_file tools/cfgs/nuscenes_models/company_fshnet_26cls_trainval.yaml \
  --batch_size 1 \
  --epochs 1 \
  --workers 0 \
  --extra_tag smoke_company_fshnet
```

Evaluation smoke test:

```bash
python tools/test.py \
  --cfg_file tools/cfgs/nuscenes_models/company_fshnet_26cls_trainval.yaml \
  --batch_size 1 \
  --workers 0 \
  --ckpt <checkpoint> \
  --extra_tag smoke_company_fshnet_eval
```

`CompanyNuScenesDataset.evaluation()` writes `company_metrics_summary.json` next to `result.pkl` when `tools/test.py` supplies an output directory.
