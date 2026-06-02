import argparse
from pathlib import Path


def main():
    import os
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Smoke test CompanyNuScenes dataloader')
    parser.add_argument(
        '--cfg_file',
        type=Path,
        default=repo_root / 'tools' / 'cfgs' / 'nuscenes_models' / 'company_fshnet_26cls_trainval.yaml'
    )
    args = parser.parse_args()
    cfg_file = args.cfg_file if args.cfg_file.is_absolute() else repo_root / args.cfg_file

    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils

    cfg_from_yaml_file(str(cfg_file), cfg)
    logger = common_utils.create_logger()
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=True
    )
    batch = next(iter(dataloader))
    print('dataset_len:', len(dataset))
    print('batch_keys:', sorted(batch.keys()))
    print('points_shape:', batch['points'].shape)
    print('gt_boxes_shape:', batch['gt_boxes'].shape)
    print('first_gt_box:', batch['gt_boxes'][0, 0].tolist() if batch['gt_boxes'].shape[1] else None)


if __name__ == '__main__':
    main()
