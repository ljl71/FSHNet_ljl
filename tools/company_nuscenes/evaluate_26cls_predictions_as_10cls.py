"""
Evaluate 26-class CompanyNuScenes predictions after posterior 10-class merging.

This script does not train a model. It reads an OpenPCDet result.pkl generated
by a 26-class model, maps det_anno["name"] from original company 26-class names
to merged 10-class names, rebuilds 1-based 10-class pred_labels, and evaluates
against 10-class CompanyNuScenes info files.
"""

import argparse
import importlib.util
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_PRED_PKL = (
    'output/nuscenes_models/company_fshnet_26cls_trainval/default/'
    'eval/epoch_36/val/default/result.pkl'
)
DEFAULT_OUTPUT_DIR = 'output/company_10cls_merged_eval_from_26cls'


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate saved 26-class CompanyNuScenes predictions as merged 10-class predictions.'
    )
    parser.add_argument('--pred_pkl', type=Path, default=Path(DEFAULT_PRED_PKL))
    parser.add_argument('--data_path', type=Path, default=Path('data/nuscenes'))
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--info_pkl', type=Path, default=None)
    parser.add_argument('--output_dir', type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--min_lidar_points', type=int, default=1)
    parser.add_argument('--distance_thresholds', nargs='+', type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument('--tp_distance_threshold', type=float, default=2.0)
    parser.add_argument(
        '--min_score', type=float, default=None,
        help='Optionally drop predictions below this score during evaluation.'
    )
    return parser.parse_args()


def default_info_path(data_path, version):
    return data_path / version / 'company_nuscenes_10cls_infos_val.pkl'


def require_existing_info(info_path):
    if info_path.exists():
        return
    command = (
        'python tools/company_nuscenes/create_company_10cls_infos.py '
        '--data_path data/nuscenes --version v1.0-trainval --split_mode trainval'
    )
    raise FileNotFoundError(
        f'10-class info file not found: {info_path}\n'
        f'Please generate it first, for example:\n{command}'
    )


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def map_prediction_names_to_10cls(det_annos, name_mapping, class_names_10):
    label_by_name = {name: idx + 1 for idx, name in enumerate(class_names_10)}
    converted_annos = []
    original_counter = Counter()
    merged_counter = Counter()
    unmapped_counter = Counter()

    for frame_idx, det_anno in enumerate(det_annos):
        original_names = np.asarray(det_anno.get('name', []))
        mapped_names = []
        pred_labels = []

        for name in original_names:
            name = str(name)
            original_counter[name] += 1
            mapped_name = name_mapping.get(name)
            if mapped_name is None:
                unmapped_counter[name] += 1
                continue
            mapped_names.append(mapped_name)
            pred_labels.append(label_by_name[mapped_name])
            merged_counter[mapped_name] += 1

        if unmapped_counter:
            continue

        converted = {
            'name': np.asarray(mapped_names),
            'score': np.asarray(det_anno.get('score', [])),
            'boxes_lidar': np.asarray(det_anno.get('boxes_lidar', [])),
            'pred_labels': np.asarray(pred_labels, dtype=np.int64),
        }
        if 'frame_id' in det_anno:
            converted['frame_id'] = det_anno['frame_id']
        if 'metadata' in det_anno:
            converted['metadata'] = det_anno['metadata']

        validate_converted_lengths(converted, frame_idx)
        converted_annos.append(converted)

    if unmapped_counter:
        details = ', '.join(f'{name}: {count}' for name, count in sorted(unmapped_counter.items()))
        raise ValueError(f'Unmapped prediction classes found: {details}')

    return converted_annos, original_counter, merged_counter


def validate_converted_lengths(det_anno, frame_idx):
    num_names = len(det_anno['name'])
    for key in ('score', 'boxes_lidar', 'pred_labels'):
        if len(det_anno[key]) != num_names:
            raise ValueError(
                f'Converted prediction length mismatch at frame {frame_idx}: '
                f'name={num_names}, {key}={len(det_anno[key])}'
            )


def write_counter(counter, title, lines):
    lines.append(title)
    for name, count in sorted(counter.items()):
        lines.append(f'  {name}: {count}')
    if not counter:
        lines.append('  <empty>')


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    info_path = args.info_pkl or default_info_path(args.data_path, args.version)
    require_existing_info(info_path)

    eval_module = load_module(
        'company_nuscenes_eval',
        repo_root / 'pcdet' / 'datasets' / 'company_nuscenes' / 'company_nuscenes_eval.py'
    )
    mapping_module = load_module(
        'company_class_mapping',
        repo_root / 'pcdet' / 'datasets' / 'company_nuscenes' / 'company_class_mapping.py'
    )

    class_names_10 = mapping_module.get_company_10cls_names()
    name_mapping = mapping_module.COMPANY_26_TO_10_CLASS

    det_annos_26 = load_pickle(args.pred_pkl)
    infos_10 = load_pickle(info_path)
    det_annos_10, original_counter, merged_counter = map_prediction_names_to_10cls(
        det_annos_26, name_mapping, class_names_10
    )

    eval_module.validate_prediction_alignment(det_annos_10, infos_10)
    gt_annos, gt_stats = eval_module.build_ground_truth_annos(
        infos_10, class_names_10, min_lidar_points=args.min_lidar_points
    )
    metrics = eval_module.evaluate_company_predictions(
        gt_annos=gt_annos,
        det_annos=det_annos_10,
        class_names=class_names_10,
        distance_thresholds=args.distance_thresholds,
        tp_distance_threshold=args.tp_distance_threshold,
        min_score=args.min_score,
        gt_stats=gt_stats,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / 'result_10cls_from_26cls.pkl'
    metrics_path = args.output_dir / 'company_metrics_summary.json'
    text_path = args.output_dir / 'eval_10cls_from_26cls.txt'

    save_pickle(result_path, det_annos_10)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True, allow_nan=False)

    result_text = eval_module.format_company_results(metrics)
    extra_lines = [
        '---------------26-class predictions merged to 10 classes---------------',
        f'Prediction source: {args.pred_pkl}',
        f'10-class info: {info_path}',
        f'Converted result: {result_path}',
        '',
    ]
    write_counter(original_counter, 'Original 26-class prediction counts:', extra_lines)
    extra_lines.append('')
    write_counter(merged_counter, 'Merged 10-class prediction counts:', extra_lines)
    full_text = result_text.rstrip() + '\n\n' + '\n'.join(extra_lines) + '\n'

    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(full_text, end='')
    print(f'Merged result saved to: {result_path}')
    print(f'Metrics JSON saved to: {metrics_path}')
    print(f'Evaluation text saved to: {text_path}')


if __name__ == '__main__':
    main()
