import argparse
import copy
import importlib.util
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = REPO_ROOT / 'pcdet' / 'datasets' / 'company_nuscenes' / 'company_class_mapping.py'
spec = importlib.util.spec_from_file_location('company_class_mapping', MAPPING_PATH)
company_class_mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(company_class_mapping)

COMPANY_10_CLASS_NAMES = company_class_mapping.COMPANY_10_CLASS_NAMES
build_company_10cls_mapping = company_class_mapping.build_company_10cls_mapping


DEFAULT_INPUTS = {
    'train': 'company_nuscenes_infos_train.pkl',
    'val': 'company_nuscenes_infos_val.pkl',
}
DEFAULT_OUTPUTS = {
    'train': 'company_nuscenes_10cls_infos_train.pkl',
    'val': 'company_nuscenes_10cls_infos_val.pkl',
    'test': 'company_nuscenes_10cls_infos_test.pkl',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create 10-class merged CompanyNuScenes info files from existing 26-class info files.'
    )
    parser.add_argument('--data_path', type=Path, default=Path('data/nuscenes'))
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument(
        '--split_mode', choices=['trainval', 'trainvaltest'], default='trainval',
        help='trainval maps existing train/val infos; trainvaltest repartitions all source infos by scene_token.'
    )
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--train_info', type=str, default=DEFAULT_INPUTS['train'])
    parser.add_argument('--val_info', type=str, default=DEFAULT_INPUTS['val'])
    parser.add_argument('--train_output', type=str, default=DEFAULT_OUTPUTS['train'])
    parser.add_argument('--val_output', type=str, default=DEFAULT_OUTPUTS['val'])
    parser.add_argument('--test_output', type=str, default=DEFAULT_OUTPUTS['test'])
    return parser.parse_args()


def load_infos(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_infos(path, infos):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(infos, f)


def merge_info_names(infos, class_mapping):
    merged_infos = []
    counter = Counter()
    unmapped = Counter()

    for info in infos:
        merged_info = copy.deepcopy(info)
        names = np.asarray(merged_info.get('gt_names', []))
        mapped_names = []

        for name in names:
            name = str(name)
            mapped_name = class_mapping.get(name)
            if mapped_name is None:
                unmapped[name] += 1
                mapped_name = name
            else:
                counter[mapped_name] += 1
            mapped_names.append(mapped_name)

        merged_info['gt_names'] = np.asarray(mapped_names)
        merged_infos.append(merged_info)

    return merged_infos, counter, unmapped


def print_split_summary(split_name, source_label, output_path, infos, counter, unmapped):
    scene_tokens = sorted({info.get('scene_token') for info in infos})
    print(f'[{split_name}] source: {source_label}')
    print(f'[{split_name}] output: {output_path}')
    print(f'[{split_name}] scenes: {len(scene_tokens)}')
    print(f'[{split_name}] samples: {len(infos)}')
    print(f'[{split_name}] merged GT counts:')
    for class_name in COMPANY_10_CLASS_NAMES:
        print(f'  {class_name:18s}: {counter[class_name]}')
    if unmapped:
        print(f'[{split_name}] unmapped original classes:')
        for class_name, count in sorted(unmapped.items()):
            print(f'  {class_name:38s}: {count}')
    else:
        print(f'[{split_name}] all original classes were mapped successfully.')


def process_split(split_name, source_label, infos, output_path, class_mapping):
    merged_infos, counter, unmapped = merge_info_names(infos, class_mapping)
    save_infos(output_path, merged_infos)
    print_split_summary(split_name, source_label, output_path, merged_infos, counter, unmapped)
    return counter, unmapped


def validate_ratios(train_ratio, val_ratio, test_ratio):
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError('Split ratios must be non-negative.')
    total = sum(ratios)
    if total <= 0:
        raise ValueError('At least one split ratio must be positive.')
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f'Split ratios must sum to 1.0, got train+val+test={total:.6f}'
        )


def allocate_counts(num_items, ratios):
    raw_counts = [num_items * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remainder = num_items - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda idx: (raw_counts[idx] - counts[idx], ratios[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1

    positive = [idx for idx, ratio in enumerate(ratios) if ratio > 0]
    if num_items >= len(positive):
        for idx in positive:
            if counts[idx] == 0:
                donor = max(
                    (j for j in range(len(counts)) if counts[j] > 1),
                    key=lambda j: counts[j],
                )
                counts[donor] -= 1
                counts[idx] += 1
    return counts


def split_infos_by_scene(infos, train_ratio, val_ratio, test_ratio, split_seed):
    validate_ratios(train_ratio, val_ratio, test_ratio)

    by_scene = defaultdict(list)
    for info in infos:
        scene_token = info.get('scene_token')
        if scene_token is None:
            raise KeyError('Every info entry must contain scene_token for scene-level splitting.')
        by_scene[scene_token].append(info)

    scene_tokens = sorted(by_scene)
    rng = random.Random(split_seed)
    rng.shuffle(scene_tokens)
    train_count, val_count, test_count = allocate_counts(
        len(scene_tokens), [train_ratio, val_ratio, test_ratio]
    )

    train_scenes = set(scene_tokens[:train_count])
    val_scenes = set(scene_tokens[train_count:train_count + val_count])
    test_scenes = set(scene_tokens[train_count + val_count:train_count + val_count + test_count])

    if train_scenes & val_scenes or train_scenes & test_scenes or val_scenes & test_scenes:
        raise AssertionError('Scene-level split overlap detected.')
    if len(train_scenes | val_scenes | test_scenes) != len(scene_tokens):
        raise AssertionError('Scene-level split did not allocate every scene.')

    split_scenes = {
        'train': train_scenes,
        'val': val_scenes,
        'test': test_scenes,
    }
    split_infos = {
        split_name: [
            info for scene_token in scene_tokens if scene_token in scenes
            for info in by_scene[scene_token]
        ]
        for split_name, scenes in split_scenes.items()
    }
    return split_infos, split_scenes


def load_source_infos(root, args):
    train_path = root / args.train_info
    val_path = root / args.val_info
    train_infos = load_infos(train_path)
    val_infos = load_infos(val_path)
    return train_infos, val_infos, train_path, val_path


def run_trainval(root, args, class_mapping):
    train_infos, val_infos, train_path, val_path = load_source_infos(root, args)
    outputs = {
        'train': root / args.train_output,
        'val': root / args.val_output,
    }
    sources = {
        'train': str(train_path),
        'val': str(val_path),
    }
    return process_all_splits(
        split_infos={'train': train_infos, 'val': val_infos},
        source_labels=sources,
        output_paths=outputs,
        class_mapping=class_mapping,
    )


def run_trainvaltest(root, args, class_mapping):
    train_infos, val_infos, train_path, val_path = load_source_infos(root, args)
    all_infos = train_infos + val_infos
    split_infos, split_scenes = split_infos_by_scene(
        all_infos,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )
    print(
        '[scene split] mode=trainvaltest '
        f'ratios={args.train_ratio:.3f}/{args.val_ratio:.3f}/{args.test_ratio:.3f} '
        f'seed={args.split_seed}'
    )
    for split_name in ('train', 'val', 'test'):
        print(f'[scene split] {split_name}_scenes: {len(split_scenes[split_name])}')

    source_label = f'{train_path} + {val_path}'
    outputs = {
        'train': root / args.train_output,
        'val': root / args.val_output,
        'test': root / args.test_output,
    }
    sources = {split_name: source_label for split_name in outputs}
    return process_all_splits(
        split_infos=split_infos,
        source_labels=sources,
        output_paths=outputs,
        class_mapping=class_mapping,
    )


def process_all_splits(split_infos, source_labels, output_paths, class_mapping):
    total_counter = Counter()
    total_unmapped = Counter()

    for split_name in split_infos:
        counter, unmapped = process_split(
            split_name=split_name,
            source_label=source_labels[split_name],
            infos=split_infos[split_name],
            output_path=output_paths[split_name],
            class_mapping=class_mapping,
        )
        total_counter.update(counter)
        total_unmapped.update(unmapped)
    return total_counter, total_unmapped


def print_total_summary(total_counter, total_unmapped):
    print('[total] merged GT counts:')
    for class_name in COMPANY_10_CLASS_NAMES:
        print(f'  {class_name:18s}: {total_counter[class_name]}')
    if total_unmapped:
        print('[total] unmapped classes remain:')
        for class_name, count in sorted(total_unmapped.items()):
            print(f'  {class_name:38s}: {count}')
    else:
        print('[total] all original 26 classes were mapped to the 10 merged classes.')


def main():
    args = parse_args()
    root = args.data_path / args.version
    class_mapping = build_company_10cls_mapping()

    if args.split_mode == 'trainval':
        total_counter, total_unmapped = run_trainval(root, args, class_mapping)
    elif args.split_mode == 'trainvaltest':
        total_counter, total_unmapped = run_trainvaltest(root, args, class_mapping)
    else:
        raise NotImplementedError(args.split_mode)

    print_total_summary(total_counter, total_unmapped)


if __name__ == '__main__':
    main()
