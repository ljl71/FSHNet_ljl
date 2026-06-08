import argparse
import copy
import importlib.util
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = REPO_ROOT / 'pcdet' / 'datasets' / 'company_nuscenes' / 'company_class_mapping.py'
spec = importlib.util.spec_from_file_location('company_class_mapping', MAPPING_PATH)
company_class_mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(company_class_mapping)

COMPANY_10_CLASS_NAMES = company_class_mapping.COMPANY_10_CLASS_NAMES
build_company_10cls_mapping = company_class_mapping.build_company_10cls_mapping


DEFAULT_SPLITS = {
    'train': ('company_nuscenes_infos_train.pkl', 'company_nuscenes_10cls_infos_train.pkl'),
    'val': ('company_nuscenes_infos_val.pkl', 'company_nuscenes_10cls_infos_val.pkl'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create 10-class merged CompanyNuScenes info files from existing 26-class info files.'
    )
    parser.add_argument('--data_path', type=Path, default=Path('data/nuscenes'))
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--train_info', type=str, default=DEFAULT_SPLITS['train'][0])
    parser.add_argument('--val_info', type=str, default=DEFAULT_SPLITS['val'][0])
    parser.add_argument('--train_output', type=str, default=DEFAULT_SPLITS['train'][1])
    parser.add_argument('--val_output', type=str, default=DEFAULT_SPLITS['val'][1])
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


def print_split_summary(split_name, input_path, output_path, counter, unmapped):
    print(f'[{split_name}] source: {input_path}')
    print(f'[{split_name}] output: {output_path}')
    print(f'[{split_name}] merged GT counts:')
    for class_name in COMPANY_10_CLASS_NAMES:
        print(f'  {class_name:18s}: {counter[class_name]}')
    if unmapped:
        print(f'[{split_name}] unmapped original classes:')
        for class_name, count in sorted(unmapped.items()):
            print(f'  {class_name:38s}: {count}')
    else:
        print(f'[{split_name}] all original classes were mapped successfully.')


def process_split(split_name, input_path, output_path, class_mapping):
    infos = load_infos(input_path)
    merged_infos, counter, unmapped = merge_info_names(infos, class_mapping)
    save_infos(output_path, merged_infos)
    print_split_summary(split_name, input_path, output_path, counter, unmapped)
    return counter, unmapped


def main():
    args = parse_args()
    data_path = args.data_path
    root = data_path / args.version
    class_mapping = build_company_10cls_mapping()

    splits = {
        'train': (args.train_info, args.train_output),
        'val': (args.val_info, args.val_output),
    }
    total_counter = Counter()
    total_unmapped = Counter()

    for split_name, (input_name, output_name) in splits.items():
        counter, unmapped = process_split(
            split_name=split_name,
            input_path=root / input_name,
            output_path=root / output_name,
            class_mapping=class_mapping,
        )
        total_counter.update(counter)
        total_unmapped.update(unmapped)

    print('[total] merged GT counts:')
    for class_name in COMPANY_10_CLASS_NAMES:
        print(f'  {class_name:18s}: {total_counter[class_name]}')
    if total_unmapped:
        print('[total] unmapped classes remain:')
        for class_name, count in sorted(total_unmapped.items()):
            print(f'  {class_name:38s}: {count}')
    else:
        print('[total] all original 26 classes were mapped to the 10 merged classes.')


if __name__ == '__main__':
    main()
