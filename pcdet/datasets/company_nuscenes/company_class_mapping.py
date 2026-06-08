"""
Class mappings for CompanyNuScenes semantic-merging experiments.

Keep this file in explicit multi-line Python format so it remains readable in
GitHub raw view and can be checked directly with python -m py_compile.
"""

COMPANY_26_CLASS_NAMES = [
    'human_pedestrian_adult',
    'human_pedestrian_child',
    'human_pedestrian_wheelchair',
    'human_pedestrian_stroller',
    'human_pedestrian_personal_mobility',
    'vehicle_car',
    'vehicle_bus_bendy',
    'vehicle_bus_rigid',
    'vehicle_truck',
    'vehicle_construction',
    'vehicle_emergency_ambulance',
    'vehicle_emergency_police',
    'vehicle_trailer',
    'movable_object_barrier',
    'movable_object_trafficcone',
    'movable_object_pushable_pullable',
    'movable_object_debris',
    'vehicle_emergency_other',
    'vehicle_motorcycle',
    'vehicle_bicycle',
    'group_human_pedestrian',
    'group_vehicle_bicycle',
    'other',
    'animal',
    'vehicle_tricycle',
    'bicycle',
]

COMPANY_10CLS_MAPPING = {
    'pedestrian': [
        'human_pedestrian_adult',
        'human_pedestrian_child',
        'human_pedestrian_wheelchair',
        'human_pedestrian_stroller',
        'human_pedestrian_personal_mobility',
        'group_human_pedestrian',
    ],
    'car': [
        'vehicle_car',
    ],
    'bus': [
        'vehicle_bus_bendy',
        'vehicle_bus_rigid',
    ],
    'truck': [
        'vehicle_truck',
        'vehicle_construction',
        'vehicle_trailer',
    ],
    'emergency_vehicle': [
        'vehicle_emergency_ambulance',
        'vehicle_emergency_police',
        'vehicle_emergency_other',
    ],
    'two_wheeler': [
        'vehicle_motorcycle',
        'vehicle_bicycle',
        'group_vehicle_bicycle',
        'vehicle_tricycle',
        'bicycle',
    ],
    'barrier': [
        'movable_object_barrier',
    ],
    'traffic_cone': [
        'movable_object_trafficcone',
    ],
    'movable_object': [
        'movable_object_pushable_pullable',
        'movable_object_debris',
    ],
    'other': [
        'animal',
        'other',
    ],
}

COMPANY_10_CLASS_NAMES = list(COMPANY_10CLS_MAPPING.keys())


def get_company_10cls_names():
    return list(COMPANY_10_CLASS_NAMES)


def build_company_10cls_mapping():
    mapping = {}
    for merged_name, original_names in COMPANY_10CLS_MAPPING.items():
        for original_name in original_names:
            if original_name in mapping:
                raise ValueError(f'Duplicate CompanyNuScenes class mapping for {original_name}')
            mapping[original_name] = merged_name

    missing = sorted(set(COMPANY_26_CLASS_NAMES) - set(mapping))
    extra = sorted(set(mapping) - set(COMPANY_26_CLASS_NAMES))
    if missing or extra:
        raise ValueError(
            f'CompanyNuScenes 10-class mapping is incomplete: missing={missing}, extra={extra}'
        )
    return mapping


COMPANY_26_TO_10_CLASS = build_company_10cls_mapping()


def map_company_26cls_to_10cls(name):
    return COMPANY_26_TO_10_CLASS[name]
