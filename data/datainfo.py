# datainfo.py
# Auto-generated / aligned with: 分类后的数据集.xlsx
# L0: Acquisition Platform
# L1: Object
# L2: Part
# L3: Material
# L4: Task (Leaf)

# ========================
# L0 (Acquisition Platform)
# ========================
#from __future__ import annotations
modal_dict = {
    'VehicleProfiler': 'VehicleProfiler',
    'Handheld': 'Handheld',
    'VehicleDownward': 'VehicleDownward',
    'VehicleForward': 'VehicleForward',
    'FixedIndustrial': 'FixedIndustrial',
    'IndustrialLineScan': 'IndustrialLineScan',
}

modal_map = {
    'VehicleProfiler': 0,
    'Handheld': 1,
    'VehicleDownward': 2,
    'VehicleForward': 3,
    'FixedIndustrial': 4,
    'IndustrialLineScan': 5,
}

# ========================
# L4 (Task / Leaf)
# ========================
task_list = [
    'PavementCrack',
    'PavementPothole',
    'JointSealing',
    'PavementPatch',
    'RoadMarking',
    'ExpansionJoint',
    'ManholeCover',
    'ConcreteCrack',
    'ConcreteRoadCrack',
    'ConcretePavementPatch',
    'ConcretePavementPothole',
    'ConstructionJoint',
    'Efflorescence',
    'Rust',
    'HollowArea',
    'Spalling',
    'Weathering',
    'TunnelCrack',
    'WaterLeakage',
    'RailCrack',
    'RailPothole',
    'WallCrack',
    'MasonryCrack',
    'SteelCorrosion',
    'ExposedRebar',
    'DamagedBuilding',
    'Debris',
    'UndamagedRoad',
    'UndamagedBuilding',
]

task_idx = {task: idx for idx, task in enumerate(task_list)}

# ========================
# L1 (Object)
# ========================
route_level_1_dict = {
    'Road': [
        'PavementCrack', 'PavementPothole', 'JointSealing', 'PavementPatch', 'RoadMarking',
        'ExpansionJoint', 'ManholeCover', 'ConcreteRoadCrack', 'ConcretePavementPatch',
        'ConcretePavementPothole', 'ConstructionJoint'
    ],
    'Bridge': [
        'ConcreteCrack', 'Efflorescence', 'Rust', 'HollowArea', 'Spalling', 'Weathering', 'SteelCorrosion'
    ],
    'Tunnel': ['TunnelCrack', 'WaterLeakage', 'Efflorescence'],
    'Railway': ['RailCrack', 'RailPothole'],
    'Building': ['WallCrack', 'MasonryCrack', 'ExposedRebar'],
    'Infrastructure': ['DamagedBuilding', 'Debris', 'UndamagedRoad', 'UndamagedBuilding'],
}
route_level_1_map = {k: i for i, k in enumerate(route_level_1_dict.keys())}

# ========================
# L2 (Part)
# ========================
route_level_2_dict = {
    'Surface': [
        'PavementCrack', 'PavementPothole', 'JointSealing', 'PavementPatch', 'RoadMarking',
        'ExpansionJoint', 'ManholeCover', 'ConcreteRoadCrack', 'ConcretePavementPatch',
        'ConcretePavementPothole', 'ConstructionJoint', 'RailCrack', 'RailPothole'
    ],
    'Structural': [
        'ConcreteCrack', 'Efflorescence', 'Rust', 'HollowArea', 'Spalling', 'Weathering',
        'MasonryCrack', 'SteelCorrosion', 'ExposedRebar',
        'DamagedBuilding', 'Debris', 'UndamagedRoad', 'UndamagedBuilding'
    ],
    'Lining': ['TunnelCrack', 'WaterLeakage', 'Efflorescence'],
    'Wall': ['WallCrack'],
}
route_level_2_map = {k: i for i, k in enumerate(route_level_2_dict.keys())}

# ========================
# L3 (Material)
# ========================
route_level_3_dict = {
    'Asphalt': ['PavementCrack', 'PavementPothole', 'JointSealing', 'PavementPatch', 'ExpansionJoint'],
    'Marking': ['RoadMarking'],
    'Metal': ['ManholeCover', 'Rust', 'RailCrack', 'RailPothole'],
    'Concrete': [
        'ConcreteCrack', 'ConcreteRoadCrack', 'ConcretePavementPatch', 'ConcretePavementPothole',
        'ConstructionJoint', 'Efflorescence', 'HollowArea', 'Spalling', 'Weathering',
        'TunnelCrack', 'WaterLeakage', 'WallCrack', 'ExposedRebar'
    ],
    'Mixed': ['WallCrack', 'DamagedBuilding', 'Debris', 'UndamagedRoad', 'UndamagedBuilding'],
    'Masonry': ['MasonryCrack'],
    'Steel': ['SteelCorrosion'],
}
route_level_3_map = {k: i for i, k in enumerate(route_level_3_dict.keys())}

# =======================================================
# Task folder -> (L0, L1, L2, L3, L4)  精确表（强烈推荐你的任务目录命名为 L0_L4）
# =======================================================
TASK_FOLDER_META = {
    'VehicleProfiler_PavementCrack': ('VehicleProfiler', 'Road', 'Surface', 'Asphalt', 'PavementCrack'),
    'VehicleProfiler_PavementPothole': ('VehicleProfiler', 'Road', 'Surface', 'Asphalt', 'PavementPothole'),
    'VehicleProfiler_JointSealing': ('VehicleProfiler', 'Road', 'Surface', 'Asphalt', 'JointSealing'),
    'VehicleProfiler_PavementPatch': ('VehicleProfiler', 'Road', 'Surface', 'Asphalt', 'PavementPatch'),
    'VehicleProfiler_RoadMarking': ('VehicleProfiler', 'Road', 'Surface', 'Marking', 'RoadMarking'),
    'VehicleProfiler_ExpansionJoint': ('VehicleProfiler', 'Road', 'Surface', 'Asphalt', 'ExpansionJoint'),
    'VehicleProfiler_ManholeCover': ('VehicleProfiler', 'Road', 'Surface', 'Metal', 'ManholeCover'),
    'Handheld_PavementCrack': ('Handheld', 'Road', 'Surface', 'Asphalt', 'PavementCrack'),
    'VehicleDownward_PavementCrack': ('VehicleDownward', 'Road', 'Surface', 'Asphalt', 'PavementCrack'),
    'Handheld_ConcreteCrack': ('Handheld', 'Bridge', 'Structural', 'Concrete', 'ConcreteCrack'),
    'Handheld_PavementPothole': ('Handheld', 'Road', 'Surface', 'Asphalt', 'PavementPothole'),
    'VehicleForward_PavementCrack': ('VehicleForward', 'Road', 'Surface', 'Asphalt', 'PavementCrack'),
    'VehicleForward_PavementPothole': ('VehicleForward', 'Road', 'Surface', 'Asphalt', 'PavementPothole'),
    'VehicleDownward_PavementPatch': ('VehicleDownward', 'Road', 'Surface', 'Asphalt', 'PavementPatch'),
    'VehicleDownward_RoadMarking': ('VehicleDownward', 'Road', 'Surface', 'Marking', 'RoadMarking'),
    'VehicleProfiler_ConcreteRoadCrack': ('VehicleProfiler', 'Road', 'Surface', 'Concrete', 'ConcreteRoadCrack'),
    'VehicleProfiler_ConcretePavementPatch': ('VehicleProfiler', 'Road', 'Surface', 'Concrete', 'ConcretePavementPatch'),
    'VehicleProfiler_ConcretePavementPothole': ('VehicleProfiler', 'Road', 'Surface', 'Concrete', 'ConcretePavementPothole'),
    'VehicleProfiler_ConstructionJoint': ('VehicleProfiler', 'Road', 'Surface', 'Concrete', 'ConstructionJoint'),
    'Handheld_Efflorescence': ('Handheld', 'Bridge', 'Structural', 'Concrete', 'Efflorescence'),
    'Handheld_Rust': ('Handheld', 'Bridge', 'Structural', 'Metal', 'Rust'),
    'Handheld_HollowArea': ('Handheld', 'Bridge', 'Structural', 'Concrete', 'HollowArea'),
    'Handheld_Spalling': ('Handheld', 'Bridge', 'Structural', 'Concrete', 'Spalling'),
    'Handheld_Weathering': ('Handheld', 'Bridge', 'Structural', 'Concrete', 'Weathering'),
    'FixedIndustrial_TunnelCrack': ('FixedIndustrial', 'Tunnel', 'Lining', 'Concrete', 'TunnelCrack'),
    'FixedIndustrial_WaterLeakage': ('FixedIndustrial', 'Tunnel', 'Lining', 'Concrete', 'WaterLeakage'),
    'FixedIndustrial_Efflorescence': ('FixedIndustrial', 'Tunnel', 'Lining', 'Concrete', 'Efflorescence'),
    'IndustrialLineScan_RailCrack': ('IndustrialLineScan', 'Railway', 'Surface', 'Metal', 'RailCrack'),
    'IndustrialLineScan_RailPothole': ('IndustrialLineScan', 'Railway', 'Surface', 'Metal', 'RailPothole'),
    'Handheld_WallCrack': ('Handheld', 'Building', 'Wall', 'Mixed', 'WallCrack'),
    'Handheld_TunnelCrack': ('Handheld', 'Tunnel', 'Lining', 'Concrete', 'TunnelCrack'),
    'Handheld_MasonryCrack': ('Handheld', 'Building', 'Structural', 'Masonry', 'MasonryCrack'),
    'Handheld_SteelCorrosion': ('Handheld', 'Bridge', 'Structural', 'Steel', 'SteelCorrosion'),
    'Handheld_ExposedRebar': ('Handheld', 'Building', 'Structural', 'Concrete', 'ExposedRebar'),
    'Handheld_DamagedBuilding': ('Handheld', 'Infrastructure', 'Structural', 'Mixed', 'DamagedBuilding'),
    'Handheld_Debris': ('Handheld', 'Infrastructure', 'Structural', 'Mixed', 'Debris'),
    'Handheld_UndamagedRoad': ('Handheld', 'Infrastructure', 'Structural', 'Mixed', 'UndamagedRoad'),
    'Handheld_UndamagedBuilding': ('Handheld', 'Infrastructure', 'Structural', 'Mixed', 'UndamagedBuilding'),
}

# =======================================================
# Routing index aliases
# =======================================================
_modal_self_map = {v: v for v in modal_map.values()}
_l1_self_map = {v: v for v in route_level_1_map.values()}
_l2_self_map = {v: v for v in route_level_2_map.values()}
_l3_self_map = {v: v for v in route_level_3_map.values()}

modal_map_idx = _modal_self_map
route_map_idx = _l1_self_map
route_level_1_map_idx = _l1_self_map
route_level1_map_idx = _l1_self_map
route_level_2_map_idx = _l2_self_map
route_level2_map_idx = _l2_self_map
route_level_3_map_idx = _l3_self_map
route_level3_map_idx = _l3_self_map
