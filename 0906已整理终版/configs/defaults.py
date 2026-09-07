from yacs.config import CfgNode as CN

_C = CN()


# ====================================
#           Input Info
# ====================================
_C.INPUT = CN()
# Dataset Name
_C.INPUT.DATASET_NAME = "N/A"
# Root for your datasets
_C.INPUT.DATASET_ROOT = "N/A"
# Default shape for each radio map
_C.INPUT.MAP_SHAPE = (128, 128)
# Percentage of trainset data
_C.INPUT.TRAIN_PERCENT = 0.8
# All geographical types in SpectrumNet
_C.INPUT.GEO_TYPES = CN({
    '01': 'Grassland',
    '02': 'Island',
    '03': 'Ocean',
    '04': 'Lake',
    '05': 'Suburban',
    '06': 'DenseUrban',
    '07': 'Rural',
    '08': 'OrdinaryUrban',
    '09': 'Desert',
    '10': 'Mountainous',
    '11': 'Forest',
})
# One method to categorize the terrains
_C.INPUT.GEO_CAT_ONE = CN({
    '01': 'nature',
    '02': 'nature',
    '03': 'water',
    '04': 'water',
    '05': 'urban',
    '06': 'urban',
    '07': 'urban',
    '08': 'urban',
    '09': 'nature',
    '10': 'nature',
    '11': 'nature'
})

# whether to manually build hybrid datasets. If wanna mix, use List(["ones", "to", "be", "mixed"])
_C.INPUT.HYBRID_GEO = ["Grassland", "Island"]
# Specific methods for sampling within a 3D voxel
_C.INPUT.SAMPLE_EACH_HEIGHT_SEPARATELY = False
# Number of sampled points
_C.INPUT.NUM_SAMPLE = 10

# ====================================
#           Condition Info
# ====================================
_C.CONDITION = CN()
# frequency list tailored for SpectrumNet Dataset, 10MHz as unit
_C.CONDITION.FREQUENCIES_LIST = [15, 150, 170, 350, 2200]
# Height list for SpectrumNet Dataset, 10m as unit
_C.CONDITION.HEIGHT_LIST = [0.15, 3, 20]
# Target height
_C.CONDITION.TARGET_HEIGHT = "Hybrid" # ["Hybrid", "0", "1", "2"]
# Target frequency
_C.CONDITION.TARGET_FREQUENCY = "f03" # "All", "f00", "f01", "f02", "f04"


# ====================================
#           Model Info
# ====================================
_C.MODEL = CN()
# Whether to use distributed training
_C.MODEL.DIST_TRAIN = False
# Hidden dimensions of UNet, channel numbers of each UNet layer
_C.MODEL.LAYER_CHANNELS = [128, 128, 256, 256, 512, 512, 1024]
# Whether each layer uses self-attention
_C.MODEL.USE_SA = [False, False, False, False, True, True, True]
# Whether each layer uses double ResBlock
_C.MODEL.DOUBLE_RES = [False, False, False, False, False, False, True]
# DDPM sample steps
_C.MODEL.DIFFUSION_STEPS = 1000
# DDIM backward steps
_C.MODEL.DDIM_STEPS = 10
# Global time embedding dimension
_C.MODEL.GLOBAL_TIME_DIM = 256
# How many modules does each Unet layer contain
_C.MODEL.LAYER_BLOCK_NUM = 2
# Whether the model operates on a specified height
_C.MODEL.SINGLE_HEIGHT = True

# ====================================
#     Solver Info (For Training)
# ====================================
_C.SOLVER = CN()
# Batch size
_C.SOLVER.BATCH_SIZE = 64
# Worker numbers
_C.SOLVER.NUM_WORKERS = 16
# Learning Rate
_C.SOLVER.LR = 5e-7
# Weight Decay
_C.SOLVER.WEIGHT_DECAY = 1e-2
# Epochs
_C.SOLVER.EPOCHS = 60
# Warmup epochs, which is included in total epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# Log period within an epoch
_C.SOLVER.LOG_ITER = 10
# Seeds
_C.SOLVER.SEEDS = 1234

# ====================================
#           Test Info
# ====================================
_C.TEST = CN()
# Batch size for testing
_C.TEST.BATCH_SIZE = 1
# Name of the test dataset
_C.TEST.TEST_DST = "Island"
# Rerank force. Higher means stronger reranking for diffusion
_C.TEST.RERANK_FORCE = 1

# ====================================
#        Physics Model Info
# ====================================
_C.PHYSICS = CN()
_C.PHYSICS.HATA = CN()
# Lower RSS bound threshold corresbonding to INPUT.GEO_CAT_ONE
_C.PHYSICS.HATA.LOWER_BOUND = [-1, -0.388235294117647, -0.3019607843137255] 
# RSS bias when projected between different heights, corresponding to INPUT.GEO_CAT_ONE
_C.PHYSICS.HATA.PROJ_BIAS = CN({
    'urban': [
        [0, 0.02352941176470591, 0.015686274509803866], 
        [-0.02352941176470591, 0, 0.007843137254901933], 
        [-0.015686274509803866, -0.007843137254901933, 0]
    ],
    'water': [
        [0, 0, -0.015686274509803866], 
        [0, 0, -0.015686274509803866], 
        [0.015686274509803866, 0.015686274509803866, 0]
    ],
    'nature': [
        [0, 0.015686274509803977, 0.015686274509803866], 
        [-0.015686274509803977, 0, 0], 
        [-0.015686274509803866, 0, 0]
    ]
})

# ====================================
#           Debug Info
# ====================================
_C.DEBUG = CN()
_C.DEBUG.HEIGHT_IDX = 0
_C.DEBUG.SAMPLE_SEED = 20260905
_C.DEBUG.SINGLE_HEIGHT_BASELINE = False
