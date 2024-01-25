import os

from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "SMTAN"

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.NUM_PRE_CLIPS = 256

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.NAME = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL.SMTAN = CN()
_C.MODEL.SMTAN.NUM_CLIPS = 128
_C.MODEL.SMTAN.JOINT_SPACE_SIZE = 256
_C.MODEL.SMTAN.RESIDUAL = 1.0

_C.MODEL.SMTAN.FEATPOOL = CN()
_C.MODEL.SMTAN.FEATPOOL.INPUT_SIZE = 1024
_C.MODEL.SMTAN.FEATPOOL.HIDDEN_SIZE = 512
_C.MODEL.SMTAN.FEATPOOL.KERNEL_SIZE = 2

_C.MODEL.SMTAN.FEAT2D = CN()
_C.MODEL.SMTAN.FEAT2D.NAME = "pool"
_C.MODEL.SMTAN.FEAT2D.POOLING_COUNTS = [15, 8, 8, 8]

_C.MODEL.SMTAN.TEXT_ENCODER = CN()
_C.MODEL.SMTAN.TEXT_ENCODER.NAME = 'BERT'
_C.MODEL.SMTAN.TEXT_ENCODER.USE_PHRASE = False
_C.MODEL.SMTAN.TEXT_ENCODER.DROP_PHRASE = False

_C.MODEL.SMTAN.PREDICTOR = CN() 
_C.MODEL.SMTAN.PREDICTOR.HIDDEN_SIZE = 512
_C.MODEL.SMTAN.PREDICTOR.KERNEL_SIZE = 5
_C.MODEL.SMTAN.PREDICTOR.NUM_STACK_LAYERS = 8

_C.MODEL.SMTAN.LOSS = CN()
_C.MODEL.SMTAN.LOSS.MIN_IOU = 0.3
_C.MODEL.SMTAN.LOSS.MAX_IOU = 0.7
_C.MODEL.SMTAN.LOSS.BCE_WEIGHT = 1
_C.MODEL.SMTAN.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL = 1
_C.MODEL.SMTAN.LOSS.NEGATIVE_VIDEO_IOU = 0.5
_C.MODEL.SMTAN.LOSS.SENT_REMOVAL_IOU = 0.5
_C.MODEL.SMTAN.LOSS.PAIRWISE_SENT_WEIGHT = 0.0
_C.MODEL.SMTAN.LOSS.CONTRASTIVE_WEIGHT = 0.05
_C.MODEL.SMTAN.LOSS.TAU_VIDEO = 0.2
_C.MODEL.SMTAN.LOSS.TAU_SENT = 0.2
_C.MODEL.SMTAN.LOSS.MARGIN = 0.2
_C.MODEL.SMTAN.LOSS.SENT_NEG_IOU = 0.2
_C.MODEL.SMTAN.LOSS.NEGATIVE_PROP_IOU = 0.2
_C.MODEL.SMTAN.LOSS.PHRASE_ONLY = False
_C.MODEL.SMTAN.LOSS.USE_SCORE_MAP_LOSS = False
_C.MODEL.SMTAN.LOSS.USE_FOCAL_LOSS = False
_C.MODEL.SMTAN.LOSS.CONTRASTIVE = False
_C.MODEL.SMTAN.LOSS.CONSIS_WEIGHT = 1.0
_C.MODEL.SMTAN.LOSS.EXC_WEIGHT = 1.0
_C.MODEL.SMTAN.LOSS.THRESH = 0.001

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 12
_C.SOLVER.LR = 0.01
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.TEST_PERIOD = 1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.MILESTONES = (8, 11)
_C.SOLVER.RESUME = False
_C.SOLVER.RESUME_EPOCH = 1
_C.SOLVER.FREEZE_BERT = 4
_C.SOLVER.ONLY_IOU = 7
_C.SOLVER.SKIP_TEST = 0

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.NMS_THRESH = 0.5
_C.TEST.CONTRASTIVE_SCORE_POW = 0.5
 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")