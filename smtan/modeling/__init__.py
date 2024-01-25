from .smtan import SMTAN
ARCHITECTURES = {"SMTAN": SMTAN}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)
