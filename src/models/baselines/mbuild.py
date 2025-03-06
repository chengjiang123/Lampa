from ..model_utils.mamba_utils import Registry



MODELS = Registry('models')

def build_model_from_cfg(cfg, **kwargs):
    return MODELS.build(cfg, **kwargs)