from omegaconf import DictConfig
from .gnn import N2LidarGCN, N4LidarGCN

def build_model(model_cfg: DictConfig):
    name = model_cfg.name
    if name == 'N2LidarGCN':
        model = N2LidarGCN(
            input_dim=model_cfg.input_dim,
            hidden_dim=model_cfg.hidden_dim,
            output_dim=model_cfg.output_dim,
            pool_method=model_cfg.pool_method,
        )
    elif name == 'N4LidarGCN':
        model = N4LidarGCN(
            input_dim=model_cfg.input_dim,
            hidden_dim=model_cfg.hidden_dim,
            output_dim=model_cfg.output_dim,
            pool_method=model_cfg.pool_method,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

    return model