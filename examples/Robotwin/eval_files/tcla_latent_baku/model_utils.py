import torch
from omegaconf import OmegaConf

from tcla_latent_act.model_utils import (  # noqa: F401
    build_traj_a_freq_from_executed,
    encode_task_name_minilm,
    get_tcla_root,
    infer_phase_scalar,
    load_episode_task_emb,
    load_json,
    load_training_config,
    normalize_action,
    normalize_qpos,
    predict_action_chunk,
    resolve_framework_paths,
    resolve_path,
    save_phase_curve_plot,
    stack_images,
    task_name_embedding,
    unnormalize_action,
)

from tcla.model.framework import act_policy  # noqa: F401,E402
from tcla.model.framework import action_autoencoder_version3  # noqa: F401,E402
from tcla.model.framework import baku_policy  # noqa: F401,E402
from tcla.model.modules.act.policy import ACTPolicyModule  # noqa: E402
from tcla.model.tools import FRAMEWORK_REGISTRY  # noqa: E402


def build_framework(config_yaml, ckpt_path, device: str):
    tcla_root = get_tcla_root()
    cfg = load_training_config(config_yaml)
    ckpt_path = resolve_path(ckpt_path, base_dir=tcla_root)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model_cfg = OmegaConf.to_container(cfg.framework, resolve=True)
    framework_name = model_cfg.pop("name")
    model_cfg["device"] = str(device)
    model_cfg = resolve_framework_paths(model_cfg, base_dir=tcla_root)

    model = FRAMEWORK_REGISTRY[framework_name](**model_cfg)
    state_dict = ACTPolicyModule.extract_framework_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model, cfg
