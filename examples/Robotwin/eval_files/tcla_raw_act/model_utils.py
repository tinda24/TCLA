import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import OmegaConf


def get_tcla_root() -> Path:
    env_root = os.environ.get("TCLA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[4]


TCLA_ROOT = get_tcla_root()
if str(TCLA_ROOT) not in sys.path:
    sys.path.insert(0, str(TCLA_ROOT))

from tcla.model.framework import act_policy  # noqa: F401
from tcla.model.tools import FRAMEWORK_REGISTRY


def resolve_path(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    base_dir = base_dir or get_tcla_root()
    return (base_dir / path).resolve()


def load_json(path: Union[str, Path]) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def build_framework(
    config_yaml: Union[str, Path], ckpt_path: Union[str, Path], device: str
):
    config_path = resolve_path(config_yaml)
    cfg = OmegaConf.load(config_path)
    model_cfg = OmegaConf.to_container(cfg.framework, resolve=True)
    framework_name = model_cfg.pop("name")
    model_cfg["device"] = str(device)
    model = FRAMEWORK_REGISTRY[framework_name](**model_cfg)
    checkpoint = torch.load(resolve_path(ckpt_path), map_location="cpu")
    if checkpoint.get("format") == "framework_state_dict_v1":
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    elif (
        isinstance(checkpoint.get("model"), dict)
        and checkpoint["model"].get("format") == "framework_state_dict_v1"
    ):
        model.load_state_dict(checkpoint["model"]["state_dict"], strict=True)
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected framework_state_dict_v1 at top level or under 'model'."
        )
    model.eval()
    model.to(device)
    return model, cfg


def normalize_qpos(qpos: np.ndarray, policy_stats: dict) -> np.ndarray:
    mean = np.asarray(policy_stats["qpos_mean"], dtype=np.float32)
    std = np.asarray(policy_stats["qpos_std"], dtype=np.float32)
    return (np.asarray(qpos, dtype=np.float32) - mean) / std


def unnormalize_action(action: np.ndarray, action_stats: dict) -> np.ndarray:
    action_min = np.asarray(action_stats["action_min"], dtype=np.float32)
    action_max = np.asarray(action_stats["action_max"], dtype=np.float32)
    action = np.clip(action, -1.0, 1.0)
    return 0.5 * (action + 1.0) * (action_max - action_min + 1e-5) + action_min


def stack_images(
    camera_names: Sequence[str],
    *,
    head_cam: Optional[np.ndarray] = None,
    left_cam: Optional[np.ndarray] = None,
    right_cam: Optional[np.ndarray] = None,
) -> torch.Tensor:
    camera_map = {
        "cam_high": head_cam,
        "cam_left_wrist": left_cam,
        "cam_right_wrist": right_cam,
    }
    images = []
    for name in camera_names:
        if name not in camera_map:
            raise KeyError(f"Unsupported camera name: {name}")
        image = camera_map[name]
        if image is None:
            raise ValueError(
                f"Camera '{name}' was requested but no image was provided."
            )
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if image.ndim == 3 and image.shape[-1] in (3, 4):
            image = np.moveaxis(image[..., :3], -1, 0)
        images.append(image)
    stacked = np.stack(images, axis=0).astype(np.float32)
    return torch.from_numpy(stacked)


@torch.no_grad()
def predict_action_chunk(
    model,
    qpos: torch.Tensor,
    image: torch.Tensor,
) -> np.ndarray:
    policy = model.policy
    qpos = qpos.unsqueeze(0).to(model.device)
    image = image.unsqueeze(0).to(model.device)

    image = policy.normalize(image)
    pred_seq, _, _ = policy.model(qpos, image, None)
    if policy.action_space == "raw":
        return pred_seq[0].detach().cpu().numpy()
    raise ValueError(
        f"tcla_raw_act expects action_space=raw, got {policy.action_space!r}. "
        "Use tcla_latent_act for latent policies."
    )
