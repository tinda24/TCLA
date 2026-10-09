import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

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
from tcla.model.framework import action_autoencoder  # noqa: F401
from tcla.model.modules.act.policy import ACTPolicyModule
from tcla.model.tools import FRAMEWORK_REGISTRY
from tcla.configuration import load_paths


def resolve_path(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    base_dir = base_dir or get_tcla_root()
    return (base_dir / path).resolve()


def resolve_framework_paths(model_cfg: dict, base_dir: Optional[Path] = None) -> dict:
    """Resolve checkpoint/config paths relative to TCLA root (not RMBench cwd)."""
    base_dir = base_dir or get_tcla_root()
    for key in ("action_ae_config", "action_ae_ckpt"):
        if model_cfg.get(key):
            model_cfg[key] = str(resolve_path(model_cfg[key], base_dir))
    return model_cfg


def load_json(path: Union[str, Path]) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def task_name_embedding(task_name: str, dim: int = 384) -> np.ndarray:
    """Legacy hash embedding; prefer encode_task_name_minilm for RoboTwin eval."""
    digest = hashlib.sha256(task_name.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(dim).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-6)


_MINILM_ENCODER = None
_TASK_EMB_CACHE: Dict[str, np.ndarray] = {}
_TASK_EMB_CACHE_LOADED_PATH: Optional[str] = None


def _task_emb_source() -> str:
    return (os.environ.get("ROBOTWIN_TASK_EMB_SOURCE") or "cache").strip().lower()


def _default_task_emb_cache_path() -> Path:
    return Path(load_paths().paths.task_embedding_cache)


def load_task_emb_from_cache(task_name: str, *, dim: int = 384) -> np.ndarray:
    """Load the configured task embedding cache."""
    global _TASK_EMB_CACHE_LOADED_PATH

    cache_path = _default_task_emb_cache_path()
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"ROBOTWIN task_emb cache not found: {cache_path} "
            "(set ROBOTWIN_TASK_EMB_CACHE or ROBOTWIN_TASK_EMB_SOURCE=minilm)"
        )

    cache_key = str(cache_path.resolve())
    if _TASK_EMB_CACHE_LOADED_PATH != cache_key:
        import h5py

        _TASK_EMB_CACHE.clear()
        with h5py.File(cache_path, "r") as handle:
            if "task_embeddings" in handle:
                for name in handle["task_embeddings"]:
                    emb = handle[f"task_embeddings/{name}"][:].astype(np.float32)
                    if emb.ndim > 1:
                        emb = emb.reshape(-1)
                    _TASK_EMB_CACHE[str(name)] = emb
            elif "task_emb_matrix" in handle and "task_names" in handle:
                names = handle["task_names"][:]
                matrix = handle["task_emb_matrix"][:].astype(np.float32)
                for idx, raw_name in enumerate(names):
                    name = (
                        raw_name.decode("utf-8")
                        if isinstance(raw_name, bytes)
                        else str(raw_name)
                    )
                    _TASK_EMB_CACHE[name] = matrix[idx].reshape(-1)
            else:
                raise KeyError(
                    f"Unrecognized task_emb cache layout in {cache_path} "
                    "(expected task_embeddings/* or task_emb_matrix + task_names)"
                )
        _TASK_EMB_CACHE_LOADED_PATH = cache_key

    if task_name not in _TASK_EMB_CACHE:
        raise KeyError(
            f"task {task_name!r} not in cache {cache_path} "
            f"(have: {sorted(_TASK_EMB_CACHE.keys())})"
        )
    emb = _TASK_EMB_CACHE[task_name]
    if emb.shape[-1] != dim:
        raise ValueError(
            f"task_emb dim mismatch for {task_name!r}: expected {dim}, got {emb.shape[-1]}"
        )
    return emb


def task_name_for_lang_encode(task_name: str) -> str:
    """Convert a task directory name to an instruction."""
    return str(task_name).replace("_", " ")


def encode_task_name_minilm(task_name: str, *, dim: int = 384) -> np.ndarray:
    """Encode the task name with the configured language model."""
    global _MINILM_ENCODER
    model_path = load_paths().paths.language_model
    from sentence_transformers import SentenceTransformer

    if _MINILM_ENCODER is None:
        _MINILM_ENCODER = SentenceTransformer(model_path)
    text = task_name_for_lang_encode(task_name)
    emb = np.asarray(_MINILM_ENCODER.encode(text), dtype=np.float32).reshape(-1)
    if emb.shape[0] != dim:
        raise ValueError(
            f"MiniLM embedding dim {emb.shape[0]} != expected {dim} (task={task_name!r})"
        )
    return emb


def load_training_config(config_yaml: Union[str, Path]) -> OmegaConf:
    config_path = resolve_path(config_yaml)
    full_path = config_path.parent / "config.full.yaml"
    if full_path.is_file():
        return OmegaConf.load(full_path)
    return OmegaConf.load(config_path)


def build_framework(
    config_yaml: Union[str, Path], ckpt_path: Union[str, Path], device: str
):
    tcla_root = get_tcla_root()
    cfg = load_training_config(config_yaml)
    ckpt_path = resolve_path(ckpt_path, base_dir=tcla_root)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model_cfg = OmegaConf.to_container(cfg.framework, resolve=True)
    framework_name = model_cfg.pop("name")
    model_cfg["device"] = str(device)
    model_cfg = resolve_framework_paths(model_cfg, base_dir=tcla_root)

    if (
        framework_name == "ACTPolicy"
        and str(model_cfg.get("action_space", "")).lower() == "latent"
    ):
        model_cfg = ACTPolicyModule.prepare_latent_eval_model_cfg(
            model_cfg, checkpoint, base_dir=tcla_root
        )

    model = FRAMEWORK_REGISTRY[framework_name](**model_cfg)
    state_dict = ACTPolicyModule.extract_framework_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model, cfg


def normalize_qpos(qpos: np.ndarray, policy_stats: dict) -> np.ndarray:
    mean = np.asarray(policy_stats["qpos_mean"], dtype=np.float32)
    std = np.asarray(policy_stats["qpos_std"], dtype=np.float32)
    return (np.asarray(qpos, dtype=np.float32) - mean) / std


def unnormalize_action(action: np.ndarray, action_ae_stats: dict) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if "action_mean" in action_ae_stats and "action_std" in action_ae_stats:
        mean = np.asarray(action_ae_stats["action_mean"], dtype=np.float32)
        std = np.asarray(action_ae_stats["action_std"], dtype=np.float32)
        return action * std + mean
    action_min = np.asarray(action_ae_stats["action_min"], dtype=np.float32)
    action_max = np.asarray(action_ae_stats["action_max"], dtype=np.float32)
    action = np.clip(action, -1.0, 1.0)
    return 0.5 * (action + 1.0) * (action_max - action_min + 1e-5) + action_min


def normalize_action(action: np.ndarray, action_ae_stats: dict) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if "action_mean" in action_ae_stats and "action_std" in action_ae_stats:
        mean = np.asarray(action_ae_stats["action_mean"], dtype=np.float32)
        std = np.asarray(action_ae_stats["action_std"], dtype=np.float32)
        return (action - mean) / std
    action_min = np.asarray(action_ae_stats["action_min"], dtype=np.float32)
    action_max = np.asarray(action_ae_stats["action_max"], dtype=np.float32)
    return 2.0 * (action - action_min) / (action_max - action_min + 1e-5) - 1.0


def load_episode_task_emb(
    data_root,
    task_name: str,
    episode_id: int,
    *,
    dim: int = 384,
) -> Optional[np.ndarray]:
    source = _task_emb_source()
    if source in ("cache", "cached", "bundle", "task_emb_cache"):
        return load_task_emb_from_cache(task_name, dim=dim)
    if source == "minilm":
        return encode_task_name_minilm(task_name, dim=dim)
    if source not in ("hdf5", "file", "episode"):
        return encode_task_name_minilm(task_name, dim=dim)

    import h5py

    task_dir = resolve_path(data_root) / task_name
    path = task_dir / f"episode_{int(episode_id)}.hdf5"
    try:
        path_exists = path.is_file()
    except OSError:
        return None
    if not path_exists:
        try:
            fallback_paths = sorted(task_dir.glob("episode_*.hdf5"))
        except OSError:
            return None
        if not fallback_paths:
            return None
        path = fallback_paths[0]
    try:
        handle_ctx = h5py.File(path, "r")
    except OSError:
        return None
    with handle_ctx as handle:
        if "task_emb" in handle:
            task_emb = handle["task_emb"][:].astype(np.float32)
            if task_emb.ndim > 1:
                task_emb = task_emb[0]
            return task_emb
        if "task_name" in handle:
            stored_name = handle["task_name"][()]
            if isinstance(stored_name, bytes):
                stored_name = stored_name.decode("utf-8")
            return encode_task_name_minilm(str(stored_name), dim=dim)
    return encode_task_name_minilm(task_name, dim=dim)


@torch.no_grad()
def infer_phase_scalar(
    policy,
    executed_actions_norm: Sequence[np.ndarray],
    *,
    action_ae_stats: Optional[dict] = None,
    task_emb: Optional[torch.Tensor] = None,
) -> Optional[float]:
    """Predict scalar task progress with the same path used by AE progress loss."""
    action_ae = policy.action_ae
    if action_ae is None:
        return None
    traj_a_freq = build_traj_a_freq_from_executed(
        policy,
        executed_actions_norm,
        action_ae_stats=action_ae_stats,
    )
    device = next(action_ae.progress_model.parameters()).device
    dtype = next(action_ae.progress_model.parameters()).dtype
    traj_a_len = torch.tensor(
        [len(executed_actions_norm)], dtype=torch.long, device=device
    )
    freq = action_ae.resolve_traj_a_freq(
        traj_a_freq,
        traj_a_len=traj_a_len,
        batch_size=1,
        device=device,
        dtype=dtype,
    )
    progress_model = action_ae.progress_model
    if hasattr(progress_model, "predict_progress"):
        if task_emb is None:
            return None
        te = _task_emb_for_progress(task_emb, device=device, dtype=dtype)
        progress = progress_model.predict_progress(freq, te)
        return float(progress.reshape(-1)[0].detach().cpu().item())
    if hasattr(progress_model, "get_phase_latent"):
        if task_emb is None:
            return None
        te = _task_emb_for_progress(task_emb, device=device, dtype=dtype)
        latent = progress_model.get_phase_latent(freq, te)
        return float(latent.norm(dim=-1).reshape(-1)[0].detach().cpu().item())
    progress = progress_model(freq)
    return float(progress.reshape(-1)[0].detach().cpu().item())


def save_phase_curve_plot(
    steps: Sequence[int],
    phases: Sequence[float],
    save_path: Union[str, Path],
    *,
    episode_id: Optional[int] = None,
    task_name: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    phases_arr = np.asarray(phases, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, phases_arr, color="#2563eb", linewidth=1.5, label="predicted phase")
    ax.set_xlabel("env step")
    if phases_arr.size and (
        float(phases_arr.min()) < 0.0 or float(phases_arr.max()) > 1.0
    ):
        lo = float(phases_arr.min())
        hi = float(phases_arr.max())
        pad = max(0.05 * (hi - lo), 0.05)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel("phase metric")
    else:
        ax.set_ylabel("phase (progress)")
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    title_parts = ["phase prediction"]
    if task_name:
        title_parts.append(task_name)
    if episode_id is not None:
        title_parts.append(f"episode {episode_id}")
    ax.set_title(" | ".join(title_parts))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    csv_path = save_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("step,phase\n")
        for step, phase in zip(steps, phases):
            f.write(f"{step},{phase}\n")


def _task_emb_for_progress(
    task_emb: torch.Tensor,
    *,
    device,
    dtype,
) -> torch.Tensor:
    """Perceiver progress expects ``[B, lang_dim]``."""
    te = task_emb.to(device=device, dtype=dtype)
    if te.ndim == 1:
        te = te.unsqueeze(0)
    elif te.ndim == 3 and te.shape[1] == 1:
        te = te.squeeze(1)
    return te


def _normalize_traj_a_freq(
    freq: np.ndarray, action_ae_stats: Optional[dict]
) -> np.ndarray:
    if not action_ae_stats:
        return freq.astype(np.float32)
    if (
        "traj_a_freq_mean" not in action_ae_stats
        or "traj_a_freq_std" not in action_ae_stats
    ):
        return freq.astype(np.float32)
    mean = np.asarray(action_ae_stats["traj_a_freq_mean"], dtype=np.float32)
    std = np.asarray(action_ae_stats["traj_a_freq_std"], dtype=np.float32)
    return ((freq - mean) / std).astype(np.float32)


def build_traj_a_freq_from_executed(
    policy,
    executed_actions_norm: Sequence[np.ndarray],
    action_ae_stats: Optional[dict] = None,
) -> Optional[torch.Tensor]:
    """Build spectral context from executed actions; empty history uses AE learned init."""
    action_ae = policy.action_ae
    if not executed_actions_norm:
        return None
    trajectory = torch.from_numpy(
        np.stack(executed_actions_norm, axis=0).astype(np.float32)
    )
    freq = action_ae.trajectory_to_frequency(trajectory)
    if action_ae_stats is not None:
        freq_np = _normalize_traj_a_freq(freq.detach().cpu().numpy(), action_ae_stats)
        freq = torch.from_numpy(freq_np).to(device=freq.device, dtype=freq.dtype)
    return freq


def stack_images(
    camera_names: Sequence[str],
    *,
    head_cam: Optional[np.ndarray] = None,
    left_cam: Optional[np.ndarray] = None,
    right_cam: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Map HDF5 / sim camera keys to policy camera_names (cam_high, cam_left_wrist, cam_right_wrist)."""
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
    task_emb: torch.Tensor,
    traj_a_freq: Optional[torch.Tensor] = None,
    traj_a_len: Optional[int] = None,
) -> np.ndarray:
    policy = model.policy
    qpos = qpos.unsqueeze(0).to(model.device)
    image = image.unsqueeze(0).to(model.device)
    task_emb = task_emb.unsqueeze(0).to(model.device)
    if traj_a_freq is not None:
        traj_a_freq = traj_a_freq.to(model.device)
    traj_a_len_tensor = None
    if traj_a_len is not None:
        traj_a_len_tensor = torch.tensor(
            [traj_a_len], dtype=torch.long, device=model.device
        )

    decoded, _ = policy.predict_action(
        qpos,
        image,
        task_emb=task_emb,
        traj_a_freq=traj_a_freq,
        traj_a_len=traj_a_len_tensor,
    )
    return decoded[0].detach().cpu().numpy()
