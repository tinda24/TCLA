import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import OmegaConf

from tcla.model.framework import action_autoencoder_version3  # noqa: F401
from tcla.model.modules.act.detr.main import build_ACT_model_and_optimizer
from tcla.model.tools import FRAMEWORK_REGISTRY


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            return {
                "l1": l1,
                "kl": total_kld[0],
                "loss": l1 + total_kld[0] * self.kl_weight,
            }

        a_hat, _, (_, _) = self.model(qpos, image, env_state)
        return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def get_samples(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        a_hat, _, (_, _) = self.model.get_samples(
            qpos, image, env_state, num_samples=10
        )
        return a_hat


class ACTPolicyModule(nn.Module):
    def __init__(
        self,
        act_config,
        action_space="latent",
        action_ae_enabled=True,
        action_ae_config=None,
        action_ae_ckpt=None,
        kl_weight=10.0,
        decoded_loss_coef=0.1,
        latent_loss_coef=1.0,
        device="cuda",
    ):
        super().__init__()
        self.device_name = device
        self.action_space = action_space.strip().lower()
        if self.action_space not in {"raw", "latent"}:
            raise ValueError(f"Unsupported action_space: {action_space}")
        self.action_ae_enabled = bool(action_ae_enabled)
        self.kl_weight = float(kl_weight)
        self.decoded_loss_coef = float(decoded_loss_coef)
        self.latent_loss_coef = float(latent_loss_coef)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        act_args = dict(act_config)
        self.action_ae = None
        if self.action_space == "latent":
            if not self.action_ae_enabled:
                raise ValueError("latent action_space requires action_ae_enabled=true.")
            if action_ae_config is None or action_ae_ckpt is None:
                raise ValueError(
                    "latent action_space requires action_ae_config and action_ae_ckpt."
                )
            act_args = self._sync_act_config_from_action_ae(act_args, action_ae_config)
            self.action_ae = self._load_action_autoencoder(
                action_ae_config, action_ae_ckpt, device
            )

        model, _ = build_ACT_model_and_optimizer(act_args)
        self.model = model
        self.to(device)

    @staticmethod
    def _ae_framework_dims(config_yaml: str) -> dict:
        fw = OmegaConf.load(config_yaml).framework
        chunk_size = int(fw.chunk_size)
        downsample_rate = int(fw.downsample_rate)
        if chunk_size % downsample_rate != 0:
            raise ValueError(
                f"AE chunk_size ({chunk_size}) must be divisible by downsample_rate ({downsample_rate})."
            )
        return {
            "latent_action_dim": int(fw.latent_action_dim),
            "chunk_size": chunk_size,
            "num_queries": chunk_size // downsample_rate,
        }

    @staticmethod
    def extract_framework_state_dict(checkpoint: dict) -> dict:
        if checkpoint.get("format") == "framework_state_dict_v1":
            return checkpoint["state_dict"]
        if (
            isinstance(checkpoint.get("model"), dict)
            and checkpoint["model"].get("format") == "framework_state_dict_v1"
        ):
            return checkpoint["model"]["state_dict"]
        if isinstance(checkpoint, dict) and all(
            isinstance(k, str) for k in checkpoint.keys()
        ):
            return checkpoint
        raise ValueError(
            "Unsupported checkpoint format. Expected framework_state_dict_v1 at top level or under 'model'."
        )

    @staticmethod
    def _find_state_dict_tensor(state_dict: dict, suffix: str):
        for key, value in state_dict.items():
            if key.endswith(suffix):
                return value
        return None

    @classmethod
    def infer_latent_dims_from_checkpoint(cls, checkpoint) -> dict:
        """Infer ACT latent token dims from a saved ACTPolicy checkpoint (authoritative at eval)."""
        state_dict = cls.extract_framework_state_dict(checkpoint)
        enc_w = cls._find_state_dict_tensor(
            state_dict, "policy.model.encoder_action_proj.weight"
        )
        query_w = cls._find_state_dict_tensor(
            state_dict, "policy.model.query_embed.weight"
        )
        if enc_w is None or query_w is None:
            raise KeyError(
                "Cannot infer latent ACT dims: missing policy.model.encoder_action_proj.weight "
                "or policy.model.query_embed.weight in checkpoint."
            )
        return {
            "action_dim": int(enc_w.shape[1]),
            "num_queries": int(query_w.shape[0]),
        }

    @classmethod
    def _ae_latent_dim(cls, action_ae_config: str) -> int:
        return cls._ae_framework_dims(action_ae_config)["latent_action_dim"]

    @classmethod
    def _resolve_action_ae_for_eval(
        cls,
        model_cfg: dict,
        checkpoint: dict,
        inferred_action_dim: int,
        base_dir: Path,
    ) -> dict:
        """Pick frozen AE paths that match checkpoint latent width."""
        if checkpoint.get("resolved_action_ae_config"):
            model_cfg["action_ae_config"] = checkpoint["resolved_action_ae_config"]
        if checkpoint.get("resolved_action_ae_ckpt"):
            model_cfg["action_ae_ckpt"] = checkpoint["resolved_action_ae_ckpt"]

        for key in ("action_ae_config", "action_ae_ckpt"):
            if model_cfg.get(key):
                model_cfg[key] = cls._resolve_path(model_cfg[key], base_dir)

        ae_config = model_cfg.get("action_ae_config")
        if not ae_config:
            return model_cfg

        if cls._ae_latent_dim(ae_config) == inferred_action_dim:
            return model_cfg

        yaml_dim = cls._ae_latent_dim(ae_config)
        print(
            "[ACTPolicyModule] Warning: yaml action_ae latent_action_dim="
            f"{yaml_dim} does not match checkpoint action_dim={inferred_action_dim}."
        )

        env_config = os.environ.get("ACTION_AE_CONFIG")
        env_ckpt = os.environ.get("ACTION_AE_CKPT")
        if env_config and env_ckpt:
            env_config = cls._resolve_path(env_config, base_dir)
            env_ckpt = cls._resolve_path(env_ckpt, base_dir)
            if cls._ae_latent_dim(env_config) == inferred_action_dim:
                print(
                    "[ACTPolicyModule] Using ACTION_AE_CONFIG/ACTION_AE_CKPT from environment."
                )
                model_cfg["action_ae_config"] = env_config
                model_cfg["action_ae_ckpt"] = env_ckpt
                return model_cfg

        sidecar = base_dir / "action_ae_resolved.json"
        if sidecar.is_file():
            import json

            with sidecar.open("r", encoding="utf-8") as handle:
                resolved = json.load(handle)
            sc_config = cls._resolve_path(resolved["action_ae_config"], base_dir)
            sc_ckpt = cls._resolve_path(resolved["action_ae_ckpt"], base_dir)
            if cls._ae_latent_dim(sc_config) == inferred_action_dim:
                print(f"[ACTPolicyModule] Using action_ae_resolved.json: {sidecar}")
                model_cfg["action_ae_config"] = sc_config
                model_cfg["action_ae_ckpt"] = sc_ckpt
                return model_cfg

        raise ValueError(
            "Action-AE config does not match the policy checkpoint: "
            f"yaml AE latent_action_dim={yaml_dim} ({ae_config}) vs "
            f"checkpoint action_dim={inferred_action_dim}. "
            "Set ACTION_AE_CONFIG and ACTION_AE_CKPT to the AE used during policy training, "
            "or retrain / use a checkpoint that includes resolved_action_ae_* fields."
        )

    @staticmethod
    def _resolve_path(path: Union[str, Path], base_dir: Path) -> str:
        path = Path(path)
        if path.is_absolute():
            return str(path)
        return str((base_dir / path).resolve())

    @classmethod
    def prepare_latent_eval_model_cfg(
        cls,
        model_cfg: dict,
        checkpoint: dict,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> dict:
        base = (
            Path(base_dir)
            if base_dir is not None
            else Path(__file__).resolve().parents[3]
        )

        inferred = cls.infer_latent_dims_from_checkpoint(checkpoint)
        model_cfg = cls._resolve_action_ae_for_eval(
            model_cfg, checkpoint, inferred["action_dim"], base
        )
        act_config = dict(model_cfg.get("act_config") or {})

        if model_cfg.get("action_ae_config"):
            act_config = cls._sync_act_config_from_action_ae(
                act_config, model_cfg["action_ae_config"]
            )

        act_config["action_dim"] = inferred["action_dim"]
        act_config["num_queries"] = inferred["num_queries"]
        model_cfg["act_config"] = act_config

        print(
            "[ACTPolicyModule] Eval build dims from checkpoint: "
            f"action_dim={inferred['action_dim']}, num_queries={inferred['num_queries']}"
        )
        return model_cfg

    @classmethod
    def _sync_act_config_from_action_ae(
        cls, act_config: dict, action_ae_config: str
    ) -> dict:
        """Align ACT token shape with the frozen action autoencoder."""
        act_args = dict(act_config)
        ae_dims = cls._ae_framework_dims(action_ae_config)

        prev_action_dim = act_args.get("action_dim")
        prev_num_queries = act_args.get("num_queries")
        prev_chunk_size = act_args.get("chunk_size")

        act_args["action_dim"] = ae_dims["latent_action_dim"]
        act_args["num_queries"] = ae_dims["num_queries"]
        act_args["chunk_size"] = ae_dims["chunk_size"]

        if (
            prev_action_dim is not None
            and int(prev_action_dim) != ae_dims["latent_action_dim"]
        ):
            print(
                "[ACTPolicyModule] act_config.action_dim "
                f"{prev_action_dim} -> {ae_dims['latent_action_dim']} (from action AE)"
            )
        if (
            prev_num_queries is not None
            and int(prev_num_queries) != ae_dims["num_queries"]
        ):
            print(
                "[ACTPolicyModule] act_config.num_queries "
                f"{prev_num_queries} -> {ae_dims['num_queries']} (from action AE)"
            )
        if (
            prev_chunk_size is not None
            and int(prev_chunk_size) != ae_dims["chunk_size"]
        ):
            print(
                "[ACTPolicyModule] act_config.chunk_size "
                f"{prev_chunk_size} -> {ae_dims['chunk_size']} (from action AE)"
            )
        print(
            "[ACTPolicyModule] Latent ACT dims synced from AE: "
            f"action_dim={ae_dims['latent_action_dim']}, num_queries={ae_dims['num_queries']}, "
            f"chunk_size={ae_dims['chunk_size']}"
        )
        return act_args

    def _load_action_autoencoder(self, config_yaml, checkpoint_path, device):
        cfg = OmegaConf.load(config_yaml)
        model_cfg = OmegaConf.to_container(cfg.framework, resolve=True)
        framework_name = model_cfg.pop("name")
        model_cfg["device"] = str(device)
        action_ae = FRAMEWORK_REGISTRY[framework_name](**model_cfg)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        missing, unexpected = action_ae.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                f"ActionAutoencoder checkpoint loaded with missing={len(missing)}, "
                f"unexpected={len(unexpected)}"
            )
        action_ae.requires_grad_(False)
        action_ae.eval()
        self.action_ae_chunk_size = int(cfg.framework.chunk_size)
        self.action_ae_latent_tokens = int(cfg.framework.chunk_size) // int(
            cfg.framework.downsample_rate
        )
        self.action_ae_lang_len = int(cfg.framework.lang_len)
        self.action_ae_lang_dim = int(cfg.framework.lang_dim)
        self.action_ae_is_relative_action = bool(
            cfg.framework.get("is_relative_action", False)
        )
        return action_ae

    def _action_ae_lang(self, task_emb, batch_size, device, dtype):
        if task_emb is None:
            return torch.zeros(
                batch_size,
                self.action_ae_lang_len,
                self.action_ae_lang_dim,
                device=device,
                dtype=dtype,
            )
        return task_emb.to(device=device, dtype=dtype)

    def _encode_action_ae(self, action, task_emb):
        lang = self._action_ae_lang(
            task_emb, action.shape[0], action.device, action.dtype
        )
        if self.action_ae_is_relative_action:
            raise ValueError(
                "Latent policy path needs qpos support for is_relative_action=True action autoencoders."
            )
        return self.action_ae.encoder(action, lang)

    def _decode_action_ae(
        self, latent, task_emb, phase=None, traj_a_freq=None, traj_a_len=None
    ):
        lang = self._action_ae_lang(
            task_emb, latent.shape[0], latent.device, latent.dtype
        )
        raw_freq_action = self.action_ae.resolve_traj_a_freq(
            traj_a_freq,
            traj_a_len=traj_a_len,
            batch_size=latent.shape[0],
            device=latent.device,
            dtype=latent.dtype,
        )
        decoded = self.action_ae.decode_latent(
            latent, lang, phase, raw_freq_action=raw_freq_action, traj_a_len=traj_a_len
        )
        if self.action_ae_is_relative_action:
            _, decoded = decoded
        return decoded

    def _prepare_traj_a_freq(self, traj_a_freq: torch.Tensor) -> torch.Tensor:
        """Progress model expects [B, num_freqs, 2 * action_dim]."""
        if traj_a_freq.ndim == 2:
            return traj_a_freq.unsqueeze(0)
        if traj_a_freq.ndim == 3:
            return traj_a_freq
        raise ValueError(
            f"traj_a_freq must be [num_freqs, 2*D] or [B, num_freqs, 2*D], got {tuple(traj_a_freq.shape)}."
        )

    def _infer_phase(
        self,
        traj_a_freq: Optional[torch.Tensor],
        task_emb=None,
        traj_a_len: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if self.action_ae is None:
            return None
        device = self.action_ae.initial_traj_a_freq.device
        dtype = next(self.action_ae.progress_model.parameters()).dtype
        lang = None
        if batch_size is not None and hasattr(
            self.action_ae.progress_model, "get_phase_latent"
        ):
            lang = self._action_ae_lang(task_emb, batch_size, device, dtype)
        with torch.no_grad():
            return self.action_ae.phase_from_freq(
                traj_a_freq,
                task_emb=lang,
                traj_a_len=traj_a_len,
                batch_size=batch_size,
            )

    def forward(
        self,
        qpos,
        image,
        action=None,
        task_emb=None,
        is_pad=None,
        traj_a_freq=None,
        traj_a_len=None,
    ):
        image = self.normalize(image)
        env_state = None
        if action is None:
            action_pred, latent_pred = self._predict_action(
                qpos,
                image,
                env_state,
                task_emb=task_emb,
                traj_a_freq=traj_a_freq,
                traj_a_len=traj_a_len,
            )
            return action_pred, latent_pred

        if is_pad is None:
            is_pad = torch.zeros(
                action.shape[:2], dtype=torch.bool, device=action.device
            )
        if self.action_space == "raw":
            target = action[:, : self.model.num_queries]
            target_is_pad = is_pad[:, : self.model.num_queries]
            pred_seq, _, (mu, logvar) = self.model(
                qpos, image, env_state, target, target_is_pad
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(target, pred_seq, reduction="none")
            l1_loss = (all_l1 * ~target_is_pad.unsqueeze(-1)).mean()
            decoded_action_l1 = pred_seq.new_zeros(())
            total_loss = l1_loss + self.kl_weight * total_kld[0]
            decoded_actions = pred_seq
            latent_pred = None
            latent_target = None
        else:
            ae_action = action[:, : self.action_ae_chunk_size]
            phase = self._infer_phase(
                traj_a_freq,
                task_emb=task_emb,
                traj_a_len=traj_a_len,
                batch_size=action.shape[0],
            )
            with torch.no_grad():
                latent_target = self._encode_action_ae(ae_action, task_emb)

            latent_action = latent_target
            latent_is_pad = is_pad[:, : latent_target.shape[1]]
            latent_pred_seq, _, (mu, logvar) = self.model(
                qpos, image, env_state, latent_action, latent_is_pad
            )
            latent_pred = latent_pred_seq[:, : latent_target.shape[1]]
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            l1_loss = F.l1_loss(latent_pred, latent_target)
            decoded_actions = self._decode_action_ae(
                latent_pred,
                task_emb,
                phase=phase,
                traj_a_freq=traj_a_freq,
                traj_a_len=traj_a_len,
            )
            decoded_action_l1 = F.l1_loss(decoded_actions, ae_action)
            total_loss = (
                self.latent_loss_coef * l1_loss
                + self.kl_weight * total_kld[0]
                + self.decoded_loss_coef * decoded_action_l1
            )
        return {
            "loss": total_loss,
            "l1_loss": l1_loss,
            "decoded_action_l1_loss": decoded_action_l1,
            "kl_loss": total_kld[0],
            "latent_pred": latent_pred,
            "latent_target": latent_target,
            "decoded_actions": decoded_actions,
        }

    def _predict_action(
        self, qpos, image, env_state, task_emb=None, traj_a_freq=None, traj_a_len=None
    ):
        pred_seq, is_pad_hat, stats = self.model(qpos, image, env_state)
        if self.action_space == "raw":
            return pred_seq, None
        latent_pred = pred_seq[:, : self.action_ae.action_latent_tokens]
        phase = self._infer_phase(
            traj_a_freq,
            task_emb=task_emb,
            traj_a_len=traj_a_len,
            batch_size=qpos.shape[0],
        )
        decoded_actions = self._decode_action_ae(
            latent_pred,
            task_emb,
            phase=phase,
            traj_a_freq=traj_a_freq,
            traj_a_len=traj_a_len,
        )
        return decoded_actions, latent_pred

    def predict_action(
        self, qpos, image, task_emb=None, traj_a_freq=None, traj_a_len=None
    ):
        self.eval()
        with torch.no_grad():
            actions, latent_pred = self.forward(
                qpos,
                image,
                action=None,
                task_emb=task_emb,
                traj_a_freq=traj_a_freq,
                traj_a_len=traj_a_len,
            )
        return actions, latent_pred


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld
