import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import OmegaConf

from tcla.model.framework import action_autoencoder_version3  # noqa: F401
from tcla.model.modules.act.policy import ACTPolicyModule
from tcla.model.tools import FRAMEWORK_REGISTRY


class BakuImageEncoder(nn.Module):
    """Small ResNet image tokenizer used by the copied Baku-style policy."""

    def __init__(self, hidden_dim: int, pretrained_backbone: bool = False):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
            backbone = resnet18(weights=weights)
        except Exception:
            from torchvision.models import resnet18

            backbone = resnet18(pretrained=pretrained_backbone)

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(512, hidden_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError(f"image must be [B, N, C, H, W], got {tuple(image.shape)}")
        batch, num_views, channels, height, width = image.shape
        flat = image.reshape(batch * num_views, channels, height, width)
        feat = self.encoder(flat).flatten(1)
        feat = self.proj(feat)
        return feat.reshape(batch, num_views, -1)


class BakuBackbone(nn.Module):
    """GPT-like observation backbone with learned action query tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        num_layers: int = 4,
        nheads: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.action_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        action_tokens = self.action_tokens.expand(obs_tokens.shape[0], -1, -1)
        tokens = torch.cat([obs_tokens, action_tokens], dim=1)
        encoded = self.norm(self.encoder(tokens))
        return encoded[:, -action_tokens.shape[1] :, :]


class BakuActionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_dim = hidden_dim
        for _ in range(max(0, int(num_layers) - 1)):
            layers.extend(
                [nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        return self.net(action_features)


class BakuPolicyNet(nn.Module):
    def __init__(
        self,
        camera_names,
        state_dim: int,
        action_dim: int,
        num_queries: int,
        hidden_dim: int = 512,
        enc_layers: int = 4,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        head_layers: int = 2,
        head_dropout: float = 0.0,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.camera_names = list(camera_names)
        self.num_queries = int(num_queries)
        self.image_encoder = BakuImageEncoder(
            hidden_dim, pretrained_backbone=pretrained_backbone
        )
        self.qpos_proj = nn.Linear(state_dim, hidden_dim)
        self.prompt = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.backbone = BakuBackbone(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=enc_layers,
            nheads=nheads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )
        self.action_head = BakuActionHead(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_layers=head_layers,
            dropout=head_dropout,
        )

    def forward(self, qpos: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_encoder(image)
        qpos_token = self.qpos_proj(qpos).unsqueeze(1)
        prompt = self.prompt.expand(qpos.shape[0], -1, -1)
        obs_tokens = torch.cat([prompt, qpos_token, image_tokens], dim=1)
        action_features = self.backbone(obs_tokens)
        return self.action_head(action_features)


class BakuPolicyModule(nn.Module):
    def __init__(
        self,
        baku_config,
        action_space="latent",
        action_ae_enabled=True,
        action_ae_config=None,
        action_ae_ckpt=None,
        decoded_loss_coef=0.1,
        latent_loss_coef=1.0,
        raw_loss_coef=1.0,
        device="cuda",
    ):
        super().__init__()
        self.device_name = device
        self.action_space = str(action_space).strip().lower()
        if self.action_space not in {"raw", "latent"}:
            raise ValueError(f"Unsupported action_space: {action_space}")
        self.action_ae_enabled = bool(action_ae_enabled)
        self.decoded_loss_coef = float(decoded_loss_coef)
        self.latent_loss_coef = float(latent_loss_coef)
        self.raw_loss_coef = float(raw_loss_coef)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        model_args = dict(baku_config)
        self.action_ae = None
        if self.action_space == "latent":
            if not self.action_ae_enabled:
                raise ValueError("latent action_space requires action_ae_enabled=true.")
            if action_ae_config is None or action_ae_ckpt is None:
                raise ValueError(
                    "latent action_space requires action_ae_config and action_ae_ckpt."
                )
            action_ae_config, action_ae_ckpt = self._resolve_action_ae_paths(
                action_ae_config,
                action_ae_ckpt,
            )
            model_args = self._sync_config_from_action_ae(model_args, action_ae_config)
            self.action_ae = self._load_action_autoencoder(
                action_ae_config, action_ae_ckpt, device
            )

        self.model = BakuPolicyNet(**model_args)
        self.to(device)

    @staticmethod
    def _resolve_action_ae_paths(
        action_ae_config: str, action_ae_ckpt: str
    ) -> tuple[str, str]:
        cfg = OmegaConf.load(action_ae_config)
        if "framework" in cfg:
            return action_ae_config, action_ae_ckpt
        if "action_ae_config" not in cfg or "action_ae_ckpt" not in cfg:
            raise ValueError(
                "action_ae_config must be an AE config.yaml with a framework section "
                "or an action_ae_resolved.json containing action_ae_config/action_ae_ckpt."
            )
        return str(cfg.action_ae_config), str(cfg.action_ae_ckpt)

    @classmethod
    def _sync_config_from_action_ae(cls, config: dict, action_ae_config: str) -> dict:
        args = dict(config)
        fw = OmegaConf.load(action_ae_config).framework
        chunk_size = int(fw.chunk_size)
        downsample_rate = int(fw.downsample_rate)
        if chunk_size % downsample_rate != 0:
            raise ValueError(
                f"AE chunk_size ({chunk_size}) must be divisible by downsample_rate ({downsample_rate})."
            )
        args["action_dim"] = int(fw.latent_action_dim)
        args["num_queries"] = chunk_size // downsample_rate
        return args

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

    def _phase_from_traj_a_freq(
        self, traj_a_freq, task_emb, batch_size, device, dtype, traj_a_len=None
    ):
        lang = self._action_ae_lang(task_emb, batch_size, device, dtype)
        if hasattr(self.action_ae.progress_model, "get_phase_latent"):
            return self.action_ae.phase_from_freq(
                traj_a_freq,
                task_emb=lang,
                traj_a_len=traj_a_len,
                batch_size=batch_size,
            )
        return self.action_ae.phase_from_freq(
            traj_a_freq,
            traj_a_len=traj_a_len,
            batch_size=batch_size,
        )

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
            latent,
            lang,
            phase,
            raw_freq_action=raw_freq_action,
            traj_a_len=traj_a_len,
        )
        if self.action_ae_is_relative_action:
            _, decoded = decoded
        return decoded

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
        pred_seq = self.model(qpos, image)
        if action is None:
            if self.action_space == "raw":
                return pred_seq, None
            latent_pred = pred_seq[:, : self.action_ae.action_latent_tokens]
            with torch.no_grad():
                phase = self._phase_from_traj_a_freq(
                    traj_a_freq,
                    task_emb,
                    batch_size=qpos.shape[0],
                    device=qpos.device,
                    dtype=qpos.dtype,
                    traj_a_len=traj_a_len,
                )
            decoded = self._decode_action_ae(
                latent_pred,
                task_emb,
                phase=phase,
                traj_a_freq=traj_a_freq,
                traj_a_len=traj_a_len,
            )
            return decoded, latent_pred

        if is_pad is None:
            is_pad = torch.zeros(
                action.shape[:2], dtype=torch.bool, device=action.device
            )
        if self.action_space == "raw":
            target = action[:, : self.model.num_queries]
            target_is_pad = is_pad[:, : self.model.num_queries]
            pred = pred_seq[:, : target.shape[1]]
            all_loss = F.mse_loss(pred, target, reduction="none")
            loss = (all_loss * ~target_is_pad.unsqueeze(-1)).mean()
            zero = pred.new_zeros(())
            return {
                "loss": self.raw_loss_coef * loss,
                "l1_loss": loss,
                "decoded_action_l1_loss": zero,
                "kl_loss": zero,
                "latent_pred": None,
                "latent_target": None,
                "decoded_actions": pred,
            }

        ae_action = action[:, : self.action_ae_chunk_size]
        with torch.no_grad():
            phase = self._phase_from_traj_a_freq(
                traj_a_freq,
                task_emb,
                batch_size=action.shape[0],
                device=action.device,
                dtype=action.dtype,
                traj_a_len=traj_a_len,
            )
        with torch.no_grad():
            lang = self._action_ae_lang(
                task_emb, ae_action.shape[0], ae_action.device, ae_action.dtype
            )
            if self.action_ae_is_relative_action:
                raise ValueError(
                    "Latent Baku path needs qpos support for is_relative_action=True action autoencoders."
                )
            latent_target = self.action_ae.encoder(ae_action, lang)
        latent_pred = pred_seq[:, : latent_target.shape[1]]
        latent_loss = F.mse_loss(latent_pred, latent_target)
        decoded_actions = self._decode_action_ae(
            latent_pred,
            task_emb,
            phase=phase,
            traj_a_freq=traj_a_freq,
            traj_a_len=traj_a_len,
        )
        decoded_action_loss = F.mse_loss(decoded_actions, ae_action)
        zero = latent_pred.new_zeros(())
        total_loss = (
            self.latent_loss_coef * latent_loss
            + self.decoded_loss_coef * decoded_action_loss
        )
        return {
            "loss": total_loss,
            "l1_loss": latent_loss,
            "decoded_action_l1_loss": decoded_action_loss,
            "kl_loss": zero,
            "latent_pred": latent_pred,
            "latent_target": latent_target,
            "decoded_actions": decoded_actions,
        }

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

    @staticmethod
    def extract_framework_state_dict(checkpoint: dict) -> dict:
        return ACTPolicyModule.extract_framework_state_dict(checkpoint)
