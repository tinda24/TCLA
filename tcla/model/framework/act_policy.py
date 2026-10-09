import torch
import torch.nn as nn

from tcla.model.modules.act.policy import ACTPolicyModule
from tcla.model.tools import FRAMEWORK_REGISTRY


class ACTPolicyFramework(nn.Module):
    def __init__(
        self,
        act_config,
        action_space="latent",
        action_ae_enabled=True,
        action_ae_config=None,
        action_ae_ckpt=None,
        image_key="image",
        qpos_key="qpos",
        action_key="action",
        lang_key="task_emb",
        is_pad_key="is_pad",
        traj_a_freq_key="traj_a_freq",
        kl_weight=10.0,
        decoded_loss_coef=0.1,
        latent_loss_coef=1.0,
        device="cuda",
    ):
        super().__init__()
        self.action_space = action_space.strip().lower()
        self.image_key = image_key
        self.qpos_key = qpos_key
        self.action_key = action_key
        self.lang_key = lang_key
        self.is_pad_key = is_pad_key
        self.traj_a_freq_key = traj_a_freq_key
        self.traj_a_len_key = "traj_a_len"
        self.policy = ACTPolicyModule(
            act_config=act_config,
            action_space=self.action_space,
            action_ae_enabled=action_ae_enabled,
            action_ae_config=action_ae_config,
            action_ae_ckpt=action_ae_ckpt,
            kl_weight=kl_weight,
            decoded_loss_coef=decoded_loss_coef,
            latent_loss_coef=latent_loss_coef,
            device=device,
        )
        self.to(device)

    def forward(self, examples):
        data = self.prepare_data(examples)
        traj_a_freq = data.get(self.traj_a_freq_key)
        traj_a_len = data.get(self.traj_a_len_key)
        outputs = self.policy(
            qpos=data[self.qpos_key],
            image=data[self.image_key],
            action=data[self.action_key],
            task_emb=data.get(self.lang_key),
            is_pad=data.get(self.is_pad_key),
            traj_a_freq=traj_a_freq,
            traj_a_len=traj_a_len,
        )
        return {
            "action_loss": outputs["loss"],
            "l1_loss": outputs["l1_loss"],
            "latent_l1_loss": outputs.get("latent_l1_loss", outputs["l1_loss"]),
            "decoded_action_l1_loss": outputs["decoded_action_l1_loss"],
            "kl_loss": outputs["kl_loss"],
        }

    @torch.no_grad()
    def predict_action(self, examples):
        data = self.prepare_data(examples)
        normalized_actions, latents = self.policy.predict_action(
            qpos=data[self.qpos_key],
            image=data[self.image_key],
            task_emb=data.get(self.lang_key),
            traj_a_freq=data.get(self.traj_a_freq_key),
            traj_a_len=data.get(self.traj_a_len_key),
        )
        return {
            "normalized_actions": normalized_actions,
            "latents": latents,
        }

    def prepare_data(self, examples):
        keys = examples[0].keys()
        return {
            key: self._stack_values([example[key] for example in examples])
            for key in keys
        }

    def _stack_values(self, values):
        first = values[0]
        if torch.is_tensor(first):
            return torch.stack(values, dim=0).to(self.device)
        if isinstance(first, str):
            return list(values)
        return torch.as_tensor(values, device=self.device)

    @property
    def device(self):
        return next(self.parameters()).device


FRAMEWORK_REGISTRY.register("ACTPolicy")(ACTPolicyFramework)

ACTPolicy = ACTPolicyFramework
