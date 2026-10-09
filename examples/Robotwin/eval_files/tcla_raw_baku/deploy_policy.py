"""RoboTwin deploy entry for TCLA raw Baku policies."""

import numpy as np
import torch

from tcla_latent_baku.deploy_policy import (
    TCLALatentBakuModel,
    eval,
    reset_model,
)
from tcla_latent_baku.model_utils import (
    normalize_qpos,
    predict_action_chunk,
    stack_images,
    unnormalize_action,
)

__all__ = ["eval", "get_model", "reset_model"]


class TCLARawBakuModel(TCLALatentBakuModel):
    def _query_chunk(self, obs):
        qpos = normalize_qpos(
            np.asarray(obs["qpos"], dtype=np.float32), self.policy_stats
        )
        image = stack_images(
            self.camera_names,
            head_cam=obs.get("head_cam"),
            left_cam=obs.get("left_cam"),
            right_cam=obs.get("right_cam"),
        )
        normalized_chunk = predict_action_chunk(
            self.model,
            torch.from_numpy(qpos),
            image,
            self.task_emb,
            traj_a_freq=None,
            traj_a_len=None,
        )
        return unnormalize_action(normalized_chunk, self.action_ae_stats)


def get_model(usr_args):
    return TCLARawBakuModel(usr_args)
