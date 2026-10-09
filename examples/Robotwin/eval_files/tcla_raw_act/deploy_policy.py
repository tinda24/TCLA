"""
RoboTwin deploy entry for TCLA raw ACT policies (no action autoencoder).
"""

from pathlib import Path

import numpy as np
import torch

from .model_utils import (
    build_framework,
    load_json,
    normalize_qpos,
    predict_action_chunk,
    resolve_path,
    stack_images,
    unnormalize_action,
)


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class TCLARawACTModel:
    def __init__(self, usr_args):
        self.usr_args = usr_args
        device = usr_args.get("device", "cuda:0")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        config_yaml = usr_args["config_yaml"]
        ckpt_path = usr_args["ckpt_path"]
        run_dir = resolve_path(usr_args.get("run_dir", Path(ckpt_path).parents[1]))

        self.model, self.cfg = build_framework(config_yaml, ckpt_path, str(self.device))
        self.policy_stats = load_json(run_dir / "policy_statistics.json")
        self.action_stats = load_json(run_dir / "action_ae_statistics.json")

        self.camera_names = list(usr_args.get("camera_names") or self.cfg.camera_names)
        self.task_name = usr_args["task_name"]
        self.chunk_size = int(usr_args.get("chunk_size", self.cfg.chunk_size))
        self.temporal_agg = _as_bool(usr_args.get("temporal_agg", True), default=True)
        self.temporal_agg_k = float(usr_args.get("temporal_agg_k", 0.01))
        self._validate_camera_names()

        self.action_chunk = None
        self.chunk_index = 0
        self.t = 0
        self.max_timesteps = int(usr_args.get("max_timesteps", 3000))
        self.all_time_actions = None
        self.all_time_mask = None

        if self.action_space != "raw":
            raise ValueError(
                f"tcla_raw_act requires framework.action_space=raw, got {self.action_space!r}"
            )

        if self.temporal_agg:
            action_dim = int(self.cfg.framework.act_config.state_dim)
            self.all_time_actions = torch.zeros(
                self.max_timesteps,
                self.max_timesteps + self.chunk_size,
                action_dim,
                device=self.device,
            )
            self.all_time_mask = torch.zeros(
                self.max_timesteps,
                self.max_timesteps + self.chunk_size,
                dtype=torch.bool,
                device=self.device,
            )

        print(
            f"[tcla_raw_act] Loaded checkpoint={resolve_path(ckpt_path)} "
            f"task={self.task_name} cameras={self.camera_names} "
            f"chunk_size={self.chunk_size} temporal_agg={self.temporal_agg}"
        )

    def _validate_camera_names(self):
        model_camera_names = list(
            getattr(self.model.policy.model, "camera_names", []) or []
        )
        if not model_camera_names:
            return
        if self.camera_names != model_camera_names:
            raise ValueError(
                "camera_names must match the ACT checkpoint camera order: "
                f"got {self.camera_names}, but model expects {model_camera_names}. "
                "Use the training run config camera_names or override deploy "
                "camera_names with the same order."
            )

    @property
    def action_space(self) -> str:
        return str(self.cfg.framework.action_space).strip().lower()

    def reset(self):
        self.action_chunk = None
        self.chunk_index = 0
        self.t = 0
        if self.temporal_agg and self.all_time_actions is not None:
            self.all_time_actions.zero_()
            self.all_time_mask.zero_()

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
        )
        return unnormalize_action(normalized_chunk, self.action_stats)

    def _temporal_aggregate(self, step: int) -> np.ndarray:
        actions_for_curr_step = self.all_time_actions[:, step]
        actions_populated = self.all_time_mask[:, step]
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        if actions_for_curr_step.numel() == 0:
            return np.zeros((self.all_time_actions.shape[-1],), dtype=np.float32)
        exp_weights = np.exp(
            -self.temporal_agg_k * np.arange(len(actions_for_curr_step))
        )
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).to(
            device=actions_for_curr_step.device,
            dtype=actions_for_curr_step.dtype,
        )
        action = (actions_for_curr_step * exp_weights[:, None]).sum(dim=0)
        return action.detach().cpu().numpy()

    def get_action(self, obs):
        if obs is None:
            return None

        if (
            self.temporal_agg
            or self.action_chunk is None
            or self.chunk_index >= len(self.action_chunk)
        ):
            chunk = self._query_chunk(obs)
            if self.temporal_agg:
                step = self.t
                chunk_len = len(chunk)
                self.all_time_actions[step, step : step + chunk_len] = torch.from_numpy(
                    chunk
                ).to(self.device)
                self.all_time_mask[step, step : step + chunk_len] = True
                action = self._temporal_aggregate(step)
            else:
                self.action_chunk = chunk
                self.chunk_index = 0
                action = self.action_chunk[self.chunk_index]
                self.chunk_index += 1
        else:
            action = self.action_chunk[self.chunk_index]
            self.chunk_index += 1

        self.t += 1
        return np.asarray(action, dtype=np.float32)


def encode_obs(observation):
    obs_pack = observation["observation"]
    head_cam = obs_pack["head_camera"]["rgb"]
    left_cam = obs_pack["left_camera"]["rgb"]
    right_cam = obs_pack["right_camera"]["rgb"]
    qpos = (
        observation["joint_action"]["left_arm"]
        + [observation["joint_action"]["left_gripper"]]
        + observation["joint_action"]["right_arm"]
        + [observation["joint_action"]["right_gripper"]]
    )
    return {
        "head_cam": head_cam,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "qpos": qpos,
    }


def get_model(usr_args):
    return TCLARawACTModel(usr_args)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    actions = model.get_action(obs)
    if actions is None:
        return observation
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        if TASK_ENV.eval_success:
            break
    return observation


def reset_model(model):
    model.reset()
