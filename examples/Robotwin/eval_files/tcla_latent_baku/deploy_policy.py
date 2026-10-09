"""
RoboTwin deploy entry for TCLA latent Baku policies.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from .model_utils import (
    get_tcla_root,
    build_framework,
    build_traj_a_freq_from_executed,
    infer_phase_scalar,
    load_episode_task_emb,
    load_json,
    normalize_action,
    normalize_qpos,
    predict_action_chunk,
    resolve_path,
    save_phase_curve_plot,
    stack_images,
    unnormalize_action,
)


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class TCLALatentBakuModel:
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
        self.action_ae_stats = load_json(run_dir / "action_ae_statistics.json")

        self.camera_names = list(usr_args.get("camera_names") or self.cfg.camera_names)
        self.task_name = usr_args["task_name"]
        self.chunk_size = self._resolve_chunk_size(usr_args)
        self.temporal_agg = _as_bool(usr_args.get("temporal_agg", False), default=False)
        self.temporal_agg_k = float(usr_args.get("temporal_agg_k", 0.01))
        self.save_phase_plot = _as_bool(
            usr_args.get("save_phase_plot", True), default=True
        )
        self._validate_camera_names()

        lang_dim = 384
        action_ae_config = self.cfg.framework.get("action_ae_config")
        if action_ae_config:
            ae_cfg = OmegaConf.load(resolve_path(action_ae_config, get_tcla_root()))
            lang_dim = int(ae_cfg.framework.lang_dim)
        self.lang_dim = lang_dim
        self.task_emb = torch.from_numpy(
            load_episode_task_emb(
                self._resolve_policy_data_root(),
                self.task_name,
                0,
                dim=lang_dim,
            )
        ).float()

        self.action_chunk = None
        self.chunk_index = 0
        self.t = 0
        self.max_timesteps = int(usr_args.get("max_timesteps", 3000))
        self.all_time_actions = None
        self.all_time_mask = None
        self.executed_actions_norm = []
        self.policy_data_root = self._resolve_policy_data_root()
        self.video_save_dir = None
        self.current_episode_id = 0
        self.phase_steps: list[int] = []
        self.phase_values: list[float] = []
        self._phase_plot_saved_for_episode: Optional[int] = None

        if self.temporal_agg:
            action_dim = int(self.cfg.framework.baku_config.state_dim)
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
            f"[tcla_latent_baku] Loaded checkpoint={resolve_path(ckpt_path)} "
            f"task={self.task_name} cameras={self.camera_names} chunk_size={self.chunk_size}"
        )

    def _resolve_chunk_size(self, usr_args) -> int:
        ds = self.cfg.get("datasets", {}).get("action_data", {}).get("dataset", {})
        if ds.get("chunk_size") is not None:
            return int(ds["chunk_size"])
        ae_chunk = getattr(self.model.policy, "action_ae_chunk_size", None)
        if ae_chunk is not None:
            return int(ae_chunk)
        if usr_args.get("chunk_size") is not None:
            return int(usr_args["chunk_size"])
        return int(self.cfg.chunk_size)

    def _validate_camera_names(self):
        model_camera_names = list(
            getattr(self.model.policy.model, "camera_names", []) or []
        )
        if not model_camera_names:
            return
        if self.camera_names != model_camera_names:
            raise ValueError(
                "camera_names must match the Baku checkpoint camera order: "
                f"got {self.camera_names}, but model expects {model_camera_names}. "
                "Use the training run config camera_names or override deploy camera_names."
            )

    def _resolve_policy_data_root(self) -> Path:
        import os

        env_root = os.environ.get("ROBOTWIN_DATA_ROOT")
        if env_root:
            return resolve_path(env_root, get_tcla_root())
        ds = self.cfg.get("datasets", {}).get("action_data", {}).get("dataset", {})
        path = ds.get("path") or self.cfg.get("data_root")
        if path is None:
            raise ValueError(
                "config must set data_root or datasets.action_data.dataset.path"
            )
        return resolve_path(path, get_tcla_root())

    def set_eval_context(self, TASK_ENV):
        eval_video_path = getattr(TASK_ENV, "eval_video_path", None)
        self.video_save_dir = Path(eval_video_path) if eval_video_path else None
        self.current_episode_id = int(getattr(TASK_ENV, "test_num", 0))
        task_emb = load_episode_task_emb(
            self.policy_data_root,
            self.task_name,
            self.current_episode_id,
            dim=self.lang_dim,
        )
        self.task_emb = torch.from_numpy(task_emb).float()

    def _record_phase_step(self):
        if not self.save_phase_plot or self.model.policy.action_ae is None:
            return
        phase = infer_phase_scalar(
            self.model.policy,
            self.executed_actions_norm,
            action_ae_stats=self.action_ae_stats,
            task_emb=self.task_emb,
        )
        if phase is None:
            return
        self.phase_steps.append(self.t)
        self.phase_values.append(phase)

    def flush_phase_plot(self):
        if not self.save_phase_plot or not self.phase_steps:
            return
        if self.video_save_dir is None:
            return
        if self._phase_plot_saved_for_episode == self.current_episode_id:
            return

        episode_id = self.current_episode_id
        save_path = self.video_save_dir / f"episode{episode_id}_phase.png"
        save_phase_curve_plot(
            self.phase_steps,
            self.phase_values,
            save_path,
            episode_id=episode_id,
            task_name=self.task_name,
        )
        print(f"[tcla_latent_baku] Saved phase curve: {save_path}")
        self._phase_plot_saved_for_episode = episode_id
        self.phase_steps = []
        self.phase_values = []

    def reset(self):
        self.action_chunk = None
        self.chunk_index = 0
        self.t = 0
        self.executed_actions_norm = []
        self.phase_steps = []
        self.phase_values = []
        self._phase_plot_saved_for_episode = None
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
        traj_a_freq = build_traj_a_freq_from_executed(
            self.model.policy,
            self.executed_actions_norm,
            action_ae_stats=self.action_ae_stats,
        )
        normalized_chunk = predict_action_chunk(
            self.model,
            torch.from_numpy(qpos),
            image,
            self.task_emb,
            traj_a_freq=traj_a_freq,
            traj_a_len=len(self.executed_actions_norm),
        )
        return unnormalize_action(normalized_chunk, self.action_ae_stats)

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

        self._record_phase_step()

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
        action = np.asarray(action, dtype=np.float32)
        self.executed_actions_norm.append(
            normalize_action(action, self.action_ae_stats)
        )
        return action


def encode_obs(observation):
    obs_pack = observation["observation"]
    left_cam = obs_pack["left_camera"]["rgb"]
    right_cam = obs_pack["right_camera"]["rgb"]
    head_cam = obs_pack["head_camera"]["rgb"]
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
    return TCLALatentBakuModel(usr_args)


def eval(TASK_ENV, model, observation):
    model.set_eval_context(TASK_ENV)
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

    if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_success:
        model.flush_phase_plot()
    return observation


def reset_model(model):
    model.reset()
