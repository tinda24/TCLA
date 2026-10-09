"""RoboTwin / RMBench policy dataset aligned with ActionAutoencoder version3 stats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from tcla.dataloader.frequency_encoding import trajectory_to_frequency
from tcla.dataloader.robotwin import RoboTwinPolicyDataset


class RoboTwinPolicyVersion3Dataset(RoboTwinPolicyDataset):
    """Policy windows with mean/std action norm and DCT ``traj_a_freq`` freq-norm."""

    def __init__(self, *args, fft_type: str = "dct", **kwargs):
        self.fft_type = str(fft_type)
        super().__init__(*args, **kwargs)

    def _load_action_ae_stats(self) -> Dict[str, np.ndarray]:
        if not self.action_ae_stats_path:
            raise ValueError("version3 policy training requires action_ae_stats_path.")
        with open(self.action_ae_stats_path, "r", encoding="utf-8") as handle:
            stats = json.load(handle)
        if "action_mean" not in stats or "action_std" not in stats:
            raise KeyError(
                f"version3 policy expects action_mean/action_std in {self.action_ae_stats_path}"
            )
        loaded = {
            "action_mean": np.asarray(stats["action_mean"], dtype=np.float32),
            "action_std": np.asarray(stats["action_std"], dtype=np.float32),
        }
        if "traj_a_freq_mean" in stats and "traj_a_freq_std" in stats:
            loaded["traj_a_freq_mean"] = np.asarray(
                stats["traj_a_freq_mean"], dtype=np.float32
            )
            loaded["traj_a_freq_std"] = np.asarray(
                stats["traj_a_freq_std"], dtype=np.float32
            )
        return loaded

    def _preprocess_action(self, action: np.ndarray) -> np.ndarray:
        mean = self.action_ae_stats["action_mean"]
        std = self.action_ae_stats["action_std"]
        return ((action - mean) / std).astype(np.float32)

    def _preprocess_traj_a_freq(self, freq: np.ndarray) -> np.ndarray:
        if "traj_a_freq_mean" not in self.action_ae_stats:
            return freq.astype(np.float32)
        mean = self.action_ae_stats["traj_a_freq_mean"]
        std = self.action_ae_stats["traj_a_freq_std"]
        return ((freq - mean) / std).astype(np.float32)

    def _precompute_traj_a_freq_by_prefix(self, action: np.ndarray) -> np.ndarray:
        action_norm = self._preprocess_action(action)
        episode_len = action_norm.shape[0]
        cached = []
        for prefix_len in range(1, episode_len + 1):
            prefix = torch.from_numpy(action_norm[:prefix_len])
            freq = trajectory_to_frequency(
                prefix,
                num_freqs=self.num_freqs,
                fft_type=self.fft_type,
            ).numpy()
            cached.append(self._preprocess_traj_a_freq(freq))
        return np.stack(cached, axis=0)


def load_num_freqs_from_ae_config(config_yaml: str | Path) -> int:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(config_yaml))
    return int(cfg.framework.num_freqs)
