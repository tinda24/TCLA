import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

from tcla.dataloader.frequency_encoding import trajectory_to_frequency


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class RoboTwinPolicyDataset(IterableDataset):
    """Randomly samples policy training windows from RoboTwin/RMBench hdf5 episodes."""

    def __init__(
        self,
        path: str,
        task_name: Union[str, Sequence[str]] = "adjust_bottle",
        camera_names: Optional[Sequence[str]] = None,
        chunk_size: int = 8,
        action_ae_stats_path: Optional[str] = None,
        num_freqs: int = 256,
        num_episodes: Optional[int] = 50,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        super().__init__()
        self.path = Path(path)
        self.task_names = [task_name] if isinstance(task_name, str) else list(task_name)
        if not self.task_names:
            raise ValueError("task_name must contain at least one task.")
        self.camera_names = list(camera_names or ["cam_left_wrist", "cam_right_wrist"])
        self.chunk_size = int(chunk_size)
        self.action_ae_stats_path = action_ae_stats_path
        self.num_freqs = int(num_freqs)
        self.num_episodes = num_episodes
        self.rank = rank
        self.world_size = world_size
        self.seed = seed + rank

        self.all_episode_paths = self._load_episode_paths()
        self.episode_paths = self.all_episode_paths[self.rank :: self.world_size]
        if not self.episode_paths:
            raise RuntimeError("No RoboTwin policy episodes are assigned to this rank.")

        self.policy_stats = self._compute_policy_stats()
        self.action_ae_stats = self._load_action_ae_stats()
        self._stats = {
            "policy_stats": self.policy_stats,
            "action_ae_stats": self.action_ae_stats,
        }
        self._rng = random.Random(self.seed)
        self._episodes = self._load_all_episodes()
        print(
            f"Loaded {len(self._episodes)} RoboTwin policy episodes from "
            f"{self.task_names} in memory on rank {self.rank}"
        )

    def _task_episode_paths(self, task_name: str):
        task_dir = self.path / task_name
        if task_dir.exists():
            return self._sorted_episode_paths(task_dir)

        direct_paths = self._sorted_episode_paths(self.path)
        if direct_paths:
            return direct_paths

        raise FileNotFoundError(
            f"RoboTwin task directory does not exist and no direct episodes were found: {task_dir}"
        )

    @staticmethod
    def _sorted_episode_paths(directory: Path):
        def episode_index(path: Path):
            try:
                return int(path.stem.split("_")[-1])
            except ValueError:
                return path.stem

        return sorted(directory.glob("episode_*.hdf5"), key=episode_index)

    def _load_episode_paths(self):
        paths = []
        seen = set()
        for task_name in self.task_names:
            task_paths = self._task_episode_paths(task_name)
            if self.num_episodes is not None:
                task_paths = task_paths[: self.num_episodes]
            for path in task_paths:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(path)
        if not paths:
            raise RuntimeError("No RoboTwin policy episodes found.")
        return paths

    def _load_action_ae_stats(self):
        if self.action_ae_stats_path:
            with open(self.action_ae_stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            return {
                "action_min": np.asarray(stats["action_min"], dtype=np.float32),
                "action_max": np.asarray(stats["action_max"], dtype=np.float32),
            }

        all_actions = []
        for path in self.all_episode_paths:
            with h5py.File(path, "r") as f:
                all_actions.append(f["action"][:].astype(np.float32))
        all_actions = np.concatenate(all_actions, axis=0)
        return {
            "action_min": np.min(all_actions, axis=0),
            "action_max": np.max(all_actions, axis=0),
        }

    def _compute_policy_stats(self) -> Dict[str, np.ndarray]:
        all_qpos = []
        for path in self.all_episode_paths:
            with h5py.File(path, "r") as f:
                all_qpos.append(f["observations/qpos"][:].astype(np.float32))
        all_qpos = np.concatenate(all_qpos, axis=0)
        qpos_std = np.clip(np.std(all_qpos, axis=0), 1e-2, 10.0)
        return {
            "qpos_mean": np.mean(all_qpos, axis=0),
            "qpos_std": qpos_std,
        }

    @property
    def stats(self):
        return self._stats

    @staticmethod
    def _read_hdf5_string(value):
        if hasattr(value, "__getitem__"):
            value = value[()]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _task_name_embedding(task_name: str, dim: int = 384) -> np.ndarray:
        digest = hashlib.sha256(task_name.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal(dim).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-6)

    @staticmethod
    def _available_camera_names(handle):
        image_group = handle.get("observations/images")
        if image_group is None:
            return []
        return sorted(image_group.keys())

    def _load_episode(self, path: Path):
        episode = {}
        with h5py.File(path, "r") as f:
            task_name = (
                self._read_hdf5_string(f["task_name"])
                if "task_name" in f
                else path.parent.name
            )
            episode["qpos"] = f["observations/qpos"][:].astype(np.float32)
            episode["action"] = f["action"][:].astype(np.float32)

            available_cameras = self._available_camera_names(f)
            missing_cameras = [
                name for name in self.camera_names if name not in available_cameras
            ]
            if missing_cameras:
                raise KeyError(
                    f"Missing camera(s) {missing_cameras} in {path}. "
                    f"Available cameras: {available_cameras}"
                )
            episode["images"] = {
                camera_name: f[f"observations/images/{camera_name}"][:]
                for camera_name in self.camera_names
            }

            if "task_emb" in f:
                task_emb = f["task_emb"][:].astype(np.float32)
                if task_emb.ndim > 1:
                    task_emb = task_emb[0]
                episode["task_emb"] = task_emb
            else:
                episode["task_emb"] = self._task_name_embedding(task_name)
            episode["task_name"] = task_name
            episode["__path__"] = str(path)
        return self._attach_traj_a_freq_cache(episode)

    def _precompute_traj_a_freq_by_prefix(self, action: np.ndarray) -> np.ndarray:
        """For prefix lengths 1..T, cache trajectory_to_frequency on normalized actions."""
        action_norm = self._preprocess_action(action)
        episode_len = action_norm.shape[0]
        cached = []
        for prefix_len in range(1, episode_len + 1):
            prefix = torch.from_numpy(action_norm[:prefix_len])
            cached.append(
                trajectory_to_frequency(prefix, num_freqs=self.num_freqs).numpy()
            )
        return np.stack(cached, axis=0)

    def _attach_traj_a_freq_cache(self, episode: dict) -> dict:
        episode["traj_a_freq_by_prefix"] = self._precompute_traj_a_freq_by_prefix(
            episode["action"]
        )
        return episode

    def _load_all_episodes(self):
        episodes = []
        for path in tqdm(
            self.episode_paths,
            desc=f"Loading RoboTwin policy episodes rank{self.rank} (incl. traj_a_freq)",
        ):
            episodes.append(self._load_episode(path))
        if not episodes:
            raise RuntimeError("No RoboTwin policy episodes loaded into memory.")
        return episodes

    def _preprocess_qpos(self, qpos: np.ndarray) -> np.ndarray:
        return (qpos - self.policy_stats["qpos_mean"]) / self.policy_stats["qpos_std"]

    def _preprocess_action(self, action: np.ndarray) -> np.ndarray:
        action_min = self.action_ae_stats["action_min"]
        action_max = self.action_ae_stats["action_max"]
        return 2.0 * (action - action_min) / (action_max - action_min + 1e-5) - 1.0

    @staticmethod
    def _slice_with_padding(array, start, length):
        if start >= array.shape[0]:
            return np.repeat(array[-1:], length, axis=0)
        buffer_start = max(start, 0)
        buffer_end = min(start + length, array.shape[0])
        sample_start = buffer_start - start
        sample_end = sample_start + (buffer_end - buffer_start)
        out = np.empty((length, *array.shape[1:]), dtype=array.dtype)
        out[sample_start:sample_end] = array[buffer_start:buffer_end]
        if sample_start > 0:
            out[:sample_start] = array[0]
        if sample_end < length:
            out[sample_end:] = array[-1]
        return out

    def _sample_start(self, episode_len: int) -> int:
        return self._rng.randint(0, episode_len - 1)

    def _sample(self):
        episode = self._rng.choice(self._episodes)
        episode_len = episode["action"].shape[0]
        start = self._sample_start(episode_len)

        images = np.stack(
            [
                episode["images"][camera_name][start]
                for camera_name in self.camera_names
            ],
            axis=0,
        ).astype(np.float32)
        if images.ndim != 4:
            raise ValueError(
                f"Expected stacked images with shape [N, H, W, C], got {images.shape}."
            )
        images = np.transpose(images, (0, 3, 1, 2)) / 255.0

        raw_action = self._slice_with_padding(episode["action"], start, self.chunk_size)
        action = self._preprocess_action(raw_action).astype(np.float32)
        qpos = self._preprocess_qpos(episode["qpos"][start]).astype(np.float32)
        is_pad = np.zeros((self.chunk_size,), dtype=bool)

        prefix_len_a = start
        freq_index = max(prefix_len_a - 1, 0)

        return {
            "image": torch.from_numpy(images),
            "qpos": torch.from_numpy(qpos),
            "action": torch.from_numpy(action),
            "raw_action": torch.from_numpy(raw_action.astype(np.float32)),
            "is_pad": torch.from_numpy(is_pad),
            "task_emb": torch.from_numpy(episode["task_emb"]),
            "task_name": episode["task_name"],
            "traj_a_freq": torch.from_numpy(
                episode["traj_a_freq_by_prefix"][freq_index]
            ),
            "traj_a_len": torch.tensor(prefix_len_a, dtype=torch.long),
            "total_length": torch.tensor(episode_len, dtype=torch.long),
            "episode_start": torch.tensor(start, dtype=torch.long),
            "episode_path": episode["__path__"],
        }

    def __iter__(self) -> Iterable[dict]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self._rng = random.Random(self.seed + worker_info.id * 1000003)
        while True:
            yield self._sample()

    def __len__(self):
        total = 0
        for path in self.episode_paths:
            with h5py.File(path, "r") as f:
                total += int(f["action"].shape[0])
        return total


def collate_fn(batch):
    return batch
