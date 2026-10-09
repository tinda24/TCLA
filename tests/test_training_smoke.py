import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from tcla.configuration import PROJECT_ROOT, load_paths
from tcla.training import train_action_space as ae_train
from tcla.training import train_policy
from tcla.training.trainer_action_space import ActionSpaceTrainer
from tcla.training.trainer_policy import PolicyTrainer
from deployment.model_server.server_policy import load_tcla_policy
from deployment.model_server.tools import msgpack_numpy


class TrainingSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_data(self, action_dim, cameras):
        folder = self.root / f"data_{action_dim}" / "smoke_task"
        folder.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        for index in range(2):
            with h5py.File(folder / f"episode_{index}.hdf5", "w") as handle:
                handle["action"] = rng.normal(size=(70, action_dim)).astype(np.float32)
                handle["observations/qpos"] = rng.normal(size=(70, action_dim)).astype(
                    np.float32
                )
                handle["task_emb"] = rng.normal(size=384).astype(np.float32)
                for camera in cameras:
                    handle[f"observations/images/{camera}"] = rng.integers(
                        0, 256, (70, 32, 32, 3), dtype=np.uint8
                    )
        return folder.parent

    def make_ae(self, robot):
        franka = robot == "franka"
        path = PROJECT_ROOT / (
            "examples/franka/ae_train_files/train_ae.yaml"
            if franka
            else "examples/Robotwin_ae/train_files/train_ae.yaml"
        )
        cfg = ae_train.load_config(path, [])
        cfg.datasets.action_data.dataset.path = str(
            self.make_data(
                8 if franka else 14,
                ["front_cam", "wrist_cam"]
                if franka
                else ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            )
        )
        cfg.datasets.action_data.dataset.task_name = ["smoke_task"]
        cfg.datasets.action_data.dataset.num_episodes = 2
        if franka:
            cfg.datasets.action_data.dataset.task_emb_source = "hash"
        dataset, collate = ae_train.build_dataset(cfg, Accelerator(cpu=True))
        model = ae_train.build_model(cfg, "cpu")
        return cfg, dataset, collate, model

    def test_path_override_and_all_yaml(self):
        local = self.root / "paths.yaml"
        OmegaConf.save({"paths": {"data": {"franka": "data/custom"}}}, local)
        with patch.dict(os.environ, {"TCLA_PATHS_CONFIG": str(local)}, clear=True):
            paths = load_paths()
            self.assertEqual(
                Path(paths.paths.data.franka), PROJECT_ROOT / "data/custom"
            )
        for path in (PROJECT_ROOT / "examples").rglob("*.yaml"):
            cfg = OmegaConf.merge(load_paths(), OmegaConf.load(path))
            OmegaConf.resolve(cfg)
            self.assertIn(
                cfg.framework.name,
                {"ActionAutoencoder", "ACTPolicy", "BakuPolicy"},
            )

    def test_ae_forward_backward_for_both_robots(self):
        for robot in ("robotwin", "franka"):
            with self.subTest(robot=robot):
                cfg, dataset, _, model = self.make_ae(robot)
                batch = [next(iter(dataset)) for _ in range(2)]
                batch[0]["traj_a_len"] = torch.tensor(0)
                outputs = model(batch)
                self.assertTrue(torch.isfinite(outputs["total_loss"]))
                outputs["total_loss"].backward()
                self.assertIsNotNone(model.initial_traj_a_freq.grad)
                self.assertTrue(
                    all(p.grad is None for p in model.target_encoder.parameters())
                )
                self.assertTrue(
                    any(p.grad is not None for p in model.encoder.parameters())
                )
                self.assertTrue(
                    any(p.grad is not None for p in model.progress_model.parameters())
                )
                momentum = model.update_target_encoder(0, 2)
                self.assertEqual(momentum, cfg.framework.ema_momentum_start)

    def test_policies_data_training_inference_and_checkpoint(self):
        for robot in ("robotwin", "franka"):
            ae_cfg, ae_dataset, _, ae_model = self.make_ae(robot)
            ae_dir = self.root / f"ae_{robot}"
            ae_dir.mkdir()
            OmegaConf.save(ae_cfg, ae_dir / "config.yaml")
            torch.save({"model": ae_model.state_dict()}, ae_dir / "model.pt")
            (ae_dir / "dataset_statistics.json").write_text(
                json.dumps(
                    {key: value.tolist() for key, value in ae_dataset.stats.items()}
                ),
                encoding="utf-8",
            )
            for policy in ("act", "baku"):
                for space in ("raw", "latent"):
                    with self.subTest(robot=robot, policy=policy, space=space):
                        relative = (
                            f"examples/franka/policy_train_files/train_{space}_{policy}.yaml"
                            if robot == "franka"
                            else f"examples/Robotwin/train_files/train_{space}_{policy}_robotwin.yaml"
                        )
                        cfg = train_policy.load_config(
                            PROJECT_ROOT / relative,
                            [
                                "--action_ae.config_yaml",
                                str(ae_dir / "config.yaml"),
                                "--action_ae.ckpt_path",
                                str(ae_dir / "model.pt"),
                                "--action_ae.stats_path",
                                str(ae_dir / "dataset_statistics.json"),
                            ],
                        )
                        data_cfg = cfg.datasets.action_data.dataset
                        data_cfg.path = str(
                            self.root / ("data_8" if robot == "franka" else "data_14")
                        )
                        data_cfg.task_name = ["smoke_task"]
                        data_cfg.num_episodes = 2
                        if robot == "franka":
                            data_cfg.task_emb_source = "hash"
                        model_cfg = cfg.framework[f"{policy}_config"]
                        model_cfg.hidden_dim = 32
                        model_cfg.enc_layers = 1
                        model_cfg.dim_feedforward = 64
                        model_cfg.nheads = 4
                        if policy == "act":
                            model_cfg.dec_layers = 1
                        dataset, _ = train_policy.build_dataset(
                            cfg, Accelerator(cpu=True)
                        )
                        batch = [next(iter(dataset)) for _ in range(2)]
                        model = train_policy.build_model(cfg, "cpu")
                        model.train()
                        loss = model(batch)["action_loss"]
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        if space == "latent":
                            self.assertTrue(
                                all(
                                    p.grad is None
                                    for p in model.policy.action_ae.parameters()
                                )
                            )
                        model.eval()
                        predicted = model.predict_action(batch)["normalized_actions"]
                        self.assertEqual(
                            tuple(predicted.shape),
                            (2, cfg.chunk_size, 8 if robot == "franka" else 14),
                        )
                        run_dir = self.root / f"{robot}_{policy}_{space}"
                        (run_dir / "checkpoints").mkdir(parents=True)
                        checkpoint = run_dir / "checkpoints/policy.pt"
                        OmegaConf.save(cfg, run_dir / "config.yaml")
                        torch.save(
                            {"model": PolicyTrainer._model_state_dict(model)},
                            checkpoint,
                        )
                        clone = load_tcla_policy(checkpoint, "cpu").eval()
                        torch.testing.assert_close(
                            predicted, clone.predict_action(batch)["normalized_actions"]
                        )
                        packed = msgpack_numpy.packb({"actions": predicted})
                        unpacked = msgpack_numpy.unpackb(packed)
                        np.testing.assert_allclose(
                            unpacked["actions"], predicted.numpy()
                        )

    def test_train_eval_resume(self):
        cfg, dataset, collate, model = self.make_ae("robotwin")
        cfg.trainer.max_train_steps = 2
        cfg.trainer.save_every_steps = 2
        cfg.trainer.log_every_steps = 1
        cfg.trainer.train_mode_eval_every_steps = 1
        cfg.trainer.eval_loss_max_windows = 4
        cfg.trainer.eval_batch_size = 4
        optimizer = ae_train.build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        accelerator = Accelerator(cpu=True)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        run = self.root / "run"
        run.mkdir()
        trainer = ActionSpaceTrainer(
            cfg,
            model,
            DataLoader(dataset, batch_size=2, collate_fn=collate),
            optimizer,
            scheduler,
            accelerator,
            run,
            dataset,
        )
        training_modes = []
        hook = model.register_forward_pre_hook(
            lambda module, inputs: training_modes.append(module.training)
        )
        trainer.train()
        hook.remove()
        self.assertEqual(training_modes, [True, True])
        self.assertTrue((run / "progress_eval/eval_loss.csv").is_file())
        self.assertTrue(list((run / "latent_tsne_eval").rglob("*.png")))
        cfg.trainer.is_resume = True
        cfg.trainer.resume_checkpoint = str(
            run / "checkpoints/steps_2_pytorch_model.pt"
        )
        trainer.load_training_state_if_needed()
        self.assertEqual(trainer.completed_steps, 2)

    def test_eval_adapter_imports(self):
        import sys

        directory = str(PROJECT_ROOT / "examples/Robotwin/eval_files")
        with patch.object(sys, "path", [directory, *sys.path]):
            for name in (
                "tcla_latent_act",
                "tcla_raw_act",
                "tcla_latent_baku",
                "tcla_raw_baku",
            ):
                module = importlib.import_module(name)
                for command in ("get_model", "eval", "reset_model"):
                    self.assertTrue(callable(getattr(module, command)))


if __name__ == "__main__":
    unittest.main()
