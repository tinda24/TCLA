import argparse
import os
from pathlib import Path

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(name, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tcla.configuration import load_training_config

from tcla.dataloader.franka import FrankaPolicyDataset, FrankaPolicyVersion3Dataset
from tcla.dataloader.robotwin import RoboTwinPolicyDataset, collate_fn
from tcla.dataloader.robotwin_policy_version3 import RoboTwinPolicyVersion3Dataset
from tcla.model.framework import act_policy, baku_policy  # noqa: F401
from tcla.model.tools import FRAMEWORK_REGISTRY
from tcla.training.trainer_policy import PolicyTrainer
from tcla.training.trainer_utils import make_run_dir, set_seed
from tcla.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
)


DATASET_REGISTRY = {
    "robotwin_policy": (RoboTwinPolicyDataset, collate_fn),
    "robotwin_policy_version3": (RoboTwinPolicyVersion3Dataset, collate_fn),
    "franka_policy": (FrankaPolicyDataset, collate_fn),
    "franka_policy_version3": (FrankaPolicyVersion3Dataset, collate_fn),
}


def normalize_overrides(overrides):
    dotlist = []
    index = 0
    while index < len(overrides):
        item = overrides[index].lstrip("-")
        if "=" in item:
            dotlist.append(item)
            index += 1
        else:
            if index + 1 >= len(overrides):
                raise ValueError(f"Missing value for override: {overrides[index]}")
            dotlist.append(f"{item}={overrides[index + 1]}")
            index += 2
    return dotlist


def load_config(config_yaml, overrides):
    cfg = load_training_config(config_yaml, normalize_overrides(overrides))
    if cfg.framework.action_space == "latent":
        ae = OmegaConf.load(cfg.framework.action_ae_config).framework
        if ae.name != "ActionAutoencoder_version3":
            raise ValueError(
                "Latent policies require an AE v3 checkpoint and its saved config.yaml."
            )
        cfg.chunk_size = int(ae.chunk_size)
        cfg.latent_action_dim = int(ae.latent_action_dim)
        cfg.latent_action_tokens = int(ae.chunk_size) // int(ae.downsample_rate)
        cfg.datasets.action_data.dataset.num_freqs = int(ae.num_freqs)
        cfg.datasets.action_data.dataset.fft_type = str(ae.fft_type)
    OmegaConf.resolve(cfg)
    return cfg


def build_dataset(cfg, accelerator):
    dataset_cfg = dict(cfg.datasets.action_data.dataset)
    dataset_cfg.update(
        rank=accelerator.process_index, world_size=accelerator.num_processes
    )
    name = cfg.datasets.action_data.dataset_py
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset_py: {name}. Available: {', '.join(DATASET_REGISTRY)}"
        )
    dataset_class, collate = DATASET_REGISTRY[name]
    return dataset_class(**dataset_cfg), collate


def build_model(cfg, device):
    model_cfg = OmegaConf.to_container(cfg.framework, resolve=True)
    name = model_cfg.pop("name")
    if name not in {"ACTPolicy", "BakuPolicy"}:
        raise ValueError(f"Unsupported policy: {name}")
    model_cfg["device"] = str(device)
    for key in ("decoded_loss_coef", "kl_weight", "latent_loss_coef", "raw_loss_coef"):
        if key in model_cfg:
            model_cfg[key] = float(model_cfg[key])
    return FRAMEWORK_REGISTRY[name](**model_cfg)


def build_optimizer(model, cfg):
    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if freeze_modules:
        TrainerUtils.freeze_backbones(model, freeze_modules=freeze_modules)
    groups = build_param_lr_groups(model, cfg)
    return torch.optim.AdamW(
        groups or [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.trainer.learning_rate.base),
        betas=tuple(cfg.trainer.optimizer.betas),
        eps=float(cfg.trainer.optimizer.eps),
        weight_decay=float(cfg.trainer.optimizer.weight_decay),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config_yaml, overrides)
    accelerator = Accelerator(
        mixed_precision=cfg.accelerator.get("mixed_precision", "no"),
        gradient_accumulation_steps=int(
            cfg.accelerator.get("gradient_accumulation_steps", 1)
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_seed(int(cfg.seed) + accelerator.process_index)
    resume = (
        cfg.trainer.get("resume_checkpoint")
        if cfg.trainer.get("is_resume", False)
        else None
    )
    run_dir = make_run_dir(
        cfg.run_root_dir,
        cfg.run_name,
        accelerator,
        resume_dir=str(Path(resume).parents[1]) if resume else None,
    )
    dataset, collate = build_dataset(cfg, accelerator)
    data_cfg = cfg.datasets.action_data
    dataloader = DataLoader(
        dataset,
        batch_size=int(data_cfg.per_device_batch_size),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", accelerator.device.type == "cuda")),
        collate_fn=collate,
    )
    model = build_model(cfg, accelerator.device)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    PolicyTrainer(
        cfg=cfg,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        accelerator=accelerator,
        run_dir=run_dir,
        dataset=dataset,
    ).train()


if __name__ == "__main__":
    main()
