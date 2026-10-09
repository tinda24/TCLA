# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from types import SimpleNamespace

import torch

from .models import build_ACT_model


def build_ACT_model_and_optimizer(args_override):
    config = dict(
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        backbone="resnet18",
        dilation=False,
        position_embedding="sine",
        camera_names=[],
        state_dim=14,
        action_dim=14,
        latent_dim=32,
        pretrained_backbone=False,
        enc_layers=4,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=400,
        pre_norm=False,
        masks=False,
    )
    config.update(args_override)
    args = SimpleNamespace(**config)
    model = build_ACT_model(args)
    groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    return model, torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
