def _get_module(model, path):
    for name in path.split("."):
        model = getattr(model, name)
    return model


def build_param_lr_groups(model, cfg):
    groups = []
    used = set()
    for path, lr in cfg.trainer.learning_rate.items():
        if path == "base":
            continue
        params = [
            p
            for p in _get_module(model, path).parameters()
            if p.requires_grad and id(p) not in used
        ]
        if params:
            groups.append({"params": params, "lr": float(lr), "name": path})
            used.update(id(p) for p in params)
    params = [p for p in model.parameters() if p.requires_grad and id(p) not in used]
    if params:
        groups.append(
            {
                "params": params,
                "lr": float(cfg.trainer.learning_rate.base),
                "name": "base",
            }
        )
    return groups


class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        for path in filter(None, (name.strip() for name in freeze_modules.split(","))):
            _get_module(model, path).requires_grad_(False)
        return model
