import argparse
import os
import shlex
from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _absolute_path(value):
    path = Path(value).expanduser()
    return str(
        path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    )


def load_paths():
    config = OmegaConf.load(PROJECT_ROOT / "configs/paths.yaml")
    override = os.environ.get("TCLA_PATHS_CONFIG")
    local = (
        Path(_absolute_path(override))
        if override
        else PROJECT_ROOT / "configs/paths.local.yaml"
    )
    if override or local.is_file():
        config = OmegaConf.merge(config, OmegaConf.load(local))
    environment = {
        "DATA_ROOT": ("data.robotwin", "data.franka"),
        "ACTION_AE_RUN_DIR": ("action_ae.robotwin", "action_ae.franka"),
        "ROBOTWIN_ROOT": ("robotwin",),
        "RUN_DIR": ("policy_run",),
        "SENTENCE_ENCODER": ("language_model",),
        "ROBOTWIN_LANG_MODEL": ("language_model",),
        "ROBOTWIN_TASK_EMB_CACHE": ("task_embedding_cache",),
    }
    for variable, keys in environment.items():
        if os.environ.get(variable):
            for key in keys:
                OmegaConf.update(config.paths, key, os.environ[variable])
    for group in ("data", "output", "action_ae"):
        for key, value in config.paths[group].items():
            config.paths[group][key] = _absolute_path(value)
    for key in ("robotwin", "policy_run", "task_embedding_cache"):
        config.paths[key] = _absolute_path(config.paths[key])
    language_model = str(config.paths.language_model)
    if (
        language_model.startswith((".", "~"))
        or Path(language_model).is_absolute()
        or (PROJECT_ROOT / language_model).exists()
    ):
        config.paths.language_model = _absolute_path(language_model)
    return config


def load_training_config(path, dotlist=()):
    return OmegaConf.merge(
        load_paths(), OmegaConf.load(path), OmegaConf.from_dotlist(dotlist)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    cfg = load_paths()
    if args.shell:
        values = {
            "ROBOTWIN_ROOT": cfg.paths.robotwin,
            "RUN_DIR": cfg.paths.policy_run,
            "ROBOTWIN_DATA_ROOT": cfg.paths.data.robotwin,
            "ROBOTWIN_LANG_MODEL": cfg.paths.language_model,
            "ROBOTWIN_TASK_EMB_CACHE": cfg.paths.task_embedding_cache,
        }
        for key, value in values.items():
            print(f"export {key}={shlex.quote(str(value))}")
    else:
        print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
