import importlib.util
import os
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    tcla_root = Path(os.environ.get("TCLA_ROOT", script_dir.parents[2])).resolve()
    robotwin_root = Path(os.environ["ROBOTWIN_ROOT"]).resolve()
    eval_path = robotwin_root / "script" / "eval_policy.py"
    if not eval_path.is_file():
        raise FileNotFoundError(f"RoboTwin eval_policy.py not found: {eval_path}")
    sys.path[:0] = [
        str(script_dir),
        str(robotwin_root / "policy"),
        str(robotwin_root),
        str(tcla_root),
    ]
    spec = importlib.util.spec_from_file_location("robotwin_eval_policy", eval_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main(module.parse_args_and_config())


if __name__ == "__main__":
    main()
