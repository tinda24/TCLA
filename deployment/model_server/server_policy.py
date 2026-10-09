import argparse
import logging
import socket
from pathlib import Path

import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from tcla.model.modules.act.policy import ACTPolicyModule
from tcla.training.train_policy import build_model
from tcla.configuration import load_paths


def load_tcla_policy(ckpt_path: str, device: str = "cuda"):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_dir = ckpt_path.parents[1]
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing config.yaml next to checkpoint run dir: {config_path}"
        )

    cfg = OmegaConf.load(config_path)
    model = build_model(cfg, device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = ACTPolicyModule.extract_framework_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return model


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = args.ckpt_path or (
        Path(load_paths().paths.policy_run)
        / "checkpoints"
        / f"steps_{args.step}_pytorch_model.pt"
    )
    policy = load_tcla_policy(checkpoint, device=device)
    if args.use_bf16:
        policy = policy.to(torch.bfloat16)
    policy = policy.to(device).eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating TCLA policy server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"framework": "tcla"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--step", type=int, default=160000)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument(
        "--idle_timeout",
        type=int,
        default=1800,
        help="Idle timeout in seconds, -1 means never close",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
