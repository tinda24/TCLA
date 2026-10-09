from pathlib import Path
from contextlib import nullcontext
import json
import shutil


import torch

from tcla.training.trainer_utils import TrainerUtils, json_default


class PolicyTrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        run_dir,
        dataset,
    ):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.run_dir = Path(run_dir)
        self.dataset = dataset
        self.completed_steps = 0

        self.print_module_parameters(self.accelerator.unwrap_model(self.model))
        self.load_training_state_if_needed()
        self.save_initial_configs()
        self.save_policy_statistics()

    def train(self):
        max_train_steps = int(self.cfg.trainer.max_train_steps)
        log_every_steps = int(self.cfg.trainer.log_every_steps)
        save_every_steps = int(self.cfg.trainer.save_every_steps)
        csv_every_steps = log_every_steps

        if self.accelerator.is_main_process:
            self.accelerator.print(
                f"training metrics will be saved to csv: {self.metrics_csv_path} "
                f"(every {csv_every_steps} step(s))"
            )

        self.model.train()
        data_iter = iter(self.dataloader)

        last_saved_step = None
        use_deepspeed = (
            str(self.accelerator.distributed_type).upper().endswith("DEEPSPEED")
        )
        gradient_accumulation_steps = max(
            1, int(self.cfg.trainer.get("gradient_accumulation_steps", 1))
        )
        micro_steps = 0

        while self.completed_steps < max_train_steps:
            try:
                examples = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                examples = next(data_iter)

            train_context = (
                nullcontext()
                if use_deepspeed
                else self.accelerator.accumulate(self.model)
            )
            with train_context:
                outputs = self.model(examples)
                loss = outputs["action_loss"]

                self.optimizer.zero_grad(set_to_none=True)
                self.accelerator.backward(loss)
                self.optimizer.step()

            micro_steps += 1

            if use_deepspeed:
                is_optimization_step = micro_steps % gradient_accumulation_steps == 0
            else:
                is_optimization_step = self.accelerator.sync_gradients

            if not is_optimization_step:
                del outputs, loss, examples
                continue

            self.lr_scheduler.step()
            self.completed_steps += 1
            completed_step = self.completed_steps

            should_record_csv = csv_every_steps > 0 and (
                completed_step % csv_every_steps == 0
                or completed_step == max_train_steps
            )
            should_log = (
                completed_step % log_every_steps == 0
                or completed_step == max_train_steps
            )
            if should_record_csv or should_log:
                metrics = self._metrics_to_float(outputs)
                metrics["learning_rate"] = float(self.lr_scheduler.get_last_lr()[0])
                metrics["epoch"] = (
                    round(completed_step / len(self.dataloader), 6)
                    if hasattr(self.dataloader, "__len__")
                    else 0.0
                )
                if should_record_csv:
                    self.record_metrics(completed_step, metrics)
                if should_log:
                    self.log_metrics(
                        completed_step, metrics, record=not should_record_csv
                    )
            if save_every_steps > 0 and completed_step % save_every_steps == 0:
                self.accelerator.wait_for_everyone()
                self.save_checkpoint(completed_step)
                last_saved_step = completed_step
            del outputs, loss, examples

        self.accelerator.wait_for_everyone()
        if last_saved_step != max_train_steps:
            self.save_checkpoint(max_train_steps)
        csv_path = self.export_metrics_csv()
        self.plot_metrics()
        if self.accelerator.is_main_process and csv_path is not None:
            self.accelerator.print(f"final training metrics csv: {csv_path}")

    def plot_metrics(self):
        metrics_path = self.run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            return

        records = []
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if len(records) < 2:
            return

        imgs_dir = self.run_dir / "imgs"
        imgs_dir.mkdir(exist_ok=True)
        steps = [record["step"] for record in records]

        self._plot_metric_group(
            imgs_dir / "loss_curve.png",
            steps,
            records,
            [
                "action_loss",
                "flow_loss",
                "l1_loss",
                "decoded_action_l1_loss",
                "latent_l1_loss",
                "kl_loss",
            ],
            "Policy Training Loss",
            "Loss",
        )

    def load_training_state_if_needed(self):
        if not self.cfg.trainer.get("is_resume", False):
            return
        state_path = self.cfg.trainer.get("resume_checkpoint", None)
        if not state_path:
            raise ValueError(
                "trainer.resume_checkpoint must be set when trainer.is_resume is true."
            )
        state = torch.load(state_path, map_location="cpu")
        model = self.accelerator.unwrap_model(self.model)
        if isinstance(state, dict) and state.get("format") == "framework_state_dict_v1":
            self._load_model_state(model, state)
        else:
            self._load_model_state(model, state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.completed_steps = int(state["completed_steps"])
        self.accelerator.print(
            f"resumed training state: {state_path}, completed_steps={self.completed_steps}"
        )

    def save_checkpoint(self, step):
        checkpoint = (
            Path(self.run_dir) / "checkpoints" / f"steps_{step}_pytorch_model.pt"
        )
        if not self.accelerator.is_main_process:
            return
        model = self.accelerator.unwrap_model(self.model)
        state_dict = self.accelerator.get_state_dict(self.model)
        checkpoint_state = {
            "completed_steps": int(step),
            "format": "framework_state_dict_v1",
            "framework_name": type(model).__name__,
            "action_space": getattr(model, "action_space", None),
            "state_dict": state_dict,
        }
        if self.cfg.framework.get("action_ae_config"):
            checkpoint_state["resolved_action_ae_config"] = str(
                self.cfg.framework.action_ae_config
            )
        if self.cfg.framework.get("action_ae_ckpt"):
            checkpoint_state["resolved_action_ae_ckpt"] = str(
                self.cfg.framework.action_ae_ckpt
            )
        self.accelerator.save(checkpoint_state, checkpoint)
        if (
            self.accelerator.is_main_process
            and checkpoint_state.get("resolved_action_ae_config")
            and checkpoint_state.get("resolved_action_ae_ckpt")
        ):
            sidecar = self.run_dir / "action_ae_resolved.json"
            with open(sidecar, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "action_ae_config": checkpoint_state[
                            "resolved_action_ae_config"
                        ],
                        "action_ae_ckpt": checkpoint_state["resolved_action_ae_ckpt"],
                    },
                    handle,
                    indent=2,
                )
        self.write_summary(step)
        self.accelerator.print(f"saved checkpoint: {checkpoint}")

    def save_policy_statistics(self):
        if not self.accelerator.is_main_process:
            return
        policy_stats_path = self.run_dir / "policy_statistics.json"
        action_ae_stats_path = self.run_dir / "action_ae_statistics.json"

        stats = getattr(self.dataset, "stats", {}) or {}
        policy_stats = stats.get("policy_stats", stats)
        if hasattr(self.dataset, "save_dataset_statistics"):
            self.dataset.save_dataset_statistics(policy_stats_path)
        else:
            with open(policy_stats_path, "w") as f:
                json.dump(policy_stats, f, indent=2, default=json_default)

        action_ae_stats = stats.get("action_ae_stats", {})
        action_ae_config = (
            self.cfg.framework.get("action_ae_config", None)
            if "framework" in self.cfg
            else None
        )
        if action_ae_config:
            source_stats_path = (
                Path(action_ae_config).parent / "dataset_statistics.json"
            )
            if source_stats_path.is_file():
                shutil.copyfile(source_stats_path, action_ae_stats_path)
                return
        with open(action_ae_stats_path, "w") as f:
            json.dump(action_ae_stats, f, indent=2, default=json_default)

    @staticmethod
    def _model_state_dict(model):
        return {
            "format": "framework_state_dict_v1",
            "framework_name": type(model).__name__,
            "action_space": getattr(model, "action_space", None),
            "state_dict": model.state_dict(),
        }

    @staticmethod
    def _load_model_state(model, state_dict):
        if (
            isinstance(state_dict, dict)
            and state_dict.get("format") == "framework_state_dict_v1"
        ):
            model.load_state_dict(state_dict["state_dict"], strict=True)
            return
        if isinstance(state_dict, dict) and state_dict.get("format") == "act_policy_v1":
            model.policy.model.load_state_dict(state_dict["act_model"], strict=True)
            return
        if (
            not isinstance(state_dict, dict)
            or state_dict.get("format") != "act_policy_v1"
        ):
            raise ValueError(
                "Only framework_state_dict_v1 and legacy act_policy_v1 checkpoints are supported."
            )

    @staticmethod
    def _metrics_to_float(outputs):
        metrics = {}
        metric_keys = [
            "action_loss",
            "flow_loss",
            "l1_loss",
            "decoded_action_l1_loss",
            "kl_loss",
        ]
        for key in metric_keys:
            if key not in outputs:
                continue
            value = outputs[key]
            metrics[key] = (
                float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            )
        return metrics
