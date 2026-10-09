import time
from pathlib import Path

import torch
from tcla.training.action_ae_evaluator import ActionAETrainingEvaluator
from tcla.training.trainer_utils import TrainerUtils


def _cfg_flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


class ActionSpaceTrainer(TrainerUtils):
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
        self.train_mode_eval_every_steps = int(
            cfg.trainer.get("train_mode_eval_every_steps", 0)
        )
        self.evaluator = None
        if self.accelerator.is_main_process and self.train_mode_eval_every_steps > 0:
            self.evaluator = ActionAETrainingEvaluator(
                cfg=cfg,
                dataset=dataset,
                run_dir=self.run_dir,
                device=self.accelerator.device,
                batch_size=int(cfg.trainer.get("eval_batch_size", 256)),
                eval_loss_max_windows=cfg.trainer.get("eval_loss_max_windows", 128),
            )

        self.print_module_parameters(self.accelerator.unwrap_model(self.model))
        self.load_training_state_if_needed()
        self.save_initial_configs()
        self.save_dataset_statistics()

    def train(self):
        max_train_steps = int(self.cfg.trainer.max_train_steps)
        log_every_steps = int(self.cfg.trainer.log_every_steps)
        save_every_steps = int(self.cfg.trainer.save_every_steps)
        sync_cuda_timing = self.accelerator.device.type == "cuda" and _cfg_flag(
            self.cfg.trainer.get("sync_cuda_timing", True)
        )
        record_step_time = _cfg_flag(self.cfg.trainer.get("record_step_time", False))

        self.model.train()
        data_iter = iter(self.dataloader)

        print("----------------------------------")
        print("TRAIN ACTION AUTOENCODER IN ONE STAGE !")
        print(f"Max train steps: {max_train_steps}")
        print(f"Train-mode eval every steps: {self.train_mode_eval_every_steps}")

        last_saved_step = None
        step_time_file = None
        if record_step_time and self.accelerator.is_main_process:
            step_time_path = self.run_dir / "step_time.csv"
            step_time_file = open(step_time_path, "w", encoding="utf-8", newline="")
            step_time_file.write("step,step_time_s,data_time_s,compute_time_s\n")
            step_time_file.flush()
            self.accelerator.print(f"recording per-step time: {step_time_path}")

        try:
            for step in range(self.completed_steps, max_train_steps):
                data_t0 = time.perf_counter()
                examples = next(data_iter)
                if sync_cuda_timing:
                    torch.cuda.synchronize(self.accelerator.device)
                compute_t0 = time.perf_counter()

                with self.accelerator.accumulate(self.model):
                    outputs = self.model(examples)
                    loss = outputs["total_loss"]

                    self.optimizer.zero_grad(set_to_none=True)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    ema_momentum = unwrapped_model.update_target_encoder(
                        step, max_train_steps
                    )
                if sync_cuda_timing:
                    torch.cuda.synchronize(self.accelerator.device)
                compute_t1 = time.perf_counter()

                completed_step = step + 1
                if step_time_file is not None:
                    data_time_s = compute_t0 - data_t0
                    compute_time_s = compute_t1 - compute_t0
                    step_time_file.write(
                        f"{completed_step},{data_time_s + compute_time_s:.6f},"
                        f"{data_time_s:.6f},{compute_time_s:.6f}\n"
                    )
                    step_time_file.flush()

                if save_every_steps > 0 and completed_step % save_every_steps == 0:
                    self.accelerator.wait_for_everyone()
                    self.save_checkpoint(completed_step)
                    if sync_cuda_timing:
                        torch.cuda.synchronize(self.accelerator.device)
                    last_saved_step = completed_step

                if (
                    completed_step % log_every_steps == 0
                    or completed_step == max_train_steps
                ):
                    metrics = self._metrics_to_float(outputs)
                    metrics["ema_momentum"] = float(ema_momentum)
                    metrics["learning_rate"] = float(self.lr_scheduler.get_last_lr()[0])
                    metrics["epoch"] = (
                        round(completed_step / len(self.dataloader), 6)
                        if hasattr(self.dataloader, "__len__")
                        else 0.0
                    )
                    self.log_metrics(completed_step, metrics)

                if (
                    self.train_mode_eval_every_steps > 0
                    and completed_step % self.train_mode_eval_every_steps == 0
                ):
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        self.evaluator.run_train_mode_eval(
                            unwrapped_model, completed_step
                        )
                        self.model.train()
                    self.accelerator.wait_for_everyone()
        finally:
            if step_time_file is not None:
                step_time_file.close()

        self.accelerator.wait_for_everyone()
        if last_saved_step != max_train_steps:
            self.save_checkpoint(max_train_steps)

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
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        allowed_missing = {"initial_traj_a_freq"}
        bad_missing = [key for key in missing if key not in allowed_missing]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"Unexpected checkpoint keys while loading AE state: "
                f"missing={bad_missing}, unexpected={unexpected}"
            )
        if missing:
            self.accelerator.print(
                "resume checkpoint does not contain initial_traj_a_freq; "
                "using the newly initialized learned initial frequency."
            )
        try:
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        except ValueError as exc:
            if not missing:
                raise
            self.accelerator.print(
                "optimizer/lr_scheduler state is incompatible with the new "
                f"initial_traj_a_freq parameter; continuing with fresh optimizer state. ({exc})"
            )
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
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        checkpoint_state = {
            "completed_steps": int(step),
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        self.accelerator.save(checkpoint_state, checkpoint)
        self.write_summary(step)
        self.accelerator.print(f"saved checkpoint: {checkpoint}")
