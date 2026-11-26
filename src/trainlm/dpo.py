"""Direct Preference Optimization (DPO) support for TrainLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class DPOConfig:
    """Configuration for reference-policy Direct Preference Optimization."""

    beta: float = 0.1
    max_length: int = 4096
    max_prompt_length: int = 2048
    label_pad_token_id: int = -100
    learning_rate: float = 5e-7
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 20
    gradient_accumulation_steps: int = 1
    bf16: bool = True

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError("DPO beta must be positive.")
        if self.max_length < 2:
            raise ValueError("max_length must be >= 2.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1.")


def compute_sequence_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_pad_token_id: int = -100,
) -> torch.Tensor:
    """Sum per-token log probabilities over non-masked target positions."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"Logits shape {tuple(logits.shape)} must match labels shape {tuple(labels.shape)}."
        )
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    loss_mask = shift_labels != label_pad_token_id
    safe_labels = torch.where(loss_mask, shift_labels, 0)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(
        log_probs, dim=2, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (per_token_logps * loss_mask).sum(dim=-1)


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute standard sigmoid DPO loss and detached implicit rewards."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return loss, chosen_rewards, rejected_rewards


def _render_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_fn):
        try:
            rendered = apply_fn(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
            if isinstance(rendered, list):
                return [int(x) for x in rendered]
            if hasattr(rendered, "input_ids"):
                return [int(x) for x in rendered["input_ids"]]
        except Exception:
            pass
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return [int(x) for x in tokenizer.encode("".join(parts), add_special_tokens=False)]


def format_dpo_pair(
    prompt: Any,
    chosen: Any,
    rejected: Any,
    tokenizer: Any,
    *,
    max_length: int = 4096,
    eos_token_id: int = 0,
    label_pad_token_id: int = -100,
) -> dict[str, list[int]]:
    """Tokenize one preference pair with prompt-token label masking."""
    if isinstance(prompt, list):
        prompt_ids = _render_messages(tokenizer, prompt, add_generation_prompt=True)
    else:
        prompt_ids = [int(x) for x in tokenizer.encode(str(prompt), add_special_tokens=False)]

    def _encode_response(resp: Any) -> list[int]:
        if isinstance(resp, list):
            full = _render_messages(
                tokenizer,
                (prompt if isinstance(prompt, list) else []) + resp,
                add_generation_prompt=False,
            )
            if len(full) > len(prompt_ids) and full[: len(prompt_ids)] == prompt_ids:
                ids = full[len(prompt_ids) :]
            else:
                ids = _render_messages(tokenizer, resp, add_generation_prompt=False)
        else:
            ids = [int(x) for x in tokenizer.encode(str(resp), add_special_tokens=False)]
        if not ids or ids[-1] != eos_token_id:
            ids = [*ids, eos_token_id]
        return ids

    chosen_resp_ids = _encode_response(chosen)
    rejected_resp_ids = _encode_response(rejected)

    def _pack(resp_ids: list[int]) -> tuple[list[int], list[int]]:
        if len(prompt_ids) + len(resp_ids) > max_length:
            keep_prompt = max(0, max_length - len(resp_ids))
            p_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
            r_ids = resp_ids[: max_length - len(p_ids)]
        else:
            p_ids = prompt_ids
            r_ids = resp_ids
        input_ids = p_ids + r_ids
        labels = [label_pad_token_id] * len(p_ids) + r_ids
        return input_ids, labels

    chosen_input_ids, chosen_labels = _pack(chosen_resp_ids)
    rejected_input_ids, rejected_labels = _pack(rejected_resp_ids)
    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input_ids,
        "rejected_labels": rejected_labels,
    }


def _collate_dpo_batch(
    features: list[dict[str, list[int]]],
    *,
    pad_token_id: int,
    label_pad_token_id: int = -100,
) -> dict[str, torch.Tensor]:
    max_chosen = max(len(f["chosen_input_ids"]) for f in features)
    max_rejected = max(len(f["rejected_input_ids"]) for f in features)

    def _pad_pair(key_ids: str, key_labels: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids_rows, lbl_rows, mask_rows = [], [], []
        for f in features:
            ids = f[key_ids]
            lbls = f[key_labels]
            pad = max_len - len(ids)
            ids_rows.append(ids + [pad_token_id] * pad)
            lbl_rows.append(lbls + [label_pad_token_id] * pad)
            mask_rows.append([1] * len(ids) + [0] * pad)
        return (
            torch.tensor(ids_rows, dtype=torch.long),
            torch.tensor(lbl_rows, dtype=torch.long),
            torch.tensor(mask_rows, dtype=torch.long),
        )

    c_ids, c_lbl, c_mask = _pad_pair("chosen_input_ids", "chosen_labels", max_chosen)
    r_ids, r_lbl, r_mask = _pad_pair("rejected_input_ids", "rejected_labels", max_rejected)
    return {
        "chosen_input_ids": c_ids,
        "chosen_labels": c_lbl,
        "chosen_attention_mask": c_mask,
        "rejected_input_ids": r_ids,
        "rejected_labels": r_lbl,
        "rejected_attention_mask": r_mask,
    }


def _atomic_save_checkpoint(
    checkpoint_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    dataset_index: int,
    config: DPOConfig,
) -> Path:
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = checkpoint_dir.parent / f".{checkpoint_dir.name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(tmp_dir)
        state = {
            "step": int(step),
            "dataset_index": int(dataset_index),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "dpo_config": asdict(config),
        }
        torch.save(state, tmp_dir / "trainer_state.pt")
        (tmp_dir / "checkpoint_metadata.json").write_text(
            json.dumps({"step": int(step), "dataset_index": int(dataset_index)}, indent=2) + "\n",
            encoding="utf-8",
        )
        if checkpoint_dir.exists():
            backup = checkpoint_dir.parent / f".{checkpoint_dir.name}.bak.{os.getpid()}"
            os.replace(checkpoint_dir, backup)
            os.replace(tmp_dir, checkpoint_dir)
            shutil.rmtree(backup, ignore_errors=True)
        else:
            os.replace(tmp_dir, checkpoint_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return checkpoint_dir


class DPOTrainer:
    """Reference-policy DPO trainer compatible with TrainLM checkpoints and ROCm/CUDA/CPU."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer: Any,
        train_dataset: list[dict[str, list[int]]] | Any,
        output_dir: str | Path,
        config: DPOConfig | None = None,
        per_device_batch_size: int = 4,
        max_steps: int | None = None,
        save_steps: int = 200,
        logging_steps: int = 10,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or DPOConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_device_batch_size = per_device_batch_size
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model = model.to(self.device)
        self.ref_model = ref_model.to(
            device=self.device,
            dtype=torch.bfloat16 if self.config.bf16 and self.device.type == "cuda" else None,
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )
        total_steps = max_steps or max(
            1, len(train_dataset) // max(1, per_device_batch_size * self.config.gradient_accumulation_steps)
        )
        warmup = max(1, min(self.config.warmup_steps, total_steps))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda s: min(1.0, (s + 1) / warmup)
            * max(0.0, 1.0 - max(0, s - warmup) / max(1, total_steps - warmup)),
        )

    def train(self, *, resume_from_checkpoint: str | Path | None = None) -> dict[str, Any]:
        start_step = 0
        skip_examples = 0
        if resume_from_checkpoint is not None:
            ckpt = Path(resume_from_checkpoint)
            state = torch.load(ckpt / "trainer_state.pt", map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(state["optimizer"])
            if state.get("scheduler") is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            start_step = int(state.get("step", 0))
            skip_examples = int(state.get("dataset_index", 0))

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0

        dataset = self.train_dataset
        if skip_examples > 0 and skip_examples < len(dataset):
            dataset = dataset[skip_examples:]

        loader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            collate_fn=lambda f: _collate_dpo_batch(
                f, pad_token_id=int(pad_id), label_pad_token_id=self.config.label_pad_token_id
            ),
        )

        self.model.train()
        step = start_step
        micro_step = 0
        seen_examples = skip_examples
        accum = self.config.gradient_accumulation_steps
        use_amp = self.config.bf16 and self.device.type == "cuda"
        amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if use_amp
            else torch.no_grad.__self__ if False else __import__("contextlib").nullcontext()
        )
        self.optimizer.zero_grad(set_to_none=True)
        last_loss = 0.0
        t0 = time.monotonic()

        for batch in loader:
            seen_examples += int(batch["chosen_input_ids"].shape[0])
            c_ids = batch["chosen_input_ids"].to(self.device)
            c_lbl = batch["chosen_labels"].to(self.device)
            c_mask = batch["chosen_attention_mask"].to(self.device)
            r_ids = batch["rejected_input_ids"].to(self.device)
            r_lbl = batch["rejected_labels"].to(self.device)
            r_mask = batch["rejected_attention_mask"].to(self.device)

            with amp_ctx:
                pi_c_logits = self.model(input_ids=c_ids, attention_mask=c_mask).logits
                pi_r_logits = self.model(input_ids=r_ids, attention_mask=r_mask).logits
                with torch.no_grad():
                    ref_c_logits = self.ref_model(input_ids=c_ids, attention_mask=c_mask).logits
                    ref_r_logits = self.ref_model(input_ids=r_ids, attention_mask=r_mask).logits

                pi_c_logps = compute_sequence_logps(pi_c_logits, c_lbl, label_pad_token_id=self.config.label_pad_token_id)
                pi_r_logps = compute_sequence_logps(pi_r_logits, r_lbl, label_pad_token_id=self.config.label_pad_token_id)
                ref_c_logps = compute_sequence_logps(ref_c_logits, c_lbl, label_pad_token_id=self.config.label_pad_token_id)
                ref_r_logps = compute_sequence_logps(ref_r_logits, r_lbl, label_pad_token_id=self.config.label_pad_token_id)

                loss, c_rew, r_rew = compute_dpo_loss(
                    pi_c_logps, pi_r_logps, ref_c_logps, ref_r_logps, beta=self.config.beta
                )
            (loss / accum).backward()
            micro_step += 1
            last_loss = float(loss.detach().item())

            if micro_step % accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % self.logging_steps == 0 or step == 1:
                    elapsed = max(1e-6, time.monotonic() - t0)
                    margin = float((c_rew - r_rew).mean().item())
                    print(
                        f"[DPO] step={step} | loss={last_loss:.4f} | reward_margin={margin:.4f} | elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                if self.save_steps > 0 and step % self.save_steps == 0:
                    _atomic_save_checkpoint(
                        self.output_dir / f"checkpoint-{step:06d}",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=step,
                        dataset_index=seen_examples,
                        config=self.config,
                    )
                if self.max_steps is not None and step >= self.max_steps:
                    break

        final_ckpt = _atomic_save_checkpoint(
            self.output_dir / f"checkpoint-{step:06d}",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            dataset_index=seen_examples,
            config=self.config,
        )
        return {"step": step, "loss": last_loss, "final_checkpoint": str(final_ckpt)}
