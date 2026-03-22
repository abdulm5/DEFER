from __future__ import annotations

import inspect
import random
from pathlib import Path
from typing import Any

import numpy as np

from defer.core.io import write_json


def build_dpo_manifest(
    output_dir: str | Path,
    model_name: str,
    pair_train_path: str,
    pair_val_path: str,
    seed: int,
    mode: str = "dpo",
) -> dict:
    manifest = {
        "stage": "preference_optimization",
        "mode": mode,
        "model_name": model_name,
        "dataset": {"train": pair_train_path, "val": pair_val_path},
        "hyperparams": {
            "learning_rate": 1e-5,
            "epochs": 1,
            # Keep DPO memory-bounded on commodity 24-48GB GPUs.
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_len": 512,
            "max_prompt_len": 256,
            "beta": 0.1,
            "seed": seed,
        },
    }
    output = Path(output_dir) / f"{mode}_manifest.json"
    write_json(output, manifest)
    return manifest


def run_dpo_training(manifest: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:  # pragma: no cover - exercised only with train extras installed
        raise RuntimeError(
            "Training dependencies missing. Install with: pip install -e '.[train]'"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(manifest["hyperparams"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_name = manifest["model_name"]
    train_path = manifest["dataset"]["train"]
    val_path = manifest["dataset"]["val"]
    is_mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})
    required = {"prompt", "chosen", "rejected"}
    if not required.issubset(set(dataset["train"].column_names)):
        raise ValueError("DPO dataset must contain 'prompt', 'chosen', and 'rejected' fields.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = torch.cuda.is_available()
    quantized_4bit = False
    quantization_fallback_reason: str | None = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
            )
            quantized_4bit = True
        except Exception as exc:
            # Common incompatible stack path: accelerate dispatch calls `.to()` on a 4-bit model,
            # or bitsandbytes is not installed.
            msg = str(exc).lower()
            if not any(phrase in msg for phrase in [
                "not supported for",
                "bitsandbytes",
                "no module named",
            ]):
                raise
            quantization_fallback_reason = str(exc)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    train_batch_size = int(manifest["hyperparams"].get("batch_size", 1))
    eval_batch_size = int(manifest["hyperparams"].get("batch_size", 1))
    grad_accum = int(manifest["hyperparams"].get("gradient_accumulation_steps", 8))
    max_seq_len = int(manifest["hyperparams"].get("max_seq_len", 512))
    max_prompt_len = int(manifest["hyperparams"].get("max_prompt_len", 256))
    if is_mps:
        train_batch_size = min(train_batch_size, 1)
        eval_batch_size = min(eval_batch_size, 1)
        grad_accum = max(grad_accum, 8)
        max_seq_len = 512
        max_prompt_len = 256

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "learning_rate": float(manifest["hyperparams"]["learning_rate"]),
        "num_train_epochs": float(manifest["hyperparams"]["epochs"]),
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "logging_steps": 25,
        "save_strategy": "epoch",
        "report_to": [],
        "seed": seed,
        "remove_unused_columns": False,
        "bf16": bool(torch.cuda.is_available()),
        "fp16": False,
        "gradient_checkpointing": True,
    }
    dpo_config_names = inspect.signature(DPOConfig.__init__).parameters
    if "eval_strategy" in dpo_config_names:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in dpo_config_names:
        training_kwargs["evaluation_strategy"] = "epoch"

    dpo_kwargs = dict(training_kwargs)
    dpo_kwargs["beta"] = float(manifest["hyperparams"]["beta"])
    if "max_length" in dpo_config_names:
        dpo_kwargs["max_length"] = max_seq_len
    if "max_prompt_length" in dpo_config_names:
        dpo_kwargs["max_prompt_length"] = max_prompt_len
    args = DPOConfig(**dpo_kwargs)

    trainer_signature = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "peft_config": peft_config,
    }
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))

    summary = {
        "train_loss": float(train_result.training_loss),
        "global_step": int(train_result.global_step),
        "output_dir": str(output_dir / "model"),
        "train_rows": int(len(dataset["train"])),
        "val_rows": int(len(dataset["validation"])),
        "is_mps": is_mps,
        "effective_batch_size": train_batch_size,
        "effective_eval_batch_size": eval_batch_size,
        "effective_grad_accum_steps": grad_accum,
        "effective_max_seq_len": max_seq_len,
        "effective_max_prompt_len": max_prompt_len,
        "quantized_4bit": quantized_4bit,
        "quantization_fallback_reason": quantization_fallback_reason,
    }
    write_json(output_dir / "dpo_train_summary.json", summary)
    return summary
