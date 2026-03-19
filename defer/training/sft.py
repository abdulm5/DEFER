from __future__ import annotations

import inspect
import random
from pathlib import Path
from typing import Any

import numpy as np

from defer.core.io import write_json


def build_sft_manifest(
    output_dir: str | Path,
    model_name: str,
    train_path: str,
    val_path: str,
    seed: int,
) -> dict:
    manifest = {
        "stage": "sft",
        "method": "qlora",
        "model_name": model_name,
        "dataset": {"train": train_path, "val": val_path},
        "hyperparams": {
            "learning_rate": 2e-4,
            "epochs": 2,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "lora_r": 64,
            "lora_alpha": 16,
            "seed": seed,
        },
    }
    output = Path(output_dir) / "sft_manifest.json"
    write_json(output, manifest)
    return manifest


def run_sft_training(manifest: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTConfig, SFTTrainer
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
    if "prompt" not in dataset["train"].column_names or "response" not in dataset["train"].column_names:
        raise ValueError("SFT dataset must contain 'prompt' and 'response' fields.")

    def _to_text(example: dict[str, Any]) -> dict[str, str]:
        return {"text": f"<|user|>\n{example['prompt']}\n<|assistant|>\n{example['response']}"}

    train_ds = dataset["train"].map(_to_text, remove_columns=dataset["train"].column_names)
    eval_ds = dataset["validation"].map(_to_text, remove_columns=dataset["validation"].column_names)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = torch.cuda.is_available()
    quantization_config = None
    device_map = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    peft_config = LoraConfig(
        r=int(manifest["hyperparams"]["lora_r"]),
        lora_alpha=int(manifest["hyperparams"]["lora_alpha"]),
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    train_batch_size = int(manifest["hyperparams"]["batch_size"])
    eval_batch_size = int(manifest["hyperparams"]["batch_size"])
    grad_accum = int(manifest["hyperparams"]["gradient_accumulation_steps"])
    max_seq_len = 1024
    if is_mps:
        # Keep Mac validation runs stable; paper runs use CUDA where these caps do not apply.
        train_batch_size = min(train_batch_size, 1)
        eval_batch_size = min(eval_batch_size, 1)
        grad_accum = max(grad_accum, 8)
        max_seq_len = 512

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
    }
    sft_config_names = inspect.signature(SFTConfig.__init__).parameters
    if "eval_strategy" in sft_config_names:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sft_config_names:
        training_kwargs["evaluation_strategy"] = "epoch"

    trainer_signature = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_signature:
        sft_kwargs = dict(training_kwargs)
        if "dataset_text_field" in sft_config_names:
            sft_kwargs["dataset_text_field"] = "text"
        if "max_length" in sft_config_names:
            sft_kwargs["max_length"] = max_seq_len
        args = SFTConfig(**sft_kwargs)
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    else:
        arg_names = inspect.signature(TrainingArguments.__init__).parameters
        legacy_kwargs = dict(training_kwargs)
        if "evaluation_strategy" in arg_names:
            legacy_kwargs["evaluation_strategy"] = "epoch"
        elif "eval_strategy" in arg_names:
            legacy_kwargs["eval_strategy"] = "epoch"
        args = TrainingArguments(**legacy_kwargs)
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            dataset_text_field="text",
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_seq_length=max_seq_len,
        )
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))

    summary = {
        "train_loss": float(train_result.training_loss),
        "global_step": int(train_result.global_step),
        "output_dir": str(output_dir / "model"),
        "train_rows": int(len(train_ds)),
        "val_rows": int(len(eval_ds)),
        "is_mps": is_mps,
        "effective_batch_size": train_batch_size,
        "effective_eval_batch_size": eval_batch_size,
        "effective_grad_accum_steps": grad_accum,
        "effective_max_seq_len": max_seq_len,
    }
    write_json(output_dir / "sft_train_summary.json", summary)
    return summary
