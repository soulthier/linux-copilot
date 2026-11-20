import argparse
import json
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_chat_dataset(jsonl_path: Path, tokenizer) -> Dataset:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages") or []
            if len(messages) < 3:
                continue
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            records.append({"text": text})
    return Dataset.from_list(records)


def enforce_gpu_only(model) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU-only training is required.")
    if str(model.device) != "cuda:0":
        raise RuntimeError(f"Model is on {model.device}, expected cuda:0.")
    for name, param in model.named_parameters():
        if param.device.type != "cuda":
            raise RuntimeError(f"Parameter {name} is on {param.device} instead of cuda.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="data/processed/qlora_train.jsonl")
    parser.add_argument("--out", default="outputs/qwen25_7b_secops_lora")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU-only training is required.")

    torch.cuda.set_device(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
        max_memory={0: "11500MB", "cpu": "0MB"},
        dtype=torch.bfloat16,
    )

    enforce_gpu_only(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    dataset = load_chat_dataset(Path(args.data), tokenizer)

    training_args = SFTConfig(
        output_dir=args.out,
        dataset_text_field="text",
        max_length=args.max_seq_len,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    enforce_gpu_only(trainer.model)

    trainer.train()
    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[+] Saved QLoRA adapter -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
