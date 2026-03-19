import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import tqdm
import yaml
from accelerate import infer_auto_device_map, init_empty_weights
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    VideoMAEImageProcessor,
)
from transformers.utils import logging

from utils.dataset import TactileLLMDataset
from utils.model import MultimodalLLMForCausalLM


MODEL_NAME_MAP = {
    "qwen2.5-14b": "Qwen/Qwen2.5-14B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_new_tokens(llm, tokenizer, new_tokens):
    tokenizer_vocab = tokenizer.get_vocab()
    new_tokens = list(set(new_tokens) - set(tokenizer_vocab.keys()))
    if len(new_tokens) == 0:
        print("No new tokens added to tokenizer.")
        return

    n_new_tokens = tokenizer.add_tokens(new_tokens)
    if n_new_tokens <= 0:
        print("No new tokens added to tokenizer.")
        return

    print(f"{n_new_tokens} tokens added to tokenizer.")
    llm.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        input_embeddings = llm.get_input_embeddings().weight
        input_embeddings_avg = input_embeddings[:-n_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-n_new_tokens:] = input_embeddings_avg


def parse_gpu_config(gpu_config_path):
    with open(gpu_config_path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def infer_llm_device_map(model_path, gpu_max_mem_config):
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_path)
        auto_model = AutoModelForCausalLM.from_config(config)
    return infer_auto_device_map(
        auto_model,
        max_memory=gpu_max_mem_config,
        no_split_module_classes=["LLaMADecoderLayer", "LlamaDecoderLayer", "QWenBlock", "Qwen2DecoderLayer"],
    )


def cast_lora_dtype_if_needed(peft_llm):
    base_dtype = peft_llm.base_model.dtype
    print(f"Checking and casting LoRA parameter dtypes to match base model ({base_dtype})...")
    cast_count = 0
    for name, param in peft_llm.named_parameters():
        if "lora_" in name and param.dtype != base_dtype:
            print(f"Casting LoRA parameter '{name}' from {param.dtype} to {base_dtype}")
            param.data = param.data.to(base_dtype)
            cast_count += 1
    if cast_count > 0:
        print(f"Casted {cast_count} LoRA parameters.")
    else:
        print("LoRA parameters already have the correct dtype or no casting needed.")


def resolve_max_new_tokens(configs, question_type):
    max_new_tokens_cfg = configs.get("max_new_tokens", {})
    if question_type in max_new_tokens_cfg:
        return max_new_tokens_cfg[question_type]

    question_suffix = question_type.split("_", 1)[-1] if "_" in question_type else question_type
    for key, value in max_new_tokens_cfg.items():
        if key.endswith(question_suffix):
            return value

    return max_new_tokens_cfg.get("default", 200)


def flatten_question(question):
    chunks = []
    for chunk in question:
        if isinstance(chunk, (list, tuple)):
            chunks.append(str(chunk[0]) if len(chunk) > 0 else "")
        else:
            chunks.append(str(chunk))
    return "".join(chunks)


def normalize_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def normalize_paths(paths):
    if isinstance(paths, tuple):
        paths = list(paths)
    if isinstance(paths, list):
        normalized = []
        for path in paths:
            if isinstance(path, (list, tuple)) and len(path) == 1:
                normalized.append(path[0])
            else:
                normalized.append(path)
        return normalized
    return paths


def build_dataloader(dataset, batch_size, shuffle, g):
    if batch_size != 1:
        print(
            "Warning: Current multimodal collate pipeline assumes batch_size=1. "
            f"Received batch_size={batch_size}, this may fail on variable-length samples."
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=g,
    )


def load_llm_and_tokenizer(configs, new_tokens):
    if configs["model_type"] not in MODEL_NAME_MAP:
        raise ValueError(f"Unsupported model_type: {configs['model_type']}")

    tokenizer_path = MODEL_NAME_MAP[configs["model_type"]]
    model_path = MODEL_NAME_MAP[configs["model_type"]]

    if configs.get("tokenizer_path") is not None:
        tokenizer_path = configs["tokenizer_path"]
    if not configs.get("lora_trained", False) and configs.get("llm_path") is not None:
        model_path = configs["llm_path"]

    os.makedirs(configs["offload_dir"], exist_ok=True)

    bnb_config = None
    if configs["quantized"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    gpu_max_mem_config = None
    device_map = "auto"
    if configs.get("gpu_config") is not None:
        gpu_max_mem_config = parse_gpu_config(configs["gpu_config"])
        device_map = infer_llm_device_map(model_path, gpu_max_mem_config)

    llm_load_kwargs = {
        "device_map": device_map,
        "offload_folder": configs["offload_dir"],
    }
    if gpu_max_mem_config is not None:
        llm_load_kwargs["max_memory"] = gpu_max_mem_config

    if configs["quantized"]:
        llm_load_kwargs["quantization_config"] = bnb_config
        llm_load_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        llm_load_kwargs["attn_implementation"] = "flash_attention_2"
        llm_load_kwargs["torch_dtype"] = torch.bfloat16

    llm = AutoModelForCausalLM.from_pretrained(model_path, **llm_load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if configs.get("tokenizer_path") is None:
        add_new_tokens(llm, tokenizer, new_tokens)

    if configs.get("lora_trained", False):
        if configs.get("llm_path") is None:
            raise ValueError("lora_trained=True requires llm_path to point to a LoRA adapter directory.")
        peft_kwargs = {"is_trainable": False, "device_map": "auto"}
        if gpu_max_mem_config is not None:
            peft_kwargs["max_memory"] = gpu_max_mem_config
        llm = PeftModel.from_pretrained(model=llm, model_id=configs["llm_path"], **peft_kwargs)
        cast_lora_dtype_if_needed(llm)

    return llm, tokenizer, model_path, gpu_max_mem_config


def maybe_save_checkpoint(model, tokenizer, configs, exp_name, suffix=None):
    if suffix is None:
        tokenizer_path = f"{configs['exps_path']}/{exp_name}/tokenizer"
        llm_path = f"{configs['exps_path']}/{exp_name}/llm_weights"
        encoder_path = f"{configs['exps_path']}/{exp_name}/encoder.pt"
        project_path = f"{configs['exps_path']}/{exp_name}/project.pt"
    else:
        tokenizer_path = f"{configs['exps_path']}/{exp_name}/tokenizer_{suffix}"
        llm_path = f"{configs['exps_path']}/{exp_name}/llm_weights_{suffix}"
        encoder_path = f"{configs['exps_path']}/{exp_name}/encoder_{suffix}.pt"
        project_path = f"{configs['exps_path']}/{exp_name}/project_{suffix}.pt"

    tokenizer.save_pretrained(tokenizer_path)

    model.llm.save_pretrained(llm_path)
    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.project.state_dict(), project_path)


def train(configs, exp_name, g):
    device = f"cuda:{configs['cuda']}"
    new_tokens = ["<video_start>", "<video_end>", "<video>"]

    llm, tokenizer, model_path, gpu_max_mem_config = load_llm_and_tokenizer(configs, new_tokens)
    video_processor = VideoMAEImageProcessor.from_pretrained(configs["videomae_model_name"])

    if configs["train"]:
        train_dataset = TactileLLMDataset(video_processor, configs["train_files"], split_name="train", tokenizer=tokenizer)
        train_loader = build_dataloader(train_dataset, configs["per_device_train_batch_size"], shuffle=True, g=g)
    if configs["val"]:
        val_dataset = TactileLLMDataset(video_processor, configs["val_files"], split_name="val", tokenizer=tokenizer)
        val_loader = build_dataloader(val_dataset, configs["per_device_val_batch_size"], shuffle=False, g=g)
    if configs["test"]:
        test_dataset = TactileLLMDataset(video_processor, configs["test_files"], split_name="test", tokenizer=tokenizer)
        test_loader = build_dataloader(test_dataset, configs["per_device_val_batch_size"], shuffle=False, g=g)

    model_args = {
        "tokenizer": tokenizer,
        "videomae_model_name": configs["videomae_model_name"],
        "encoder_output_size": configs["encoder_output_size"],
        "cutoff_len": configs["cutoff_len"],
        "llm": llm,
        "use_vqvae": configs.get("use_vqvae", False),
        "device": device,
    }
    model = MultimodalLLMForCausalLM(**model_args)
    model.encoder.to(device=device, dtype=torch.bfloat16)
    model.project.to(device=device, dtype=torch.bfloat16)

    if configs["use_lora"]:
        peft_config = LoraConfig(
            r=configs["r"],
            lora_alpha=configs["lora_alpha"],
            lora_dropout=configs["lora_dropout"],
            target_modules=configs["target_modules"],
            bias=configs["bias"],
            inference_mode=False,
            task_type="CAUSAL_LM",
            modules_to_save=configs["modules_to_save"],
        )
        llm_weights_path = f"{configs['exps_path']}/{exp_name}/llm_weights"
        if not os.path.exists(llm_weights_path):
            os.makedirs(llm_weights_path)
            llm_peft = get_peft_model(llm, peft_config)
            llm_peft.save_pretrained(llm_weights_path)
            llm_peft = None

        peft_load_kwargs = {"is_trainable": True, "device_map": "auto"}
        if gpu_max_mem_config is not None:
            peft_load_kwargs["max_memory"] = gpu_max_mem_config

        print(f"Loading PeftModel with base model '{model_path}' and adapters '{llm_weights_path}'")
        llm = PeftModel.from_pretrained(model=llm, model_id=llm_weights_path, **peft_load_kwargs)
        cast_lora_dtype_if_needed(llm)
        model.llm = llm
    else:
        model.llm = llm

    if configs["train"]:
        llm_params = []
        if not configs["use_lora"]:
            for name, param in model.llm.named_parameters():
                if "embed_tokens" in name:
                    param.requires_grad = True
                    llm_params.append(param)
                else:
                    param.requires_grad = False
        else:
            for _, param in model.llm.named_parameters():
                if param.requires_grad:
                    llm_params.append(param)

        optimizer_llm = None
        scheduler_llm = None
        if len(llm_params) > 0:
            optimizer_llm = torch.optim.AdamW(llm_params, lr=configs["llm_lr"])
            num_steps = max(1, int(len(train_loader) / configs["llm_gradient_accumulation_steps"]))
            scheduler_llm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_llm, T_max=num_steps)

    encoder_weights_loaded_from_path = False
    if configs["encoder_path"] is not None and os.path.exists(configs["encoder_path"]):
        try:
            print(f"Loading custom encoder weights from: {configs['encoder_path']}")
            state_dict = torch.load(configs["encoder_path"], map_location="cpu", weights_only=False)

            if "model" in state_dict:
                print("Found 'model' key in checkpoint, using it as state_dict")
                state_dict = state_dict["model"]
            elif "module" in state_dict:
                print("Found 'module' key in checkpoint, using it as state_dict")
                state_dict = state_dict["module"]
            elif "state_dict" in state_dict:
                print("Found 'state_dict' key in checkpoint, using it as state_dict")
                state_dict = state_dict["state_dict"]

            is_from_train_clip = any(k.startswith("model.") for k in list(state_dict.keys())[:5])
            if is_from_train_clip:
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict

            missing_keys, unexpected_keys = model.encoder.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded custom VideoMAE weights from {configs['encoder_path']}")

            if missing_keys:
                print(f"Missing keys count: {len(missing_keys)}")
                print(f"First few missing keys: {missing_keys[:5]}")
            if unexpected_keys:
                print(f"Unexpected keys count: {len(unexpected_keys)}")
                print(f"First few unexpected keys: {unexpected_keys[:5]}")

            encoder_weights_loaded_from_path = True
            print(">>> [MARKER] Successfully loaded custom encoder weights from encoder_path.")
        except Exception as e:
            print(f"Failed to load weights from encoder_path: {e}")
            print("Proceeding with default model initialization.")
    else:
        print("encoder_path is not set or file does not exist. Using default model initialization.")

    optimizer_encoder = None
    if configs["freeze_encoder"]:
        for _, param in model.encoder.named_parameters():
            param.requires_grad = False
    else:
        encoder_params = filter(lambda p: p.requires_grad, model.encoder.parameters())
        optimizer_encoder = torch.optim.AdamW(encoder_params, lr=configs.get("encoder_lr", 1e-5))

    if configs["projection_path"] is not None:
        projection_dict = torch.load(configs["projection_path"], map_location="cpu")
        model.project.load_state_dict(projection_dict)

    optimizer_project = None
    if configs["freeze_projection"]:
        for _, param in model.project.named_parameters():
            param.requires_grad = False
    else:
        for _, param in model.project.named_parameters():
            param.requires_grad = True
        optimizer_project = torch.optim.AdamW(model.project.parameters(), lr=configs["projection_lr"])

    if configs["train"]:
        model.train()
        trainable_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        trainable_params = sum(np.prod(p.size()) for p in trainable_model_parameters)
        all_params = sum(np.prod(p.size()) for p in model.parameters())

        if configs["max_train_steps"] < len(train_loader):
            print(
                f"\nFinetuning LLM for {configs['max_train_steps']} samples and "
                f"{int(configs['max_train_steps'] / configs['llm_gradient_accumulation_steps'])} gradient updates..."
            )
        else:
            print(
                f"\nFinetuning LLM for {len(train_loader)} samples and "
                f"{int(len(train_loader) / configs['llm_gradient_accumulation_steps'])} gradient updates..."
            )
        print(f"Trainable params: {trainable_params} ({trainable_params / all_params * 100:.2f}%)")

        if encoder_weights_loaded_from_path:
            print(">>> [MARKER] Starting training using custom encoder weights loaded from encoder_path.")
        else:
            print(">>> [MARKER] Starting training using encoder weights from Hugging Face hub.")

        if optimizer_encoder is not None:
            optimizer_encoder.zero_grad(set_to_none=True)
        if optimizer_project is not None:
            optimizer_project.zero_grad(set_to_none=True)
        if optimizer_llm is not None:
            optimizer_llm.zero_grad(set_to_none=True)

        grad_accum_steps = configs["llm_gradient_accumulation_steps"]
        for train_sample_step, batch in enumerate(t := tqdm.tqdm(train_loader)):
            question, answer_tokens, tactile_pixel_values, tactile_paths, question_type, question_step, all_indices = batch
            answer_tokens = answer_tokens.to(device)
            outputs, _ = model(
                question=question,
                tactile_pixel_values=tactile_pixel_values,
                answer_tokens=answer_tokens,
                all_indices=all_indices,
            )
            train_loss = outputs.loss.detach().float()
            t.set_description(f"Train loss: {train_loss}")

            loss = outputs.loss / grad_accum_steps
            loss.backward()

            reached_accum = (train_sample_step + 1) % grad_accum_steps == 0
            reached_end = (train_sample_step + 1) >= configs["max_train_steps"] or (train_sample_step + 1) == len(train_loader)
            if reached_accum or reached_end:
                if optimizer_encoder is not None:
                    optimizer_encoder.step()
                    optimizer_encoder.zero_grad(set_to_none=True)
                if optimizer_project is not None:
                    optimizer_project.step()
                    optimizer_project.zero_grad(set_to_none=True)
                if optimizer_llm is not None:
                    optimizer_llm.step()
                    if scheduler_llm is not None:
                        scheduler_llm.step()
                    optimizer_llm.zero_grad(set_to_none=True)

            if configs["save_freq"] is not None and train_sample_step != 0 and (train_sample_step + 1) % configs["save_freq"] == 0:
                print("Saving tokenizer and models...")
                maybe_save_checkpoint(model, tokenizer, configs, exp_name, suffix=train_sample_step + 1)

            if (train_sample_step + 1) >= configs["max_train_steps"]:
                break

        if configs["save_freq"] is None:
            print("Saving tokenizer and models...")
            maybe_save_checkpoint(model, tokenizer, configs, exp_name, suffix=None)
            print("LLM training done!")

    def evaluate(split_name, loader, output_name):
        print(f"\nRunning {split_name} evaluation...")
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in tqdm.tqdm(loader):
                question, answer_tokens, tactile_pixel_values, tactile_paths, question_type, question_step, all_indices = batch
                answer_tokens = answer_tokens.to(device)

                question_embeds, question_attention_mask = model.build_generation_inputs(
                    question=question,
                    tactile_pixel_values=tactile_pixel_values,
                    all_indices=all_indices,
                )
                q_type = normalize_scalar(question_type)
                max_new_tokens = resolve_max_new_tokens(configs, q_type)
                generation_tokens = model.llm.generate(
                    inputs_embeds=question_embeds,
                    attention_mask=question_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                generation = tokenizer.decode(generation_tokens[0], skip_special_tokens=True).strip()
                answer = tokenizer.decode(answer_tokens[0].cpu().numpy(), skip_special_tokens=True).strip()

                preds.append(
                    {
                        "question": flatten_question(question),
                        "question_type": q_type,
                        "question_step": normalize_scalar(question_step),
                        "sample_paths": normalize_paths(tactile_paths),
                        "answer": answer,
                        "generation": generation,
                    }
                )

        with open(f"{configs['exps_path']}/{exp_name}/{output_name}", "w") as f:
            json.dump(preds, f, indent=4)
        print(f"{split_name} evaluation done!")

    if configs["val"]:
        evaluate("validation", val_loader, "val_preds.json")

    if configs["test"]:
        evaluate("test", test_loader, "test_preds.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_llm_config.yaml", help="path to training config")
    parser.add_argument("--exp_id", default=None, help="experiment identifier (optional)")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        configs = yaml.safe_load(file)

    exp_type = "train_llm"
    if configs["train"]:
        exp_type = exp_type + "_train"
    if configs["val"]:
        exp_type = exp_type + "_val"
    if configs["test"]:
        exp_type = exp_type + "_test"
    if configs["use_lora"]:
        exp_type = exp_type + f"_lora_{configs['lora_alpha']}_{configs['r']}"
    exp_type = exp_type + f"_{configs['model_type']}"
    if configs["train"]:
        exp_type += f"_{configs['max_train_steps']}"

    if args.exp_id is None:
        try:
            exp_id = input("Identifier for experiment: ")
        except EOFError:
            exp_id = ""
    else:
        exp_id = args.exp_id

    if len(exp_id) > 0:
        exp_id = exp_type + f"_{exp_id}"
    else:
        exp_id = exp_type

    now = datetime.now()
    exp_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_name = exp_name + "_" + exp_id
    print(f"\n{exp_name}\n")
    os.makedirs(configs["exps_path"], exist_ok=True)
    os.makedirs(f"{configs['exps_path']}/{exp_name}", exist_ok=True)

    with open(f"{configs['exps_path']}/{exp_name}/{exp_type}_config.yaml", "w") as file:
        yaml.dump(configs, file)

    if configs.get("redirect_stdout_to_log", True):
        sys.stdout = open(f"{configs['exps_path']}/{exp_name}/log.txt", "w")

    logging.set_verbosity_error()

    torch.manual_seed(configs["seed"])
    torch.random.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])
    random.seed(configs["seed"])

    g = torch.Generator()
    g.manual_seed(configs["seed"])

    train(configs, exp_name, g)
