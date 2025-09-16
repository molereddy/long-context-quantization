import os
import gc
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.helpers import (
    calculate_offload_device_map
)
from llmcompressor.modifiers.quantization import QuantizationModifier

import subprocess
import logging

HF_HOME = os.getenv('HF_HOME', '~/.cache/huggingface') # custom cache

def save_bnb_quantized(model_stub, name):
    print(name)
    device_map = calculate_offload_device_map(
        model_stub, reserve_for_hessians=False, num_gpus=torch.cuda.device_count(), torch_dtype=torch.bfloat16
    )

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_stub, torch_dtype=torch.bfloat16, 
        quantization_config=quantization_config, device_map=device_map, cache_dir=HF_HOME
    )
    tokenizer = AutoTokenizer.from_pretrained(model_stub)
    save_path = os.path.join(HF_HOME, f"{name}-bnb-8bit")
    model.save_pretrained(save_path, safetensors=True)
    tokenizer.save_pretrained(save_path)
    print("Saved to", save_path)
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def save_fp8_quantized_stagewise(model_stub="Qwen/Qwen2.5-32B-Instruct", num_stages=2):
    recipe = """
    quant_stage:
        quant_modifiers:
            QuantizationModifier:
                ignore: ["lm_head"]
                config_groups:
                    group_0:
                        weights:
                            num_bits: 8
                            type: float
                            strategy: tensor
                            dynamic: false
                            symmetric: true
                        input_activations:
                            num_bits: 8
                            type: float
                            strategy: tensor
                            dynamic: false
                            symmetric: true
                        targets: ["Linear"]
    """
    model_name = model_stub.split("/")[-1]
    device_map = calculate_offload_device_map(
        model_stub, reserve_for_hessians=False, num_gpus=torch.cuda.device_count(), torch_dtype=torch.bfloat16
    )
    print(f"Using {torch.cuda.device_count()} GPUs")
    print('--'*40)
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_stub, torch_dtype=torch.bfloat16, device_map=device_map, cache_dir=HF_HOME
    )
    tokenizer = AutoTokenizer.from_pretrained(model_stub)

    output_dir = os.path.join(HF_HOME, f"{model_name}-FP8")
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"
    NUM_CALIBRATION_SAMPLES = 512
    MAX_SEQUENCE_LENGTH = 4096

    NCS_per_stage = 512//num_stages
        

    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(  # noqa: F821
                example["messages"],
                tokenize=False,
            )
        }
    
    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(  # noqa: F821
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    
    for stage in range(num_stages-1):
        print(f'Running stage {stage} quantiztaion')
        print('--'*60)
        stage_ds = ds[stage*NCS_per_stage: stage*NCS_per_stage+NCS_per_stage]
        oneshot(
            model=model,
            output_dir=output_dir,
            dataset=stage_ds,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NCS_per_stage,
            save_compressed=False
        )
    
    stage = num_stages-1
    stage_ds = ds[stage*NCS_per_stage:]
    oneshot(
        model=model,
        output_dir=output_dir,
        dataset=stage_ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=len(stage_ds),
        save_compressed=True
    )
    
    print(f"Model saved to {output_dir}")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# code from https://huggingface.co/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8
def save_fp8_quantized(model_stub="Qwen/Qwen2.5-32B-Instruct"):
    print("***"*30)
    recipe = """
    quant_stage:
        quant_modifiers:
            QuantizationModifier:
                ignore: ["lm_head"]
                config_groups:
                    group_0:
                        weights:
                            num_bits: 8
                            type: float
                            strategy: tensor
                            dynamic: false
                            symmetric: true
                        input_activations:
                            num_bits: 8
                            type: float
                            strategy: tensor
                            dynamic: false
                            symmetric: true
                        targets: ["Linear"]
    """
    model_name = model_stub.split("/")[-1]
    device_map = calculate_offload_device_map(
        model_stub, reserve_for_hessians=False, num_gpus=torch.cuda.device_count(), torch_dtype=torch.bfloat16
    )
    print(f"Using {torch.cuda.device_count()} GPUs")
    print('--'*40)
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_stub, torch_dtype=torch.bfloat16, device_map=device_map, cache_dir=HF_HOME
    )
    tokenizer = AutoTokenizer.from_pretrained(model_stub)

    output_dir = os.path.join(HF_HOME, f"{model_name}-FP8")
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"
    NUM_CALIBRATION_SAMPLES = 512
    if '70' in model_name or '72' in model_name and torch.cuda.device_count()<3:
        NUM_CALIBRATION_SAMPLES = 450
    MAX_SEQUENCE_LENGTH = 4096

    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(  # noqa: F821
                example["messages"],
                tokenize=False,
            )
        }
    
    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(  # noqa: F821
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    oneshot(
        model=model,
        output_dir=output_dir,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        save_compressed=True
    )
    print(f"Model saved to {output_dir}")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("***"*30)

# code from https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w8a8_fp8
def save_fp8_dynamic_quantized(model_stub="Qwen/Qwen2.5-32B-Instruct"):
    print("**"*20, "START QUANTIZATION", "**"*20)
    print(model_stub)
    model_name = model_stub.split("/")[1]
    num_gpus = 1 if '72' not in model_name else torch.cuda.device_count()
    device_map = calculate_offload_device_map(
        model_stub, reserve_for_hessians=False, num_gpus=num_gpus, torch_dtype=torch.bfloat16
    )
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_stub, device_map=device_map, torch_dtype=torch.bfloat16, cache_dir=HF_HOME
    )
    tokenizer = AutoTokenizer.from_pretrained(model_stub)
    output_dir = os.path.join(HF_HOME, f"{model_name}-FP8-Dynamic")
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("**"*20, "END QUANTIZATION", "**"*20)


# only for llama-3.1_70b_gptq-w8a16
def save_gptq_quantized_special_recipe(device_map="auto"):
    print("**"*20, "START QUANTIZATION", "**"*20)
    HF_HOME = os.getenv('HF_HOME', '~/.cache/huggingface')
    model_stub = '/datasets/ai/llama3/meta-llama/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393'
    num_samples = 256
    max_seq_len = 8192
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_stub, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=HF_HOME
    )
    tokenizer = AutoTokenizer.from_pretrained(model_stub)
    def preprocess_fn(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}

    ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
    ds = ds.shuffle().select(range(num_samples))
    ds = ds.map(preprocess_fn)

    output_dir = os.path.join(HF_HOME, "Meta-Llama-3.1-70B-Instruct-GPTQ-W8A16")
    recipe = """
    quant_stage:
        quant_modifiers:
            SmoothQuantModifier:
                smoothing_strength: 0.7
                mappings:
                - 
                    - ['re:.*q_proj', 're:.*k_proj', 're:.*v_proj']
                    - re:.*input_layernorm
                - 
                    - ['re:.*gate_proj', 're:.*up_proj']
                    - re:.*post_attention_layernorm
                - 
                    - ['re:.*down_proj']
                    - re:.*up_proj
            GPTQModifier:
                sequential_update: false
                dampening_frac: 0.2
                ignore: [lm_head]
                scheme: W8A16
                targets: [Linear]
                observer: mse
    """
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=max_seq_len, num_calibration_samples=num_samples)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("**"*20, "END QUANTIZATION", "**"*20)

save_gptq_quantized_special_recipe()
# save_fp8_dynamic_quantized("Qwen/Qwen2.5-72B-Instruct")
# save_fp8_dynamic_quantized("Qwen/Qwen2.5-32B-Instruct")
# save_fp8_dynamic_quantized("Qwen/Qwen2.5-7B-Instruct")
# save_fp8_quantized("meta-llama/Meta-Llama-3.1-8B-Instruct", num_stages=2)
# save_fp8_quantized("Qwen/Qwen2.5-72B-Instruct", num_stages=2)
# save_bnb_quantized("meta-llama/Meta-Llama-3.1-70B-Instruct", 'Meta-Llama-3.1-70B-Instruct')