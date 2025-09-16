import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams
import ray
import pickle
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import gc
import os
import time
import argparse
import logging
from tqdm import tqdm


os.environ['TZ'] = 'America/New_York'
time.tzset()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %Z'
)

def log_memory_usage(stage=""):
   import psutil
   import subprocess
   process = psutil.Process(os.getpid())
   mem_info = process.memory_info()
   logging.info(f"{stage} - CPU Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB RSS")
  
   result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'])
   gpu_memory = [tuple(map(int, line.split(','))) for line in result.decode('utf-8').strip().split('\n')]
   for i, (used, total) in enumerate(gpu_memory):
       logging.info(f"GPU {i}: {used}/{total} MB")

def modify_model_path(original_path, suffix="_long_ctx"):
    path_parts = original_path.split(os.sep)
    for i, part in enumerate(path_parts):
        if part.startswith("models--") or part.startswith("Qwen"):
            path_parts[i] = f"{part}{suffix}"
            break
    return os.sep.join(path_parts)

HF_HOME = os.getenv('HF_HOME', '~/.cache/huggingface')
model_dir = HF_HOME
MODEL_CONFIGS = {
    'llama3.1': {
        '8b': {
            "bf16": {'path': 'meta-llama/Llama-3.1-8B-Instruct', 
                     'bsz': 100, 'name': 'Meta-Llama-3.1-8B-Instruct', 'long_ctx_num_devices': 4, 'num_devices': 1},
            "gptq-int4": {'path': 'neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16', 
                          'bsz': 100, 'name': 'Meta-Llama-3.1-8B-Instruct-GPTQ-INT4', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "gptq-int8": {'path': 'neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16', 
                          'bsz': 100, 'name': 'Meta-Llama-3.1-8B-Instruct-GPTQ-INT8', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "awq-int4": {'path': 'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4', 
                         'bsz': 100, 'name': 'Meta-Llama-3.1-8B-Instruct-AWQ-INT4', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "fp8": {'path': 'neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic', 
                    'bsz': 100, 'name': 'Meta-Llama-3.1-8B-Instruct-FP8', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "bnb-int4": {'path': 'meta-llama/Llama-3.1-8B-Instruct', 
                         'bsz': 100, 'name': 'Meta-Llama-3.1-8B-Instruct-bnb-4bit', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "bnb-int8": {'path': 'meta-llama/Llama-3.1-8B-Instruct', 
                         'bsz': 100, 'name': 'Meta-Llama-3.1-8B-Instruct-bnb-8bit', 'long_ctx_num_devices': 1, 'num_devices': 1},
        },
        '70b': {
            "bf16": {'path': 'meta-llama/Llama-3.1-70B-Instruct', 
                     'bsz': 5, 'name': 'Meta-Llama-3.1-70B-Instruct', 'long_ctx_num_devices': 4, 'num_devices': 2},
            "gptq-int4": {'path': 'neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16', 
                          'bsz': 20, 'name': 'Meta-Llama-3.1-70B-Instruct-GPTQ-INT4', 'long_ctx_num_devices': 2, 'num_devices': 1},
            "gptq-int8": {'path': os.path.join(HF_HOME, 'Meta-Llama-3.1-70B-Instruct-GPTQ-W8A16'), 
                          'bsz': 10, 'name': 'Meta-Llama-3.1-70B-Instruct-GPTQ-INT8', 'long_ctx_num_devices': 2, 'num_devices': 1},
            "awq-int4": {'path': 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4', 
                         'bsz': 20, 'name': 'Meta-Llama-3.1-70B-Instruct-AWQ-INT4', 'long_ctx_num_devices': 2, 'num_devices': 1},
            "fp8": {'path': 'neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8-dynamic', 
                    'bsz': 10, 'name': 'Meta-Llama-3.1-70B-Instruct-FP8', 'long_ctx_num_devices': 2, 'num_devices': 1},
            "bnb-int4": {'path': 'meta-llama/Llama-3.1-70B-Instruct', 
                         'bsz': 20, 'name': 'Meta-Llama-3.1-70B-Instruct-bnb-4bit', 'long_ctx_num_devices': 2, 'num_devices': 1},
            "bnb-int8": {'path': 'meta-llama/Llama-3.1-70B-Instruct', 
                         'bsz': 20, 'name': 'Meta-Llama-3.1-70B-Instruct-bnb-8bit', 'long_ctx_num_devices': 4, 'num_devices': 2},
        }
    },
    'qwen2.5': {
        '7b': {
            "bf16": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"), 
                     'bsz': 100, 'name': 'Qwen2.5-7B-Instruct', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "awq-int4": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/b25037543e9394b818fdfca67ab2a00ecc7dd641"), 
                         'bsz': 100, 'name': 'Qwen2.5-7B-Instruct-AWQ', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "gptq-int4": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-7B-Instruct-GPTQ-Int4/snapshots/e9c932ac1893a49ae0fc497ad6e1e86e2e39af20"), 
                          'bsz': 100, 'name': 'Qwen2.5-7B-Instruct-GPTQ-Int4', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "gptq-int8": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-7B-Instruct-GPTQ-Int8/snapshots/6711f2b6fc95545efe6018c469eca6832e28cefd"), 
                          'bsz': 100, 'name': 'Qwen2.5-7B-Instruct-GPTQ-Int8', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "fp8": {'path': os.path.join(HF_HOME, "Qwen2.5-7B-Instruct-FP8-Dynamic"), 
                    'bsz': 100, 'name': 'Qwen2.5-7B-Instruct-FP8', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "bnb-int4": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"), 
                         'bsz': 100, 'name': 'Qwen2.5-7B-Instruct-bnb-4bit', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "bnb-int8": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"), 
                         'bsz': 100, 'name': 'Qwen2.5-7B-Instruct-bnb-8bit', 'long_ctx_num_devices': 1, 'num_devices': 1},
        },
        '32b': {
            "bf16": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"), 
                     'bsz': 100, 'name': 'Qwen2.5-32B-Instruct', 'long_ctx_num_devices': 2, 'num_devices': 1},
            "awq-int4": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots/5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"), 
                         'bsz': 100, 'name': 'Qwen2.5-32B-Instruct-AWQ', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "gptq-int4": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-32B-Instruct-GPTQ-Int4/snapshots/c83e67dfb2664f5039fd4cd99e206799e27dd800"), 
                          'bsz': 50, 'name': 'Qwen2.5-32B-Instruct-GPTQ-Int4', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "gptq-int8": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-32B-Instruct-GPTQ-Int8/snapshots/eddc13f573fd3648cc8a4741fdf1b70e8d6fc5c1"), 
                          'bsz': 30, 'name': 'Qwen2.5-32B-Instruct-GPTQ-Int8', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "fp8": {'path': os.path.join(HF_HOME, "Qwen2.5-32B-Instruct-FP8-Dynamic"), 
                    'bsz': 30, 'name': 'Qwen2.5-32B-Instruct-FP8', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "bnb-int4": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"), 
                         'bsz': 100, 'name': 'Qwen2.5-32B-Instruct-bnb-4bit', 'long_ctx_num_devices': 1, 'num_devices': 1},
            "bnb-int8": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"), 
                         'bsz': 30, 'name': 'Qwen2.5-32B-Instruct-bnb-8bit', 'long_ctx_num_devices': 1, 'num_devices': 1},
        },
        '72b': {
            "bf16": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842"), 
                     'bsz': 5, 'name': 'Qwen2.5-72B-Instruct', 'long_ctx_num_devices': 4, 'num_devices': 2},
            "awq-int4": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217"), 
                         'bsz': 20, 'name': 'Qwen2.5-72B-Instruct-AWQ', 'long_ctx_num_devices': 2, 'num_devices': 1},
            "gptq-int4": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-72B-Instruct-GPTQ-Int4/snapshots/da6e9d45661b91e02782f4ae2c6bb39c4a5b4821"), 
                          'bsz': 20, 'name': 'Qwen2.5-72B-Instruct-GPTQ-Int4', 'long_ctx_num_devices': 2, 'num_devices': 1},
            "gptq-int8": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-72B-Instruct-GPTQ-Int8/snapshots/4f953f7634fef56affc59a40656ebe7461f7e545"), 
                          'bsz': 10, 'name': 'Qwen2.5-72B-Instruct-GPTQ-Int8', 'long_ctx_num_devices': 2, 'num_devices': 1},
            "fp8": {'path': os.path.join(HF_HOME, "Qwen2.5-72B-Instruct-FP8-Dynamic"), 
                    'bsz': 10, 'name': 'Qwen2.5-72B-Instruct-FP8', 'long_ctx_num_devices': 2, 'num_devices': 2},
            "bnb-int4": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842"), 
                         'bsz': 20, 'name': 'Qwen2.5-72B-Instruct-bnb-4bit', 'long_ctx_num_devices': 2, 'num_devices': 1},
            "bnb-int8": {'path': os.path.join(HF_HOME, "models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842"), 
                         'bsz': 1, 'name': 'Qwen2.5-72B-Instruct-bnb-8bit', 'long_ctx_num_devices': 4, 'num_devices': 1},
        },
    }
}

NAME_TO_CONFIG = {
    'Meta-Llama-3.1-8B-Instruct': {
        'model': 'llama3.1',
        'size': '8b',
        'prec': 'bf16'
    },
    'Meta-Llama-3.1-8B-Instruct-GPTQ-INT4': {
        'model': 'llama3.1',
        'size': '8b',
        'prec': 'gptq-int4'
    },
    'Meta-Llama-3.1-8B-Instruct-GPTQ-INT8': {
        'model': 'llama3.1',
        'size': '8b',
        'prec': 'gptq-int8'
    },
    'Meta-Llama-3.1-8B-Instruct-AWQ-INT4': {
        'model': 'llama3.1',
        'size': '8b',
        'prec': 'awq-int4'
    },
    'Meta-Llama-3.1-8B-Instruct-FP8': {
        'model': 'llama3.1',
        'size': '8b',
        'prec': 'fp8'
    },
    'Meta-Llama-3.1-8B-Instruct-bnb-4bit': {
        'model': 'llama3.1',
        'size': '8b',
        'prec': 'bnb-int4'
    },
    'Meta-Llama-3.1-8B-Instruct-bnb-8bit': {
        'model': 'llama3.1',
        'size': '8b',
        'prec': 'bnb-int8'
    },
    'Meta-Llama-3.1-70B-Instruct': {
        'model': 'llama3.1',
        'size': '70b',
        'prec': 'bf16'
    },
    'Meta-Llama-3.1-70B-Instruct-GPTQ-INT4': {
        'model': 'llama3.1',
        'size': '70b',
        'prec': 'gptq-int4'
    },
    'Meta-Llama-3.1-70B-Instruct-GPTQ-INT8': {
        'model': 'llama3.1',
        'size': '70b',
        'prec': 'gptq-int8'
    },
    'Meta-Llama-3.1-70B-Instruct-AWQ-INT4': {
        'model': 'llama3.1',
        'size': '70b',
        'prec': 'awq-int4'
    },
    'Meta-Llama-3.1-70B-Instruct-FP8': {
        'model': 'llama3.1',
        'size': '70b',
        'prec': 'fp8'
    },
    'Meta-Llama-3.1-70B-Instruct-bnb-4bit': {
        'model': 'llama3.1',
        'size': '70b',
        'prec': 'bnb-int4'
    },
    'Meta-Llama-3.1-70B-Instruct-bnb-8bit': {
        'model': 'llama3.1',
        'size': '70b',
        'prec': 'bnb-int8'
    },
    'Qwen2.5-7B-Instruct': {
        'model': 'qwen2.5',
        'size': '7b',
        'prec': 'bf16'
    },
    'Qwen2.5-7B-Instruct-GPTQ-Int4': {
        'model': 'qwen2.5',
        'size': '7b',
        'prec': 'gptq-int4'
    },
    'Qwen2.5-7B-Instruct-GPTQ-Int8': {
        'model': 'qwen2.5',
        'size': '7b',
        'prec': 'gptq-int8'
    },
    'Qwen2.5-7B-Instruct-AWQ': {
        'model': 'qwen2.5',
        'size': '7b',
        'prec': 'awq-int4'
    },
    'Qwen2.5-7B-Instruct-FP8': {
        'model': 'qwen2.5',
        'size': '7b',
        'prec': 'fp8'
    },
    'Qwen2.5-7B-Instruct-bnb-4bit': {
        'model': 'qwen2.5',
        'size': '7b',
        'prec': 'bnb-int4'
    },
    'Qwen2.5-7B-Instruct-bnb-8bit': {
        'model': 'qwen2.5',
        'size': '7b',
        'prec': 'bnb-int8'
    },
    'Qwen2.5-32B-Instruct': {
        'model': 'qwen2.5',
        'size': '32b',
        'prec': 'bf16'
    },
    'Qwen2.5-32B-Instruct-GPTQ-Int4': {
        'model': 'qwen2.5',
        'size': '32b',
        'prec': 'gptq-int4'
    },
    'Qwen2.5-32B-Instruct-GPTQ-Int8': {
        'model': 'qwen2.5',
        'size': '32b',
        'prec': 'gptq-int8'
    },
    'Qwen2.5-32B-Instruct-AWQ': {
        'model': 'qwen2.5',
        'size': '32b',
        'prec': 'awq-int4'
    },
    'Qwen2.5-32B-Instruct-FP8': {
        'model': 'qwen2.5',
        'size': '32b',
        'prec': 'fp8'
    },
    'Qwen2.5-32B-Instruct-bnb-4bit': {
        'model': 'qwen2.5',
        'size': '32b',
        'prec': 'bnb-int4'
    },
    'Qwen2.5-32B-Instruct-bnb-8bit': {
        'model': 'qwen2.5',
        'size': '32b',
        'prec': 'bnb-int8'
    },
    'Qwen2.5-72B-Instruct': {
        'model': 'qwen2.5',
        'size': '72b',
        'prec': 'bf16'
    },
    'Qwen2.5-72B-Instruct-GPTQ-Int4': {
        'model': 'qwen2.5',
        'size': '72b',
        'prec': 'gptq-int4'
    },
    'Qwen2.5-72B-Instruct-GPTQ-Int8': {
        'model': 'qwen2.5',
        'size': '72b',
        'prec': 'gptq-int8'
    },
    'Qwen2.5-72B-Instruct-AWQ': {
        'model': 'qwen2.5',
        'size': '72b',
        'prec': 'awq-int4'
    },
    'Qwen2.5-72B-Instruct-bnb-4bit': {
        'model': 'qwen2.5',
        'size': '72b',
        'prec': 'bnb-int4'
    },
    'Qwen2.5-72B-Instruct-bnb-8bit': {
        'model': 'qwen2.5',
        'size': '72b',
        'prec': 'bnb-int8'
    },
    'Qwen2.5-72B-Instruct-FP8': {
        'model': 'qwen2.5',
        'size': '72b',
        'prec': 'fp8'
    }
}

def abbrv_model(full_model_name):
    model_deets = NAME_TO_CONFIG[full_model_name]
    return '_'.join([model_deets['model'],
                     model_deets['size'],
                     model_deets['prec']])

# takes the short name and returns the full official name
def full_name(abbrv_model):
    model, size, prec = abbrv_model.split('_')
    prec = prec.upper()
    return MODEL_CONFIGS[model][size][prec]['name']

def get_model_cfg(model_name):
    model, size, prec = model_name.split('_')
    return {
        'model': model, 'size': size, 'prec': prec
    }

def get_hf_cfg(model_name):
    if '_' not in model_name: # model full name given
        model_name = abbrv_model(model_name)
    model, size, prec = model_name.split('_')
    return MODEL_CONFIGS[model][size][prec]

def get_hf_bnb_model(cfg, max_seq_len=8192):
    bnb_model_path = cfg['name']
    base_model_name = bnb_model_path.split('-bnb')[-1]
    base_details = NAME_TO_CONFIG[base_model_name]
    base_model_path = MODEL_CONFIGS[base_details['model']][base_details['size']]['bf16']['path']
    if 'Qwen' in base_model_path and max_seq_len > 32768:
        # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
        # load from another folder which is the copy of the default but has config.json edited
        base_model_path = modify_model_path(base_model_path, "_long_ctx")
    model_params = {
        'pretrained_model_name_or_path': base_model_path, 
        'device_map': 'auto', 'cache_dir': HF_HOME, 
        'torch_dtype': torch.bfloat16, 'low_cpu_mem_usage': True
    }
    if '4bit' in cfg['name'] or '8bit' in cfg['name']:
        if '4bit' in cfg['name']:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif '8bit' in cfg['name']:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_params['quantization_config']= quantization_config
    model = AutoModelForCausalLM.from_pretrained(**model_params)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    return model, tokenizer

def get_vllm_model(cfg, max_seq_len=2500):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
    model_id = cfg['path']
    model_name = cfg['name']
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    ray.shutdown()
    ray.init(num_gpus=num_gpus)
    bnb_args = {}
    if 'bnb' in model_name or 'BNB' in model_name:
        # BNB not supported for loading by default, need to do give args
        bnb_args = {'quantization': "bitsandbytes", 'load_format': "bitsandbytes"}
    if 'Qwen' in model_id and max_seq_len > 32768:
        # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
        # load from another folder which is the copy of the default but has config.json edited
        model_id = modify_model_path(model_id, "_long_ctx")
    print('--'*40)
    print(f"Attempting to intialize model on all available {num_gpus} GPU(s) from {model_id}:")
    print('--'*40)
    llm = LLM(model=model_id, tensor_parallel_size=num_gpus, 
              max_model_len=max_seq_len, download_dir=HF_HOME, enable_chunked_prefill=False,
              **bnb_args)
    log_memory_usage(f"Loaded model from {model_id}.")
    tokenizer = llm.get_tokenizer()
    print("--"*30, '\t', "MODEL LOADED", '\t', "--"*30)
    return model_name, llm, tokenizer

def close_vllm_model(llm, tokenizer):
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm # Isn't necessary for releasing memory, but why not
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()

def vllm_prompt_generate(model, tokenizer, prompts, bsz, sampling_params=None):
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=2500) # greedy
    responses = []
    for i in tqdm(range(0, len(prompts), bsz)):
        batch_prompts = prompts[i: i+bsz]
        batch_prompts = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
        batch_prompts = tokenizer.apply_chat_template(batch_prompts, tokenize=False, add_generation_prompt=True)
        batch_outputs = model.generate(batch_prompts, sampling_params, use_tqdm=True)
        batch_responses = [output.outputs[0].text for output in batch_outputs]
        responses.extend(batch_responses)
    log_memory_usage("Usage post generation")
    return responses

def hf_prompt_generate(model, tokenizer, prompts, bsz):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    responses = []
    for i in tqdm(range(0, len(prompts), bsz)):
        batch_prompts = prompts[i: i + bsz]
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
        batch_encodings = tokenizer.apply_chat_template(batch_messages, add_generation_prompt=True, 
                                                 return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(inputs=batch_encodings, 
                                     max_new_tokens=2500, do_sample=False, top_p=None, 
                                     top_k=None, temperature=None)
        
        # Decode responses
        # batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # responses.extend(batch_responses)
        n = outputs.shape[0]
        for i in range(n):
            prompt_len = batch_encodings[i].shape[-1]
            output = outputs[i][prompt_len:]
            response = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(response)
    
    log_memory_usage("Usage post HF generation")
    return responses

# long context only
def load_model_and_tokenizer(model_short_name, context_size=128000):
    hf_cfg = get_hf_cfg(model_short_name)
    _, model, tokenizer = get_vllm_model(hf_cfg, max_seq_len=context_size)
    return model, tokenizer

def model_call(prompt, llm, tokenizer, temp=0.0, max_tokens=800):
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=max_tokens)
    output = vllm_prompt_generate(llm, tokenizer, [prompt], bsz=1, sampling_params=sp)[0]
    # print(f"Generated text: {output!r}")
    return output

def batch_model_call(prompts, llm, tokenizer, bsz, temp=0.0, max_tokens=800):
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=max_tokens)
    return vllm_prompt_generate(llm, tokenizer, prompts, bsz=bsz, sampling_params=sp)

# Note: 
# use latest version of vLLM: vllm==0.6.4.post1, otherwise qwen-bnb/bnb with TP will not work
# load_model_and_tokenizer by default is for long context tasks only, for short context, give the context length
# model_call supports only bsz 1, use vllm_prompt_generate for batched generation

# LONG CONTEXT MEMORY REQUIREMENTS
# number of A100s needed -> MODEL_CONFIGS[model_family][size][quant_config][long_ctx_num_devices]
# Basically: 70B needs 4 A100s for 16bit and 2 for quantized, 32B bf16 needs 2, 32B quantized can run on single GPU

# use model_short_name while requesting model: <model_family>_<size>_<quant_config>
# e.g. llama3.1_70b_gptq-int8, qwen2.5_32b_fp8, llama3.1_8b_bf16
# llama3.1 (8b, 70b), qwen2.5 (7b, 32b, 72b)
# bf16, fp8, gptq-int8, gptq-int4, awq-int4, bnb-int4

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-m', '--model_short_name', type=str, required=True)
    args = parser.parse_args()

    log_memory_usage("Initial")
    
    with open('<path>', 'rb') as f:
        books = pickle.load(f)
    
    book_content = books['<book name>']
    book_len = len(book_content)
    prompt = book_content[: int(0.9*book_len)] + '\n Give the name of the book.'
    llm, tokenizer = load_model_and_tokenizer(args.model_short_name, context_size=128000)
    _ = model_call(prompt, llm, tokenizer)

if __name__ == "__main__":
    main()