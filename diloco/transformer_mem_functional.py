from dataclasses import dataclass
from enum import Enum, auto
import math
from typing import Dict, Union
from name import Transformer, TrainingConfig, Datatype
from constants import DATATYPE_TO_SIZE, H100_MEMORY

def _calculate_num_params(transformer):
    n_params = transformer.n_layers * (
        # Self attention
        4 * transformer.hidden_size * transformer.hidden_size + 
        # MLP
        8 * transformer.hidden_size * transformer.hidden_size +
        # Layer norms
        4 * transformer.hidden_size
    )
    return n_params

def calculate_model_memory(transformer: Transformer, config: TrainingConfig) -> float:
    # params = transformer.n_layers * (
    #     # Self attention
    #     4 * transformer.hidden_size * transformer.hidden_size + 
    #     # MLP
    #     8 * transformer.hidden_size * transformer.hidden_size +
    #     # Layer norms
    #     4 * transformer.hidden_size
    # )
    n_params = _calculate_num_params(transformer)
    
    # bytes_per_param = 2 if config.weight_dtype == Datatype.BFLOAT16 else 4
    bytes_per_param = DATATYPE_TO_SIZE[config.weight_dtype]
    
    model_mem = (n_params * bytes_per_param) / (config.tp_size * config.pp_size)
    if config.zero1:
        model_mem /= config.num_gpus
        
    return model_mem

def calculate_kv_cache_memory(transformer: Transformer, config: TrainingConfig) -> float:
    # bytes_per_param = 2 if config.weight_dtype == Datatype.BFLOAT16 else 4
    bytes_per_param = DATATYPE_TO_SIZE[config.weight_dtype]
    
    kv_cache_mem = (bytes_per_param * 2 * transformer.n_layers * 
                    transformer.n_heads * 
                    (transformer.hidden_size / transformer.n_heads) * 
                    config.ctx_length)
    
    return kv_cache_mem

def calculate_gradient_memory(transformer: Transformer, config: TrainingConfig) -> float:
    # params = transformer.n_layers * (
    #     4 * transformer.hidden_size * transformer.hidden_size + 
    #     8 * transformer.hidden_size * transformer.hidden_size +
    #     4 * transformer.hidden_size
    # )
    n_params = _calculate_num_params(transformer)
    
    # bytes_per_grad = 2 if config.gradient_dtype == Datatype.BFLOAT16 else 4
    bytes_per_grad = DATATYPE_TO_SIZE[config.gradient_dtype]
    
    gradient_mem = n_params * bytes_per_grad
    if config.zero1:
        gradient_mem /= config.num_gpus
    gradient_mem /= config.pp_size
    
    return gradient_mem

def calculate_activation_memory(transformer: Transformer, config: TrainingConfig) -> float:
    assert config.tp_size == 1, "not support tensor parallelism"
    # NOTE: store all activations to do backpropogation
    # NOTE: reference: https://arxiv.org/abs/2205.05198
    # section 4.1

    # NOTE:
    # qkv_matmul = 2sbh
    # qk_scores = 4sbh
    # attn_softmax = as^2b
    # attn_v = 2as^2b + 2sbh
    # mlp: 19sbh
    # layer_norm: 4sbh

    linear_proj_input = 2 * config.ctx_length * config.batch_size_per_replicas * transformer.hidden_size
        
    attn_qkv_matmul = 2 * config.ctx_length * config.batch_size_per_replicas * transformer.hidden_size
    attn_qk_scores = 4 * config.ctx_length * config.batch_size_per_replicas * transformer.hidden_size
    attn_softmax = 2 * transformer.n_heads * (config.ctx_length **2) * config.batch_size_per_replicas
    attn_dropout = transformer.n_heads * (config.ctx_length **2) * config.batch_size_per_replicas
    attn_v = 2 * transformer.n_heads * (config.ctx_length**2) * config.batch_size_per_replicas + 2 * config.ctx_length * config.batch_size_per_replicas * transformer.hidden_size
    attn_drop_mask = config.ctx_length * config.batch_size_per_replicas * transformer.hidden_size

    mlp = 19 * config.ctx_length * config.batch_size_per_replicas * transformer.hidden_size
    ln = 4 * config.ctx_length * config.batch_size_per_replicas * transformer.hidden_size
    
    total_activation_mem_per_layer_dict = {
        "linear_proj_input": linear_proj_input,
        "attn_qkv_matmul": attn_qkv_matmul,
        "attn_qk_scores": attn_qk_scores,
        "attn_softmax": attn_softmax,
        "attn_dropout": attn_dropout,
        "attn_v": attn_v,
        "attn_drop_mask": attn_drop_mask,
        "mlp": mlp,
        "ln": ln
    }

    # total_activation_mem_per_layer = linear_proj_input + attn_qkv_matmul + attn_qk_scores + attn_softmax + attn_dropout + attn_v + attn_drop_mask + mlp + ln
    total_activation_mem_per_layer = sum(total_activation_mem_per_layer_dict.values())
    # NOTE: add percent to the dict
    total_activation_mem_per_layer_dict = {k: (v, round(((v / total_activation_mem_per_layer) * 100), 2)) for k, v in total_activation_mem_per_layer_dict.items()}

    if config.checkpoint_activations:
        total_activation_mem = (config.ctx_length * 
                         config.batch_size_per_replicas * 
                         transformer.hidden_size * 
                         transformer.n_layers * 
                         (10 + (24 / config.tp_size)))
    else:
        ref_total_activation_mem = (config.ctx_length * 
                         config.batch_size_per_replicas * 
                         transformer.hidden_size * 
                         transformer.n_layers * 
                         (10 + (24 / config.tp_size) + 
                          5 * ((transformer.n_heads * config.ctx_length) / 
                               (transformer.hidden_size * config.tp_size))))
        total_activation_mem = total_activation_mem_per_layer * transformer.n_layers
        assert total_activation_mem == ref_total_activation_mem, f"{total_activation_mem} != {ref_total_activation_mem}"
    
    if config.partition_activations:
        total_activation_mem /= config.tp_size

    # NOTE: the formula above already takes into account 2 bytes per value (16 bit)
    # bytes_per_act = DATATYPE_TO_SIZE[config.weight_dtype]
    # total_activation_mem *= bytes_per_act
        
    return total_activation_mem, total_activation_mem_per_layer_dict

def calculate_optimizer_memory(transformer: Transformer, config: TrainingConfig) -> float:
    # params = transformer.n_layers * (
    #     4 * transformer.hidden_size * transformer.hidden_size + 
    #     8 * transformer.hidden_size * transformer.hidden_size +
    #     4 * transformer.hidden_size
    # )
    n_params = _calculate_num_params(transformer)
    
    # bytes_multiplier = 8 if config.weight_dtype == Datatype.FP8 else 12
    # optimizer_state_mem = params * bytes_multiplier
    
    optim_first_state_mem = DATATYPE_TO_SIZE[config.optim_first_state_dtype] * n_params
    optim_second_state_mem = DATATYPE_TO_SIZE[config.optim_second_state_dtype] * n_params
    optimizer_state_mem = optim_first_state_mem + optim_second_state_mem

    if config.master_weight_dtype is not None:
        optimizer_master_weight_mem = DATATYPE_TO_SIZE[config.master_weight_dtype] * n_params 
    else:
        optimizer_master_weight_mem = 0

    optimizer_mem = optimizer_state_mem + optimizer_master_weight_mem
    
    if config.zero1:
        optimizer_mem /= config.num_gpus
        
    return optimizer_mem

def calculate_communication_memory(transformer: Transformer, config: TrainingConfig) -> float:
    # params = transformer.n_layers * (
    #     4 * transformer.hidden_size * transformer.hidden_size + 
    #     8 * transformer.hidden_size * transformer.hidden_size +
    #     4 * transformer.hidden_size
    # )
    n_params = _calculate_num_params(transformer)
    
    # bytes_per_param = 2 if config.weight_dtype == Datatype.BFLOAT16 else 4
    bytes_per_param = DATATYPE_TO_SIZE[config.weight_dtype]
    communication_mem = 0
    
    if config.zero1:
        # Assuming fixed bucket size of 500MB
        communication_mem += 5e8 * bytes_per_param
        # Assuming max live params of 1B
        communication_mem += 1e9 * bytes_per_param
        
    return communication_mem

def calculate_memory_requirements(
    transformer: Transformer,
    config: TrainingConfig,
    include_percent: bool = False
    # misc_mem: float = 0
) -> Dict[str, float]:

    assert config.partition_activations is False, "not support partition activations"
    
    model_mem = calculate_model_memory(transformer, config)
    kv_cache_mem = calculate_kv_cache_memory(transformer, config)
    grad_mem = calculate_gradient_memory(transformer, config)
    activation_mem = calculate_activation_memory(transformer, config)[0]
    optim_mem = calculate_optimizer_memory(transformer, config)
    # comm_mem = calculate_communication_memory(transformer, config)

    inference_memory = model_mem + kv_cache_mem
    # training_memory = sum(memory_components.values()) - memory_components["kv_cache_mem"]
    training_memory = model_mem + grad_mem + activation_mem + optim_mem

    # + memory_components["misc_mem"])
    memory_components = {
        "model_mem": (model_mem, round(((model_mem / training_memory) * 100), 2)),
        "activation_mem": (activation_mem, round(((activation_mem / training_memory) * 100), 2)),
        "kv_cache_mem": (kv_cache_mem, 0),
        "grad_mem": (grad_mem, round(((grad_mem / training_memory) * 100), 2)),
        "optim_mem": (optim_mem, round(((optim_mem / training_memory) * 100), 2)),
        # "comm_mem": (comm_mem, round(((comm_mem / training_memory) * 100), 2)),
    }
    
    memory_components_gb = {k: v for k, v in memory_components.items()}
        
    memory_components_gb.update({
        "total_training_mem": (training_memory, 100),
        "total_inference_mem": (inference_memory, 0)
    })

    return memory_components_gb
