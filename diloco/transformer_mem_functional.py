from dataclasses import dataclass
from enum import Enum, auto
import math
from typing import Dict, Union
from name import Transformer, TrainingConfig, Datatype
from constants import DATATYPE_TO_SIZE

def calculate_model_memory(transformer: Transformer, config: TrainingConfig) -> float:
    params = transformer.n_layers * (
        # Self attention
        4 * transformer.hidden_size * transformer.hidden_size + 
        # MLP
        8 * transformer.hidden_size * transformer.hidden_size +
        # Layer norms
        4 * transformer.hidden_size
    )
    
    # bytes_per_param = 2 if config.weight_dtype == Datatype.BFLOAT16 else 4
    bytes_per_param = DATATYPE_TO_SIZE[config.weight_dtype]
    
    model_mem = (params * bytes_per_param) / (config.tp_size * config.pp_size)
    if config.zero1:
        model_mem /= config.num_gpus
        
    return model_mem

def calculate_kv_cache_memory(transformer: Transformer, config: TrainingConfig) -> float:
    # bytes_per_param = 2 if config.weight_dtype == Datatype.BFLOAT16 else 4
    bytes_per_param = DATATYPE_TO_SIZE[config.weight_dtype]
    
    kv_cache_mem = (bytes_per_param * 2 * transformer.n_layers * 
                    transformer.n_heads * 
                    (transformer.hidden_size / transformer.n_heads) * 
                    transformer.ctx_length)
    
    return kv_cache_mem

def calculate_gradient_memory(transformer: Transformer, config: TrainingConfig) -> float:
    params = transformer.n_layers * (
        4 * transformer.hidden_size * transformer.hidden_size + 
        8 * transformer.hidden_size * transformer.hidden_size +
        4 * transformer.hidden_size
    )
    
    # bytes_per_grad = 2 if config.gradient_dtype == Datatype.BFLOAT16 else 4
    bytes_per_grad = DATATYPE_TO_SIZE[config.gradient_dtype]
    
    gradient_mem = params * bytes_per_grad
    if config.zero1:
        gradient_mem /= config.num_gpus
    gradient_mem /= config.pp_size
    
    return gradient_mem

def calculate_activation_memory(transformer: Transformer, config: TrainingConfig) -> float:
    if config.checkpoint_activations:
        activation_mem = (transformer.ctx_length * 
                         config.batch_size_per_replicas * 
                         transformer.hidden_size * 
                         transformer.n_layers * 
                         (10 + (24 / config.tp_size)))
    else:
        activation_mem = (transformer.ctx_length * 
                         config.batch_size_per_replicas * 
                         transformer.hidden_size * 
                         transformer.n_layers * 
                         (10 + (24 / config.tp_size) + 
                          5 * ((transformer.n_heads * transformer.ctx_length) / 
                               (transformer.hidden_size * config.tp_size))))
    
    if config.partition_activations:
        activation_mem /= config.tp_size
        
    return activation_mem

def calculate_optimizer_memory(transformer: Transformer, config: TrainingConfig) -> float:
    params = transformer.n_layers * (
        4 * transformer.hidden_size * transformer.hidden_size + 
        8 * transformer.hidden_size * transformer.hidden_size +
        4 * transformer.hidden_size
    )
    
    # bytes_multiplier = 8 if config.weight_dtype == Datatype.FP8 else 12
    # optimizer_state_mem = params * bytes_multiplier
    
    optim_first_state_mem = DATATYPE_TO_SIZE[config.optim_first_state_dtype] * params
    optim_second_state_mem = DATATYPE_TO_SIZE[config.optim_second_state_dtype] * params
    optimizer_state_mem = optim_first_state_mem + optim_second_state_mem

    if config.master_weight_dtype is not None:
        optimizer_master_weight_mem = DATATYPE_TO_SIZE[config.master_weight_dtype] * params 
    else:
        optimizer_master_weight_mem = 0

    optimizer_mem = optimizer_state_mem + optimizer_master_weight_mem
    
    if config.zero1:
        optimizer_mem /= config.num_gpus
        
    return optimizer_mem

def calculate_communication_memory(transformer: Transformer, config: TrainingConfig) -> float:
    params = transformer.n_layers * (
        4 * transformer.hidden_size * transformer.hidden_size + 
        8 * transformer.hidden_size * transformer.hidden_size +
        4 * transformer.hidden_size
    )
    
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
    misc_mem: float = 0
) -> Dict[str, float]:
    
    memory_components = {
        "Model Memory": calculate_model_memory(transformer, config),
        "KV Cache Memory": calculate_kv_cache_memory(transformer, config),
        "Gradient Memory": calculate_gradient_memory(transformer, config),
        "Activation Memory": calculate_activation_memory(transformer, config),
        "Optimizer Memory": calculate_optimizer_memory(transformer, config),
        "Communication Memory": calculate_communication_memory(transformer, config),
        "Miscellaneous Memory": misc_mem
    }
    
    # Convert all components to GB
    memory_components_gb = {k: v for k, v in memory_components.items()}
    
    # Calculate total training and inference memory
    inference_memory = (memory_components["Model Memory"] + 
                       memory_components["KV Cache Memory"] + 
                       memory_components["Miscellaneous Memory"])
    
    training_memory = sum(memory_components.values()) - memory_components["KV Cache Memory"]
    
    memory_components_gb.update({
        "Total Training Memory (GB)": training_memory,
        "Total Inference Memory (GB)": inference_memory
    })

    return memory_components_gb
