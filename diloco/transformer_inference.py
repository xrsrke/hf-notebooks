def calc_inference_memory(
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    sequence_length: int,
    num_parameters: int,
    bytes_per_param
) -> dict:
    """
    Calculate memory requirements for transformer inference.
    
    Returns a dictionary with memory requirements in GB for:
    - weights: Model weights
    - kv_cache: Key-value cache memory
    - activations: Activation memory for forward pass
    - total: Total memory required
    """
    
    # Basic constants
    # bytes_per_param = 2 if fp16 else 4
    GB = 1024**3
    
    # Model weights memory
    weights_mem = num_parameters * bytes_per_param
    
    # KV cache memory
    # 2 for keys and values, each head has (hidden_size/heads) dimension
    kv_cache_mem = (
        bytes_per_param * 
        2 * 
        num_layers * 
        num_attention_heads * 
        (hidden_size / num_attention_heads) * 
        sequence_length
    )
    
    # Activation memory (simplified estimate for forward pass)
    # Core attention operations + feed forward
    activations_mem = (
        sequence_length * 
        hidden_size * 
        num_layers * 
        (10 + 24) * 
        bytes_per_param
    )
    
    return {
        "weights": weights_mem / GB,
        "kv_cache": kv_cache_mem / GB,
        "activations": activations_mem / GB,
        "total": (weights_mem + kv_cache_mem + activations_mem) / GB
    }
