#VERBOSE=0 torchrun --nproc_per_node 3 self_contained_pp_LOC.py
import os, random, numpy as np, torch, torch.nn as nn, torch.distributed as dist, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

STEP, local_rank, world_size, verbose = 0, int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), os.environ.get("VERBOSE", "0") == "1"

def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class MicroBatchDataLoader(DataLoader):
    def __init__(self, global_batch_size, micro_batch_size, data_parallel_size, seq_length, dataset_name, tokenizer_name, split="train", num_samples=None):
        self.global_batch_size, self.micro_batch_size, self.data_parallel_size, self.seq_length = global_batch_size, micro_batch_size, data_parallel_size, seq_length
        self.local_batch_size = self.global_batch_size // self.data_parallel_size
        self.num_local_micro_batches = self.local_batch_size // self.micro_batch_size
        self.num_global_micro_batches = self.global_batch_size // self.micro_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = load_dataset(dataset_name, split=split)
        if num_samples: self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        dist.barrier()
        self.dataset = self.dataset.map(lambda examples: self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.seq_length + 1, return_special_tokens_mask=False), batched=True, remove_columns=self.dataset.column_names).with_format("torch", columns=["input_ids"])
        super().__init__(self.dataset, batch_size=micro_batch_size, collate_fn=self.collate_batch, pin_memory=True, num_workers=3, sampler=DistributedSampler(self.dataset, num_replicas=data_parallel_size, rank=0, shuffle=False), shuffle=False)

    def collate_batch(self, batch_data):
        batch_input_ids = torch.stack([item['input_ids'] for item in batch_data])
        batch_size, seq_len = batch_input_ids.shape
        return {"input_ids": batch_input_ids[:, :-1].T.contiguous(), "target_ids": batch_input_ids[:, 1:].T.contiguous(), "position_index": torch.arange(seq_len-1, dtype=torch.long).unsqueeze(1).expand(-1, batch_size).contiguous(), "attn_mask": torch.tril(torch.ones((seq_len-1, seq_len-1), dtype=torch.bool)).unsqueeze(0).expand(batch_size, -1, -1).contiguous(), "hidden_states": None}

class ProcessGroupManager:
    def __init__(self, pp_rank, pp_world_size):
        self.pp_rank, self.pp_world_size = pp_rank, pp_world_size
        self.pp_next_rank = None if self.pp_rank == self.pp_world_size - 1 else (self.pp_rank + 1) % self.pp_world_size
        self.pp_prev_rank = None if self.pp_rank == 0 else (self.pp_rank - 1) % self.pp_world_size
        self.is_pipeline_last_stage = self.pp_rank == self.pp_world_size - 1
        self.is_pipeline_first_stage = self.pp_rank == 0
        self.pp_group = dist.new_group(list(range(self.pp_world_size)))

class PipelineParallel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(model_name, config=self.config)
        layer_distribution = self.distribute_layers(self.config.num_hidden_layers)
        self.embed_tokens = base_model.model.embed_tokens if process_group_manager.is_pipeline_first_stage else nn.Identity()
        self.decoder_layers = nn.ModuleDict({str(i): base_model.model.layers[i] for i in layer_distribution})
        self.norm = base_model.model.norm if process_group_manager.is_pipeline_last_stage else nn.Identity()
        self.lm_head = base_model.lm_head if process_group_manager.is_pipeline_last_stage else nn.Identity()
        del base_model

    def distribute_layers(self, num_layers):
        layers_per_gpu = [num_layers // process_group_manager.pp_world_size + (1 if i < num_layers % process_group_manager.pp_world_size else 0) for i in range(process_group_manager.pp_world_size)]
        start_layer = sum(layers_per_gpu[:process_group_manager.pp_rank])
        return list(range(start_layer, start_layer + layers_per_gpu[process_group_manager.pp_rank]))

    def forward(self, batch, device):
        x = batch["hidden_states"].to(device) if batch["hidden_states"] is not None else batch["input_ids"].to(device)
        x = self.embed_tokens(x)
        for layer in self.decoder_layers.values():
            x = layer(x, position_ids=batch["position_index"].to(device))[0]
        x = self.norm(x)
        return self.lm_head(x)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None: input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        return input_tensor.grad if input_tensor is not None else None

def communicate(tensor=None, shapes=None, dtype=None, operation='send_forward'):
    global STEP
    if operation == 'recv_forward':
        if process_group_manager.is_pipeline_first_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device='cuda', dtype=dtype)
        src = process_group_manager.pp_prev_rank
    elif operation == 'send_forward':
        if process_group_manager.is_pipeline_last_stage: return
        dest = process_group_manager.pp_next_rank
    elif operation == 'recv_backward':
        if process_group_manager.is_pipeline_last_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device='cuda', dtype=dtype)
        src = process_group_manager.pp_next_rank
    elif operation == 'send_backward':
        if process_group_manager.is_pipeline_first_stage: return
        dest = process_group_manager.pp_prev_rank
    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)
    if verbose: print(f"{operation} | {'sending' if is_send else 'receiving'} {operation.split('_')[1]} {process_group_manager.pp_rank} {'→' if is_send else '←'} {peer_rank} | STEP:{STEP} | RANK:{process_group_manager.pp_rank}", flush=True)
    [req.wait() for req in dist.batch_isend_irecv([op])]
    torch.cuda.synchronize()
    if verbose: STEP += 1
    return tensor if not is_send else None

def bidirectional_communicate(op, send_tensor, recv_shapes, dtype, device):
    global STEP
    is_fwd = (op == 'send_fwd_recv_bwd')
    if (is_fwd and process_group_manager.is_pipeline_last_stage) or (not is_fwd and process_group_manager.is_pipeline_first_stage): return None
    peer_rank = process_group_manager.pp_next_rank if is_fwd else process_group_manager.pp_prev_rank
    recv_tensor = torch.empty(recv_shapes, requires_grad=True, device=device, dtype=dtype)
    reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, send_tensor, peer_rank), dist.P2POp(dist.irecv, recv_tensor, peer_rank)])
    if verbose: print(f"{op} | sending {'next' if is_fwd else 'prev'} {process_group_manager.pp_rank} -> {peer_rank} | "f"receiving {'next' if is_fwd else 'prev'} {peer_rank} -> {process_group_manager.pp_rank} | "f"STEP {STEP=} | RANK:{process_group_manager.pp_rank}", flush=True)
    [req.wait() for req in reqs]
    torch.cuda.synchronize()
    if verbose: STEP += 1
    return recv_tensor

def pipeline_parallel_1f1b(model, data_loader, tensor_shapes, device):
    num_warmup_microbatches = min(process_group_manager.pp_world_size - process_group_manager.pp_rank - 1, data_loader.num_local_micro_batches)
    num_microbatches_remaining = data_loader.num_local_micro_batches - num_warmup_microbatches
    logging_loss, input_tensors, output_tensors  = 0.0, [], []
    
    def _forward_step(input_tensor):
        batch = next(iter(data_loader))
        batch["hidden_states"] = input_tensor
        output_tensor = model.forward(batch, device)
        if process_group_manager.is_pipeline_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            nonlocal logging_loss
            logging_loss += output_tensor.item()
        return output_tensor

    for _ in range(num_warmup_microbatches): # Warmup forward passes
        input_tensor = communicate(shapes=tensor_shapes, dtype=torch.float32, operation='recv_forward')
        output_tensor = _forward_step(input_tensor)
        communicate(tensor=output_tensor, operation='send_forward')
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    if num_microbatches_remaining > 0:
        input_tensor = communicate(shapes=tensor_shapes, dtype=torch.float32, operation='recv_forward')
    
    for i in range(num_microbatches_remaining):  # 1F1B steady state
        output_tensor = _forward_step(input_tensor)
        output_tensor_grad = bidirectional_communicate('send_fwd_recv_bwd', output_tensor, tensor_shapes, torch.float32, device)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        if i == num_microbatches_remaining - 1: # last iteration
            input_tensor = None
            communicate(tensor=input_tensor_grad, operation='send_backward')
        else:
            input_tensor = bidirectional_communicate('send_bwd_recv_fwd', input_tensor_grad, tensor_shapes, torch.float32, device)

    for _ in range(num_warmup_microbatches): # Cooldown backward passes
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = communicate(shapes=tensor_shapes, dtype=torch.float32, operation='recv_backward')
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        communicate(tensor=input_tensor_grad, operation='send_backward')
    return logging_loss

def pipeline_parallel_afab(model, data_loader, tensor_shapes, device):
    logging_loss, input_tensors, output_tensors = 0.0, [], []
    
    for _ in range(data_loader.num_local_micro_batches): # All forward passes
        input_tensor = communicate(shapes=tensor_shapes, dtype=torch.float32, operation='recv_forward')
        batch = next(iter(data_loader))
        batch["hidden_states"] = input_tensor
        output_tensor = model.forward(batch, device)
        communicate(tensor=output_tensor, operation='send_forward')
        if process_group_manager.is_pipeline_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            logging_loss += output_tensor.item()
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
    
    for _ in range(data_loader.num_local_micro_batches): # All backward passes
        output_tensor_grad = communicate(shapes=tensor_shapes, dtype=torch.float32, operation='recv_backward')
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        communicate(tensor=input_tensor_grad, operation='send_backward')
    
    return logging_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SEQ_LEN, GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, LEARNING_RATE, NUM_SAMPLES, MAX_TOKENS = 10, 6, 2, 1e-4, 20, 1800
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
process_group_manager = ProcessGroupManager(pp_rank=local_rank, pp_world_size=world_size)
set_all_seed(seed=42)
model = PipelineParallel("HuggingFaceTB/SmolLM-360M-Instruct").to(device)
data_loader = MicroBatchDataLoader(GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, 1, SEQ_LEN, "roneneldan/TinyStories", "HuggingFaceTB/SmolLM-360M-Instruct", num_samples=NUM_SAMPLES)
tensor_shapes = (SEQ_LEN, data_loader.micro_batch_size, model.config.hidden_size)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
trained_tokens, step = 0, 0
tokens_per_step = data_loader.num_global_micro_batches * data_loader.micro_batch_size * SEQ_LEN
while trained_tokens < MAX_TOKENS:
    optimizer.zero_grad()
    loss = pipeline_parallel_1f1b(model, data_loader, tensor_shapes, device) #loss = pipeline_parallel_afab(model, data_loader, tensor_shapes, device)
    optimizer.step()
    trained_tokens += tokens_per_step
    step += 1
    if process_group_manager.is_pipeline_last_stage:
        print(f"Step: {step}, Loss: {loss:.4f}, Tokens: {trained_tokens}/{MAX_TOKENS}")
