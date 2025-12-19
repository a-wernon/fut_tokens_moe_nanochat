"""
Finetune a pretrained model. Supports:
1. Continued training with the same architecture
2. Converting from classical MoE to lookforward MoE
3. Converting from single-token to multi-token prediction

Run as:
    python -m scripts.base_finetune --source_model_tag=moe_d12 --source_step=7080 --moe_type=lookforward --n_predict_tokens=2

or distributed as:
    torchrun --nproc_per_node=8 -m scripts.base_finetune --source_model_tag=moe_d12 --moe_type=lookforward
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# User settings

# Source checkpoint
source_model_tag = "" # model tag of the checkpoint to finetune from (required)
source_step = -1 # step of the checkpoint to load (-1 = last step)

# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect)

# Target model architecture (defaults to source architecture, can override)
moe_type = "" # "classical" or "lookforward" (empty => same as source)
n_predict_tokens = -1 # number of tokens to predict ahead (-1 => same as source)

# Training horizon
num_iterations = -1 # explicit number of steps (-1 = disable)
target_flops = -1.0 # calculate iterations from target flops (-1 = disable)
target_param_data_ratio = 5 # data:param ratio for finetuning (lower than pretraining)

# Optimization
device_batch_size = 64 # per-device batch size
total_batch_size = 262144 # smaller batch for finetuning
embedding_lr = 0.05 # lower LR for finetuning
unembedding_lr = 0.001
weight_decay = 0.0
matrix_lr = 0.005 # lower LR for finetuning
grad_clip = 1.0
warmup_ratio = 0.1 # more warmup for finetuning
warmdown_ratio = 0.2
final_lr_frac = 0.0

# Evaluation
eval_every = 100
eval_tokens = 10*524288
core_metric_every = 500
core_metric_max_per_task = 500
sample_every = 500
save_every = -1

# Output
model_tag = "" # output model tag (empty => auto-generate from source + changes)

# Weight initialization for new parameters
init_posterior_from_prior = True # initialize gate_posterior from gate_prior (vs random)
init_new_heads_from_first = True # initialize new lm_heads from first head (vs random)

# now allow CLI to override settings
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Validate inputs
assert source_model_tag, "source_model_tag is required"

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# Load source checkpoint metadata
base_dir = get_base_dir()
source_checkpoint_dir = os.path.join(base_dir, "base_checkpoints", source_model_tag)
if source_step == -1:
    source_step = find_last_step(source_checkpoint_dir)
print0(f"Loading source checkpoint from {source_checkpoint_dir} step {source_step}")

# Load source model data and metadata
source_model_data, _, source_meta = load_checkpoint(source_checkpoint_dir, source_step, device, load_optimizer=False)
source_model_data = {k.removeprefix("_orig_mod."): v for k, v in source_model_data.items()}
source_config = source_meta["model_config"].copy()

# Handle old GPTMoEConfig checkpoints
if "n_expert" in source_config and "use_moe" not in source_config:
    print0("Detected old GPTMoEConfig checkpoint, adding compatibility fields")
    source_config["use_moe"] = True
    source_config["moe_type"] = "classical"
    source_config.setdefault("alpha", 0.01)
    source_config.setdefault("moe_kl_loss_coef", 1.0)
    source_config.setdefault("n_predict_tokens", 1)
    source_config.setdefault("use_liger_kernel", False)

print0(f"Source model config: {source_config}")

# Determine target config (inherit from source, then override)
target_config = source_config.copy()
source_moe_type = source_config.get("moe_type", "classical")
source_n_predict_tokens = source_config.get("n_predict_tokens", 1)

if moe_type:
    target_config["moe_type"] = moe_type
if n_predict_tokens > 0:
    target_config["n_predict_tokens"] = n_predict_tokens

target_moe_type = target_config.get("moe_type", "classical")
target_n_predict_tokens = target_config.get("n_predict_tokens", 1)

# Determine what architecture changes are being made
converting_moe = source_moe_type != target_moe_type
converting_heads = source_n_predict_tokens != target_n_predict_tokens

print0(f"Target model config: {target_config}")
print0(f"Converting MoE: {source_moe_type} -> {target_moe_type}" if converting_moe else "MoE type unchanged")
print0(f"Converting heads: {source_n_predict_tokens} -> {target_n_predict_tokens}" if converting_heads else "Prediction heads unchanged")

# Generate output model tag if not provided
if not model_tag:
    model_tag = f"{source_model_tag}_ft"
    if converting_moe:
        model_tag += f"_{target_moe_type}"
    if converting_heads:
        model_tag += f"_npt{target_n_predict_tokens}"

print0(f"Output model tag: {model_tag}")

# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
assert vocab_size == target_config["vocab_size"]

# -----------------------------------------------------------------------------
# Build target model and transfer weights

max_seq_len = target_config["sequence_len"]
print0(f"Building target model...")
with torch.device("meta"):
    model_config = GPTConfig(**target_config)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# Weight transfer logic
def transfer_weights(source_data, model, source_moe_type, target_moe_type, 
                     source_n_predict, target_n_predict, 
                     init_posterior_from_prior, init_new_heads_from_first):
    """
    Transfer weights from source checkpoint to target model.
    Handles MoE type conversion and multi-token prediction head conversion.
    """
    target_state = model.state_dict()
    transferred = 0
    initialized_new = 0
    
    # Build mapping from source keys to target keys
    key_mapping = {}
    for key in source_data.keys():
        new_key = key
        
        # Handle MoE gate conversion: classical -> lookforward
        if source_moe_type == "classical" and target_moe_type == "lookforward":
            if ".moe.gate.weight" in key:
                # gate -> gate_prior
                new_key = key.replace(".moe.gate.weight", ".moe.gate_prior.weight")
        
        # Handle MoE gate conversion: lookforward -> classical
        elif source_moe_type == "lookforward" and target_moe_type == "classical":
            if ".moe.gate_prior.weight" in key:
                new_key = key.replace(".moe.gate_prior.weight", ".moe.gate.weight")
            elif ".moe.gate_posterior.weight" in key:
                continue  # Skip posterior gate when going back to classical
        
        # Handle lm_head -> lm_heads conversion
        if source_n_predict == 1 and target_n_predict > 1:
            if key == "lm_head.weight":
                new_key = "lm_heads.0.weight"
        
        # Handle lm_heads -> lm_head conversion
        if source_n_predict > 1 and target_n_predict == 1:
            if key == "lm_heads.0.weight":
                new_key = "lm_head.weight"
            elif key.startswith("lm_heads.") and key != "lm_heads.0.weight":
                continue  # Skip extra heads
        
        key_mapping[key] = new_key
    
    # Transfer mapped weights
    for source_key, target_key in key_mapping.items():
        if target_key in target_state:
            if source_data[source_key].shape == target_state[target_key].shape:
                target_state[target_key] = source_data[source_key]
                transferred += 1
            else:
                print0(f"Shape mismatch for {source_key} -> {target_key}: "
                       f"{source_data[source_key].shape} vs {target_state[target_key].shape}")
    
    # Initialize new parameters that weren't transferred
    
    # gate_posterior for lookforward MoE (new parameter)
    if target_moe_type == "lookforward" and source_moe_type == "classical":
        for key in target_state.keys():
            if ".moe.gate_posterior.weight" in key:
                prior_key = key.replace(".moe.gate_posterior.weight", ".moe.gate_prior.weight")
                if init_posterior_from_prior and prior_key in target_state:
                    print0(f"Initializing {key} from {prior_key}")
                    target_state[key] = target_state[prior_key].clone()
                else:
                    print0(f"Randomly initializing {key}")
                    # Keep the init_weights initialization
                initialized_new += 1
    
    # New lm_heads for multi-token prediction
    if target_n_predict > 1 and source_n_predict == 1:
        for i in range(1, target_n_predict):
            key = f"lm_heads.{i}.weight"
            if key in target_state:
                if init_new_heads_from_first:
                    print0(f"Initializing {key} from lm_heads.0.weight")
                    target_state[key] = target_state["lm_heads.0.weight"].clone()
                else:
                    print0(f"Randomly initializing {key}")
                initialized_new += 1
    
    model.load_state_dict(target_state, strict=True)
    print0(f"Transferred {transferred} parameters, initialized {initialized_new} new parameters")
    return model

model = transfer_weights(
    source_model_data, model, 
    source_moe_type, target_moe_type,
    source_n_predict_tokens, target_n_predict_tokens,
    init_posterior_from_prior, init_new_heads_from_first
)
del source_model_data  # Free memory

model.train()
orig_model = model
use_moe = target_config.get("use_moe", False)
compile_dynamic = use_moe
model = torch.compile(model, dynamic=compile_dynamic)

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# -----------------------------------------------------------------------------
# Calculate training horizon

tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Gradient accumulation steps: {grad_accum_steps}")

assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated iterations from data:param ratio: {num_iterations:,}")

total_tokens = total_batch_size * num_iterations
print0(f"Total finetuning tokens: {total_tokens:,}")

# -----------------------------------------------------------------------------
# Initialize Optimizer (fresh optimizer state for finetuning)

optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr, 
    embedding_lr=embedding_lr, 
    matrix_lr=matrix_lr, 
    weight_decay=weight_decay
)
adamw_optimizer, muon_optimizer = optimizers

# -----------------------------------------------------------------------------
# DataLoaders

train_loader = tokenizing_distributed_data_loader_with_state(device_batch_size, max_seq_len, split="train", device=device, resume_state_dict=None)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# Schedulers

def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# TensorBoard
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", model_tag)
tb_logger = SummaryWriter(log_dir=os.path.join(base_dir, "tb_logs", "base", model_tag))

# -----------------------------------------------------------------------------
# Loop state

step = 0
min_val_bpb = float("inf")
smooth_train_loss = 0
smooth_train_scalars = defaultdict(lambda: 0)
total_training_time = 0

# -----------------------------------------------------------------------------
# Training loop

while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Validation
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        tb_logger.add_scalar("val/bpb", val_bpb, step)
        model.train()

    # CORE metric
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        tb_logger.add_scalar("core_metric", results["core_metric"], step)
        for k, v in results["centered_results"].items():
            tb_logger.add_scalar(f"centered_results/{k}", v, step)
        model.train()

    # Sampling
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # Save checkpoint
    if last_step or (step > 0 and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": target_config,
                "user_config": user_config,
                "source_checkpoint": {
                    "model_tag": source_model_tag,
                    "step": source_step,
                },
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "smooth_train_scalars": dict(smooth_train_scalars),
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # Training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss, scalars = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)
    
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item()
    
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for group in muon_optimizer.param_groups:
        group["momentum"] = get_muon_momentum(step)
    
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    for k, v in scalars.items():
        smooth_train_scalars[k] = ema_beta * smooth_train_scalars[k] + (1 - ema_beta) * v
    debiased_smooth_scalars = {k: v / (1 - ema_beta**(step + 1)) for k, v in smooth_train_scalars.items()}
    
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
    if step > 10:
        total_training_time += dt
    
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f}%")
    
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            **{f"train/{k}": v for k, v in debiased_smooth_scalars.items()},
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        for k, v in log_data.items():
            tb_logger.add_scalar(k, v, step)

    step += 1

# Print final stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model finetuning", data=[
    user_config,
    {
        "Source checkpoint": f"{source_model_tag} step {source_step}",
        "Source MoE type": source_moe_type,
        "Target MoE type": target_moe_type,
        "Source n_predict_tokens": source_n_predict_tokens,
        "Target n_predict_tokens": target_n_predict_tokens,
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "DDP world size": ddp_world_size,
    },
    {
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

compute_cleanup()

