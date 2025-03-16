import os
import time
from datetime import timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.profiler
from torchvision import datasets, transforms
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from timm import create_model

def train(rank, world_size):

    # Get local rank from torchrun
    local_rank = int(os.environ["LOCAL_RANK"])

    # NCCL boilerplate
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_TIMEOUT"] = "600"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=300))
    torch.cuda.set_device(local_rank)

    # Global batch size, gradient accumulation steps, per-GPU batch size setup
    global_batch_size = 1080
    grad_accum_steps = 3
    micro_batch_size = global_batch_size // (grad_accum_steps * world_size)

    # Load ViT-Huge
    model = create_model("vit_huge_patch14_224_in21k", pretrained=False, num_classes=101)
    model = model.to(local_rank)

    # Mixed precision policy setup
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    # Wrap model with FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device()
    )

    # Dataset prep (Food101)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = datasets.Food101(root="./data", split="train", download=True, transform=transform)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, sampler=sampler, num_workers=1, pin_memory=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Print "metadata"
    if rank == 0:
        print(f"Using {world_size} GPUs across nodes.")
        print(f"Global Batch Size: {global_batch_size}, Gradient Accum Steps: {grad_accum_steps}")
        print(f"Micro Batch Size per GPU: {micro_batch_size}")

    # Profiling requires only one epoch
    num_epochs = 1
    torch.cuda.synchronize()

    # Profiler setup boilerplate
    profiler_dir = f"./profiler_outputs/fsdp/{world_size}GPUs"
    os.makedirs(profiler_dir, exist_ok=True)
    profiler_summary_path = os.path.join(profiler_dir, "rank0_summary.txt")
    profiler_trace_path = os.path.join(profiler_dir, "rank0_trace.json")

    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir) if rank == 0 else None,
        record_shapes=True,
        with_stack=True
    )

    # Start profiler
    profiler.start()

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        data_iter = iter(dataloader)

        for batch_idx in range(len(dataloader) // grad_accum_steps):
            # Synchronize at start of iteration
            torch.cuda.synchronize()

            optimizer.zero_grad()
            for _ in range(grad_accum_steps):
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    images, labels = next(data_iter)

                images, labels = images.to(local_rank), labels.to(local_rank)
                outputs = model(images)
                loss = criterion(outputs, labels) / grad_accum_steps
                loss.backward()

            optimizer.step()

            # Synchronize at end of iteration
            torch.cuda.synchronize()

            profiler.step()

    profiler.stop()

    # Synchronize at the very very end
    dist.barrier()
    torch.cuda.synchronize()

    # Save profiler summary and trace only for rank 0
    if rank == 0:
        with open(profiler_summary_path, "w") as f:
            f.write(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(f"Profiler summary saved to {profiler_summary_path}")
        print(f"Profiler trace saved to {profiler_trace_path}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE"))
    rank = int(os.getenv("RANK", 0))
    train(rank, world_size)