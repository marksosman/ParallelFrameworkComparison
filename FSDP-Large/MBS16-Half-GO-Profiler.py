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
    local_rank = int(os.environ["LOCAL_RANK"])

    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_TIMEOUT"] = "900"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_P2P_DISABLE"] = "0"

    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=600))
    torch.cuda.set_device(local_rank)

    grad_accum_steps = 1
    micro_batch_size = 16

    model = create_model("vit_large_patch16_224", pretrained=False, num_classes=101)
    model = model.to(local_rank)

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device()
    )

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if rank == 0:
        print(f"Using {world_size} GPUs across nodes.")
        print(f"Global Batch Size: {micro_batch_size * grad_accum_steps * world_size}")
        print(f"Micro Batch Size per GPU: {micro_batch_size}")

    torch.cuda.synchronize()

    profiler_dir = os.path.join(
        "./profiler_outputs",
        "fsdp",
        f"MBS{micro_batch_size}-Half-GO",
        f"{world_size}GPUs"
    )
    os.makedirs(profiler_dir, exist_ok=True)
    profiler_summary_path = os.path.join(profiler_dir, "rank0_summary.txt")

    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir) if rank == 0 else None,
        record_shapes=True,
        with_stack=True
    )

    profiler.start()
    model.train()
    data_iter = iter(dataloader)
    profiled_steps = 0

    while profiled_steps < 5:
        torch.cuda.synchronize()
        optimizer.zero_grad()

        for _ in range(grad_accum_steps):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                images, labels = next(data_iter)

            images, labels = images.to(local_rank).half(), labels.to(local_rank)
            outputs = model(images)
            loss = criterion(outputs, labels).float() / grad_accum_steps
            loss.backward()

        optimizer.step()
        torch.cuda.synchronize()
        profiler.step()
        profiled_steps += 1

    profiler.stop()
    dist.barrier()
    torch.cuda.synchronize()

    if rank == 0:
        with open(profiler_summary_path, "w") as f:
            f.write(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(f"Profiler summary saved to {profiler_summary_path}")
        print(f"TensorBoard trace directory: {profiler_dir}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE"))
    rank = int(os.getenv("RANK", 0))
    train(rank, world_size)
