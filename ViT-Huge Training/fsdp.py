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
    global_batch_size = 1800
    grad_accum_steps = 5
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

    # Training and timing setup
    num_epochs = 5
    torch.cuda.synchronize()
    start_time = time.time()


    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        data_iter = iter(dataloader)

        for batch_idx in range(len(dataloader) // grad_accum_steps):
            torch.cuda.synchronize()
            iter_start = time.time()

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

            torch.cuda.synchronize()
            iter_end = time.time()
            iteration_time = iter_end - iter_start

            if rank == 0:
                print(f"Epoch {epoch+1}, Iter {batch_idx}: {iteration_time:.5f} seconds")

    torch.cuda.synchronize()
    if rank == 0:
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE"))
    rank = int(os.getenv("RANK", 0))
    train(rank, world_size)
