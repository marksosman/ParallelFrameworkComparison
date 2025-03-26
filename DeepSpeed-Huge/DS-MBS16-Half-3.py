import os
import time
import deepspeed
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from timm import create_model
import torch.profiler

def train():

    # Basic DeepSpeed setup
    deepspeed.init_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    # NCCL configs
    # Blocking NCCL operations, will finish before anything continues
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    # Async error handling, errors get handled "immediately"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    # Timeout, set larger with larger models
    os.environ["NCCL_TIMEOUT"] = "900"
    # Infiniband comms (instead of TCP/IP), set to 1 if no Infiniband on cluster
    os.environ["NCCL_IB_DISABLE"] = "0"
    # Direct GPU 2 GPU comms, set to 1 if only 1 node OR 0 no NVLink on cluster
    os.environ["NCCL_P2P_DISABLE"] = "0"

    # Gradient accumulation steps, per-GPU batch size setup
    grad_accum_steps = 1
    micro_batch_size = 16

    # Load ViT-Huge
    model = create_model("vit_huge_patch14_224_in21k", pretrained=False, num_classes=101)
    model = model.to(local_rank)

    # Measure model size prior to sharding
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    if rank == 0:
        print(f"Total Parameters: {total_params}")
        print(f"Model Size: {total_size_mb:.2f} MB")

    # DeepSpeed config
    ds_config = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        },
        "zero_optimization": {
            "stage": 3
        },
        "fp16": {
            "enabled": True
        }
    }

    # Wrap model with DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=ds_config
    )

    ds_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ds_size_mb = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1e6
    if rank == 0:
        print(f"DeepSpeed: Total Parameters: {ds_params:,}")
        print(f"DeepSpeed: Model Size: {ds_size_mb:.2f} MB")

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

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Print "metadata"
    if rank == 0:
        # Training metadata
        print(f"Using {world_size} GPUs across nodes.")
        print(f"Global Batch Size: {micro_batch_size*grad_accum_steps*world_size}, Gradient Accum Steps: {grad_accum_steps}")
        print(f"Micro Batch Size per GPU: {micro_batch_size}")
        # Memory metadata (in MB)
        total_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1e6
        allocated = torch.cuda.memory_allocated(local_rank) / 1e6
        reserved = torch.cuda.memory_reserved(local_rank) / 1e6
        print(f"Total Memory: {total_memory:.2f} MB")
        print( f"Allocated: {allocated:.2f} MB")
        print(f"Reserved: {reserved:.2f} MB")

    # Training and timing setup
    num_iterations = 40
    torch.cuda.synchronize()
    start_time = time.time()

    torch.cuda.empty_cache()

    model.train()
    data_iter = iter(dataloader)
    for iteration in range(num_iterations):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, labels = next(data_iter)

        torch.cuda.synchronize()
        iter_start = time.time()

        optimizer.zero_grad()

        images, labels = images.to(local_rank).half(), labels.to(local_rank)

        loss = criterion(model(images), labels).float()
        model.backward(loss)
        model.step()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        iter_end = time.time()
        iteration_time = iter_end - iter_start

        if rank == 0:
            allocated = torch.cuda.memory_allocated(local_rank) / 1e6
            reserved = torch.cuda.memory_reserved(local_rank) / 1e6
            print(f"Iter {iteration+1}: {iteration_time:.5f} sec, Allocated {allocated: 2f} MB, Reserved {reserved: 2f} MB")


    torch.cuda.synchronize()
    if rank == 0:
        allocated = torch.cuda.memory_allocated(local_rank) / 1e6
        reserved = torch.cuda.memory_reserved(local_rank) / 1e6
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        print(f"Allocated memory: {allocated: .2f} MB, Reserved memory: {reserved: .2f}")
        print(torch.cuda.memory_summary(local_rank))

    dist.destroy_process_group()

if __name__ == "__main__":
    train()