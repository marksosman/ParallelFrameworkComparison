import os
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

    # Global batch size, gradient accumulation steps, per-GPU batch size setup
    global_batch_size = 360
    grad_accum_steps = 1
    micro_batch_size = global_batch_size // (grad_accum_steps * world_size)

    # Load ViT-Huge
    model = create_model("vit_huge_patch14_224_in21k", pretrained=False, num_classes=101)
    model = model.to(local_rank)

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
        print(f"Using {world_size} GPUs across nodes.")
        print(f"Global Batch Size: {global_batch_size}, Gradient Accum Steps: {grad_accum_steps}")
        print(f"Micro Batch Size per GPU: {micro_batch_size}")

    # Profiling requires only one epoch
    num_epochs = 1
    torch.cuda.synchronize()

    # Profiler setup boilerplate
    profiler_dir = f"./profiler_outputs/deepspeed/{world_size}GPUs"
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

        for batch_idx, (images, labels) in enumerate(dataloader):
            # Synchronize at start of iteration
            torch.cuda.synchronize()

            optimizer.zero_grad()

            # Convert images to half precision
            images, labels = images.to(local_rank).half(), labels.to(local_rank)

            loss = criterion(model(images), labels).float()
            model.backward(loss)
            model.step()

            # Synchronize at end of iteration
            torch.cuda.synchronize()


            # Update profiler
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
    train()