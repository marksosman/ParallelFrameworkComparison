#!/bin/bash
#SBATCH --job-name=fsdp           # Job name
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Only one task-per-node is EVER needed
#SBATCH --gres=gpu:6              # Request 6 GPUs per node
#SBATCH --cpus-per-task=1         # CPU cores per task
#SBATCH --mem=512G                # Memory per node
#SBATCH --time=10:00:00           # Max time (hh:mm:ss)
#SBATCH --output=fsdp-1-6-%j.out  # Standard output log
#SBATCH --error=fsdp-1-6-%j.err   # Error log

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355  # Choose an available port
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))

# Run the script using torchrun (recommended for PyTorch 1.10+)
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_ON_NODE --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d fsdp.py