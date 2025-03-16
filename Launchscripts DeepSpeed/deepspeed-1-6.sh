#!/bin/bash
#SBATCH --job-name=deepspeed            # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Only one task per node is EVER needed
#SBATCH --gres=gpu:6                    # Request 6 GPUs per node
#SBATCH --cpus-per-task=1               # CPU cores per task
#SBATCH --mem=512G                      # Memory per node
#SBATCH --time=10:00:00                 # Max time (hh:mm:ss)
#SBATCH --output=DeepSpeed-1-6-%j.out   # Standard output log
#SBATCH --error=DeepSpeed-1-6-%j.err    # Error log

set -x

# Set environment variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355
export GPUS_PER_NODE=6
export WORLD_SIZE=$(($SLURM_NNODES * $GPUS_PER_NODE))
export NCCL_DEBUG=INFO

# Generate DeepSpeed config dynamically
srun bash -c "cat <<EOT > \"accelerate_config_\"\$SLURM_JOB_ID\"_\$(hostname -s).yaml\"
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 8
  steps_per_print: 100
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
dynamo_config:
  dynamo_backend: AOT_CUDAGRAPHS
main_training_function: main
mixed_precision: bf16
machine_rank: '\$SLURM_NODEID'
main_process_ip: '\$MASTER_ADDR'
main_process_port: \$MASTER_PORT
num_machines: \$SLURM_JOB_NUM_NODES
num_processes: \$WORLD_SIZE
rdzv_backend: c10d
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOT"

# Run training with DeepSpeed
srun bash -c "accelerate launch --config_file=accelerate_config_\"\$SLURM_JOB_ID\"_\$(hostname -s).yaml DeepSpeed.py"