#!/bin/bash

# Define the range of nodes to use (adjust as needed)
MIN_NODES=3
MAX_NODES=6
GPUS_PER_NODE=6
BASE_PORT=12000

for NODES in $(seq $MIN_NODES $MAX_NODES); do
    JOB_NAME="deepspeed_${NODES}nodes"
    MASTER_PORT=$((BASE_PORT + (NODES * 10)))

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:$GPUS_PER_NODE
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=10:00:00
#SBATCH --output=${JOB_NAME}-%j.out
#SBATCH --error=${JOB_NAME}-%j.err

set -x

# --- Setup environment ---
export MASTER_ADDR=\$(scontrol show hostname \$SLURM_NODELIST | head -n 1)
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=\$((\$SLURM_NNODES * \$SLURM_GPUS_ON_NODE))
export NCCL_DEBUG=INFO
export NODE_RANK=\$SLURM_NODEID

echo "Running DeepSpeed job on $NODES nodes..."
echo "Master Address: \$MASTER_ADDR"
echo "Master Port: \$MASTER_PORT"
echo "World Size: \$WORLD_SIZE"
echo "Node Rank: \$NODE_RANK"

# --- Generate shared DeepSpeed accelerate config (once) ---
if [[ \$SLURM_NODEID -eq 0 ]]; then
cat <<EOF > accelerate_config.yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_multinode_launcher: standard
  zero_stage: 3
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
distributed_type: DEEPSPEED
dynamo_config:
  dynamo_backend: AOT_CUDAGRAPHS
main_training_function: main
mixed_precision: fp16
machine_rank: \$SLURM_NODEID
main_process_ip: \$MASTER_ADDR
main_process_port: \$MASTER_PORT
num_machines: \$SLURM_JOB_NUM_NODES
num_processes: \$WORLD_SIZE
rdzv_backend: c10d
same_network: true
use_cpu: false
EOF
fi

# --- Wait a moment to ensure config file exists across nodes ---
sleep 10

# --- Launch training ---
srun accelerate launch --config_file=accelerate_config.yaml DeepSpeed.py

EOT

    echo "Submitted DeepSpeed job with $NODES nodes (Port: $MASTER_PORT)!"
    sleep 2
done