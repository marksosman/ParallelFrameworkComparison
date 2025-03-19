#!/bin/bash

# Define the range of nodes to use (adjust as needed)
MIN_NODES=3
MAX_NODES=6
GPUS_PER_NODE=6  # Adjust based on your setup
BASE_PORT=12000  # Starting base port for unique comms

for NODES in $(seq $MIN_NODES $MAX_NODES); do
    JOB_NAME="fsdp_${NODES}nodes"
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

# --- Dynamic Environment Setup ---
export MASTER_ADDR=\$(scontrol show hostname \$SLURM_NODELIST | head -n 1)
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=\$((\$SLURM_NNODES * \$SLURM_GPUS_ON_NODE))
export NCCL_DEBUG=INFO
export NODE_RANK=\$SLURM_NODEID

echo "Running FSDP job on $NODES nodes..."
echo "Master Address: \$MASTER_ADDR"
echo "Master Port: \$MASTER_PORT"
echo "World Size: \$WORLD_SIZE"
echo "Node Rank: \$NODE_RANK"

# --- Run Multi-Node FSDP Training ---
srun torchrun \
    --nnodes=\$SLURM_NNODES \
    --nproc_per_node=\$SLURM_GPUS_ON_NODE \
    --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
    --rdzv_backend=c10d \
    fsdp.py
EOT

    echo "Submitted FSDP job with $NODES nodes (Port: $MASTER_PORT)!"
    sleep 2
done
