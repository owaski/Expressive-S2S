#!/bin/bash
#SBATCH --job-name=indextts_8gpu_training
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/indextts_8gpu_%j_%x.out
#SBATCH --error=logs/indextts_8gpu_%j_%x.err
sbatch index-tts/scripts/run_training_8gpu.slurm
# Job information
echo "=========================================="
echo "IndexTTS 8-GPU Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_GPUS_PER_NODE * SLURM_NNODES))"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE MB"
echo "Time Limit: $SLURM_TIMELIMIT"
echo "Start Time: $(date)"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Change to project directory
cd /data/user_data/willw2/course_project_repo/Expressive-S2S/ || {
    echo "ERROR: Could not change to project directory"
    exit 1
}

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "index-tts/.venv/bin/activate" ]; then
    source index-tts/.venv/bin/activate
else
    echo "ERROR: Virtual environment not found at index-tts/.venv/bin/activate"
    exit 1
fi

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi

# Set environment variables for optimal performance
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH=/data/user_data/willw2/course_project_repo/Expressive-S2S/index-tts:$PYTHONPATH

# Set WandB API key (if needed), the following is Will Wang's API key
export WANDB_API_KEY=

# Log system information
echo "System Information:"
echo "- Python: $(python --version)"
echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "- CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "- GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================="

echo "Starting IndexTTS training with 8 GPUs..."

# Run the training with srun for proper SLURM integration
# For 8-GPU DDP training across 1 nodes: 8 tasks per node, 1 GPU per task
srun python index-tts/src/train.py \
--config index-tts/config/finetune_paraspeechcaps_stress17k_config.yaml