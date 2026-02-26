#!/bin/bash
#SBATCH --job-name=distribution_estimation
#SBATCH --output=/data/user_data/willw2/logs/job_%j_%x.out
#SBATCH --error=/data/user_data/willw2/logs/job_%j_%x.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00


mkdir -p /data/user_data/willw2/logs

# activate virtual environment
cd /data/user_data/willw2/CosyVoice
source .venv/bin/activate
# direct to working directory
cd /data/user_data/willw2/course_project_repo/Expressive-S2S/index-tts/data_processing/tts_data_synth

# check GPU availability
nvidia-smi

# Delete stale TRT engine plans so they rebuild for the current GPU
echo "Removing stale TRT engine plans..."
find /data/user_data/willw2/CosyVoice/pre_trained_models/Fun-CosyVoice3-0.5B \( -name "*.plan" -o -name "*.trt" -o -name "*.engine" \) -delete
echo "TRT engine plans removed."

echo "Starting the job..."

python gen_data_in_distribution.py --target_language_code en --num_data 16

python gen_data_in_distribution.py --target_language_code zh --num_data 16
