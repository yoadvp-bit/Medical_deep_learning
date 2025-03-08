import os

learning_rates = [0.0001, 0.001, 0.01]
optimizer_set = ['adam']
batch_set = [16, 32, 64]
convolutional_channels = [[1,2], [16, 32], [32, 64], [64, 128]]

for lr in learning_rates:
    for optimizer in optimizer_set:
        for batch_size in batch_set:
            for conv_channels in convolutional_channels:
                experiment_name = f"lr{lr}_opt{optimizer}_bs{batch_size}_ch{conv_channels}"
                slurm_script = f"""#!/bin/bash
#SBATCH --job-name={experiment_name}
#SBATCH --output=slurm_logs/{experiment_name}.out
#SBATCH --error=slurm_logs/{experiment_name}.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com

module load python/3.8
module load cuda/11.3

source ~/your_env/bin/activate

python main_CNN.py --optimizer_lr {lr} --batch_size {batch_size} --model_name custom_convnet --optimizer_name {optimizer} --Conv_Channels {','.join(map(str, conv_channels))} --dropout_rate 0 --max_epochs 4 --experiment_name {experiment_name} --checkpoint_folder_save checkpoints/
"""

                os.makedirs("slurm_jobs", exist_ok=True)
                os.makedirs("slurm_logs", exist_ok=True)

                script_path = f"slurm_jobs/{experiment_name}.slurm"
                with open(script_path, "w") as f:
                    f.write(slurm_script)

                os.system(f"sbatch {script_path}")
