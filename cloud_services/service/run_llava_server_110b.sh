#!/bin/bash
#SBATCH --job-name=llava110b
#SBATCH --output=result_llava.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=20000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anxingxiao@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --nodelist=crane5
nvidia-smi
source activate learning
python /data/home/anxing/cloud_services/service/llava/llava_server.py --port 55576 --model_id llava-hf/llava-next-110b-hf --load_in_4bit