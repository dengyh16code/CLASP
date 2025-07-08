#!/bin/bash
#SBATCH --job-name=server
#SBATCH --output=result_all.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=20000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anxingxiao@gmail.com
#SBATCH --gres=gpu:2
#SBATCH --nodelist=crane5
nvidia-smi
source activate learning

python /data/home/anxing/cloud_services/service/llava/llava_server_hf_110b.py &
python /data/home/anxing/cloud_services/service/owl_vit/server.py &
python /data/home/anxing/cloud_services/service/sam/server.py &

wait