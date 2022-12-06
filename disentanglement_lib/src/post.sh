# !/bin/sh
. /home/wliang/miniconda3/etc/profile.d/conda.sh
conda activate tf

echo $CONDA_PREFIX
# module load cuda/11.3/ --gpus 0 1 2 3 --per-gpu 8 --procs-no 32 --results-path results
module load cuda/11.3
python --version
cd /home/wliang/miniconda3/envs/tf/lib/python3.7/site-packages/disentanglement_lib/src/
liftoff liftoff_postprocess.py ./results/2022Oct02-095204_post_hsicdipvae/ --gpus 0 --per-gpu 2 --procs-no 2 --results-path results
