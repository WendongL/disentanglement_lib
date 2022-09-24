# !/bin/sh
. /home/wliang/miniconda3/etc/profile.d/conda.sh
conda activate tf

echo $CONDA_PREFIX
# module load cuda/11.3/ --gpus 0 1 2 3 --per-gpu 8 --procs-no 32 --results-path results
module load cuda/11.3
python --version
cd /home/wliang/miniconda3/envs/tf/lib/python3.7/site-packages/disentanglement_lib/src/
liftoff liftoff_postprocess.py ./results/2022Sep19-141845_post_hsicbetavae/ --gpus 0 --per-gpu 1 --procs-no 1 --results-path results