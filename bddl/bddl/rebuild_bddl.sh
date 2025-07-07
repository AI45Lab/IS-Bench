export OMNIGIBSON_ROOT=/mnt/petrelfs/luxiaoya/code/EAI/apptainer/omnigibson
export APPTAINER_IMAGE=/mnt/petrelfs/luxiaoya/code/EAI/apptainer/rekep3.sif
export MAMBA_ROOT_PREFIX=/micromamba

PYTHONPATH=./:$PYTHONPATH \
OMNIGIBSON_HEADLESS=1 \
srun -p AI4Good_P \
    --gres=gpu:0 \
apptainer run \
    --nv \
    --bind /mnt:/mnt,/etc:/etc,$OMNIGIBSON_ROOT/isaac-sim:/isaac-sim,$OMNIGIBSON_ROOT/src:/omnigibson-src \
    $APPTAINER_IMAGE \
micromamba run -n omnigibson \
python data_generation/generate_datafiles.py 
