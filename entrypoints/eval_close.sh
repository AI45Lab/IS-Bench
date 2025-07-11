#!/bin/bash
export NUM_GPUS=1
export PYTHONPATH=./:$PYTHONPATH

mkdir -p logs
START_TIME=`date +%Y%m%d-%H:%M:%S`
LOG_FILE=logs/exec_$START_TIME.log

if [ -f "entrypoints/env.sh" ]; then
    source entrypoints/env.sh
fi

if [[ -v PARTITION ]]; then
    echo "Submit to $PARTITION"
fi

MODEL_NAME=$1
DATA_PARALLEL=${2:-1}

WORK_DIR=./results/$MODEL_NAME

python -m og_ego_prim.cli.online_benchmark_all \
    --data_parallel $DATA_PARALLEL \
    --task_list entrypoints/task_list.txt \
    --work_dir $WORK_DIR \
    --model $MODEL_NAME \
    --draw_bbox_2d \
    --prompt_setting 'v1' \
    2>&1 | tee -a "$LOG_FILE" > /dev/null & 

sleep 0.5s
tail -f $LOG_FILE