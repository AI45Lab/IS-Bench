#!/bin/bash
TASK_NAME=$1
SCENE_NAME=$2
MODEL_NAME_OR_PATH=$3

# export OMNIGIBSON_HEADLESS=1
export NUM_GPUS=1
export PYTHONPATH=./:$PYTHONPATH

mkdir -p logs
START_TIME=`date +%Y%m%d-%H:%M:%S`
LOG_FILE=logs/exec_$START_TIME.log

if [ -f "entrypoints/env.sh" ]; then
    source entrypoints/env.sh
fi

source entrypoints/launcher.sh
LAUNCHER+=(
    "python" "-m" "og_ego_prim.cli.online_benchmark_once"
    "--task" $TASK_NAME
    "--scene" $SCENE_NAME
    "--model" $MODEL_NAME_OR_PATH
    "--draw_bbox_2d"
    # "--not_eval_awareness"
    # "--debug"
)

"${LAUNCHER[@]}" 2>&1 | tee -a "$LOG_FILE" > /dev/null &
sleep 0.5s
tail -f $LOG_FILE
