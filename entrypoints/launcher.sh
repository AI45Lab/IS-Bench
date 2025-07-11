LAUNCHER=()

export OMNIGIBSON_HEADLESS=1
echo "HEADLESS=1"

#### launcher for slurm ####
if [[ -v PARTITION ]]; then
    echo "Submit to $PARTITION"   
    LAUNCHER+=(
        "srun" "-p" $PARTITION
        "--gres=gpu:1"
        "-N1"
    )
fi

#### launcher for apptainer or docker ####
if [[ -v APPTAINER_IMAGE ]]; then
    LAUNCHER+=("apptainer" "run" "--nv")
    if [[ -n "${BINDING}" ]]; then
        LAUNCHER+=("--bind" "${BINDING}")
    fi
    LAUNCHER+=("${APPTAINER_IMAGE}")
fi

# if [[ -v MAMBA_ROOT_PREFIX ]]; then
#     LAUNCHER+=("micromamba" "run" "-n" "omnigibson")
# fi