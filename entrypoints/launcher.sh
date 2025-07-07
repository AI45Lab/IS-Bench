LAUNCHER=()

if [[ -v PARTITION ]]; then
    echo "Submit to $PARTITION"
    export OMNIGIBSON_HEADLESS=1
    echo "HEADLESS=1"
    LAUNCHER+=(
        "srun" "-p" $PARTITION
        # "--debug"
        "--gres=gpu:1"
        "-N1"
    )
fi

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