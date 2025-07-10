LOCAL_MODEL_PATH=$1
GPUS=$2

export model_name=$(basename "$LOCAL_MODEL_PATH")
echo "model_name: $model_name"

NODE_IP=$(hostname -I | awk '{print $1}')
echo "vLLM API Server will run on IP: ${NODE_IP}"

srun -p $PARTITION --gres=gpu:${GPUS} -J ${model_name} \
    python -m vllm.entrypoints.openai.api_server \
    --model ${LOCAL_MODEL_PATH} \
    --dtype auto  \
    --api-key sk-123456 \
    --port 23333 \
    --trust-remote-code \
    --enforce-eager \
    --tensor-parallel-size ${GPUS} \
    --max-num-seqs 16