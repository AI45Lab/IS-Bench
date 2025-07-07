LOCAL_MODEL_PATH="/mnt/petrelfs/share_data/ai4good_shared/models/meta-llama/Llama-3.2-11B-Vision-Instruct"
GPUS=4


# LOCAL_MODEL_PATH="/mnt/petrelfs/luxiaoya/code/EAI/og_ego_prim/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct"
# export GPUS=1
# /mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-7B-Instruct
#   /mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-72B-Instruct

#  /mnt/petrelfs/share_data/wangwenhai/internvl/release/InternVL3-38B-Instruct
#  /mnt/petrelfs/share_data/wangwenhai/internvl/release/InternVL3-14B-Instruct
#   /mnt/petrelfs/share_data/wangwenhai/internvl/release/InternVL3-8B-Instruct

#  /mnt/petrelfs/share_data/wangwenhai/internvl/release/InternVL2_5-78B  
#  /mnt/petrelfs/share_data/wangwenhai/internvl/release/InternVL2_5-78B-MPO

#  /mnt/petrelfs/share_data/ai4good_shared/models/meta-llama/Llama-3.2-11B-Vision-Instruct
#  /mnt/petrelfs/share_data/quxiaoye/models/Llama-3.2-90B-Vision-Instruct
export model_name=$(basename "$LOCAL_MODEL_PATH")
echo "model_name: $model_name"

# NODE_IP=$(hostname -I | awk '{print $1}')
# echo "vLLM API Server will run on IP: ${NODE_IP}"
# srun -p AI4Good_L1_p --gres=gpu:${GPUS} -J ${model_name} python -m vllm.entrypoints.openai.api_server --model ${LOCAL_MODEL_PATH} --dtype auto  --api-key sk-123456 --port 23333 --trust-remote-code --enforce-eager --tensor-parallel-size ${GPUS} --max-num-seqs 16
# sbatch -p AI4Good_L1_p --gres=gpu:${GPUS} -J ${model_name} ./entrypoints/vllm_serve_sbatch.sh

srun -p AI4Good_L1_p \
    --gres=gpu:${GPUS} \
    -J llama11b \
python -m vllm.entrypoints.openai.api_server \
    --model ${LOCAL_MODEL_PATH} \
    --limit-mm-per-prompt image=6 \
    --dtype auto \
    --api-key sk-123456 \
    --port 21111 \
    --trust-remote-code \
    --enforce-eager \
    --tensor-parallel-size ${GPUS} \
    --max-num-seqs 16
# srun -p AI4Good_L1_p --gres=gpu:${GPUS} -J ${model_name} python -m vllm.entrypoints.openai.api_server --model ${LOCAL_MODEL_PATH} --dtype auto  --api-key sk-123456 --port 23333 --trust-remote-code --enforce-eager --tensor-parallel-size ${GPUS} --max-num-seqs 16
