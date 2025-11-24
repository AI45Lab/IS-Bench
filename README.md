<h1>IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks</h1>

<span style="color:red">📢 <strong><i>We are currently organizing the code for IS-Bench. If you are interested in our work, please star ⭐ our project.</i></strong></span>

<a href='https://www.arxiv.org/abs/2506.16402'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://github.com/AI45Lab/IS-Bench'><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://huggingface.co/datasets/Ursulalala/IS_Bench_scenes'><img src='https://img.shields.io/badge/🤗-Dataset-blue'></a>
</a>

<h2 id="updates">🔥 Updates</h2>

📆[2025-11-08] 🎈 Our paper has been accepted to AAAI Conference. See you in Singapore~ 🎈

📆[2025-07-07] 🎈 Our paper, code and dataset are released! 🎈

<h2 id="IS-Bench">🎉 Introduction</h2>

<div align="center">
<img src="assets/motivation.png" width="900" alt="Intro_img">
</div>

Existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. 

<h2 id="IS-Bench">📍 Results of IS-Bench </h2>

<div align="center">
<img src="assets/results.png" width="900" alt="Intro_img">
</div>

Our experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. 

<h2 id="dataset">⚙️ Installation</h2>

### System Requirements of Omnigibson
```
OS: Linux (Ubuntu 20.04+), Windows 10+

RAM: 32GB+ recommended

VRAM: 8GB+

GPU: NVIDIA RTX 2080+
```

1. Install Vulkan
```bash
sudo apt update
sudo apt install vulkan-utils libvulkan1
vulkaninfo | grep "Vulkan Instance Version" # test
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:$PATH
```

2. Install Omnigibson
``` bash
conda create -n isbench python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 "numpy<2" -c pytorch -c nvidia
conda activate isbench
pip install omnigibson==1.1.1 --index-url https://pypi.org/simple
python -m pip install --force-reinstall numpy scipy --index-url https://pypi.org/simple 
python -m omnigibson.install    # install omnigibson assets and datasets
```

If you want to use Omnigibson in Docker, please see this [document](https://behavior.stanford.edu/omnigibson/getting_started/installation.html#__tabbed_1_1):

```bash
docker pull stanfordvl/omnigibson:1.1.1 # this image has already contained Vulkan
docker run stanfordvl/omnigibson:1.1.1
micromamba run -n omnigibson pip install python -m omnigibson.install # install dataset
```

3. Download Source Code and BDDL of IS-Bench
``` bash
git clone https://github.com/AI45Lab/IS-Bench
cd IS-Bench
pip install -r requirements.txt # for docker image: micromamba run -n omnigibson pip install -r requirements.txt
pip install -e ./bddl # for docker image: micromamba run -n omnigibson pip install -e ./bddl
```

4. Download Scene Dataset
``` bash
cd ../data
wget https://huggingface.co/datasets/Ursulalala/IS_Bench_scenes/resolve/main/scenes.tar.gz
tar -xzvf scenes.tar.gz
rm scenes.tar.gz
```

<h2 id="quick-start">🚀 Usage</h2>

If you are using slurm to run IS-Bench, please first revise your launcher for benchmark at scripts/launcher.sh

### Evaluate Close-Source Models
Our code supports api-based model with openai or google-genai format.

1. Configure api_base and api_key in `entrypoints/env.sh`
2. Add proxy at `og_ego_prim/models/server_inference.py` if needed.
3. Execute the following script: 

```bash
bash entrypoints/eval_close.sh $MODEL_NAME $DATA_PARALLEL
```

### Evaluate Open-Source Models

1. Execute `entrypoints/vllm_serve.sh` to deploy a serve for the evaluated model and check the serve ip.

```bash
bash entrypoints/vllm_serve.sh $LOCAL_MODEL_PATH $GPUS
```

2. Execute the following script:
```bash
bash entrypoints/eval_open.sh $MODEL_NAME_OR_PATH $SERVER_IP $DATA_PARALLEL
```

### Advanced Configuration
1. Revise `entrypoints/task_list.txt` to specify the tasks that need to be evaluated.

2. Revise `prompt_setting` to change safety reminder:
- v0: no safety reminder.
- v1: implicit safety reminder. 
- v2: safety Chain-of-Thought (CoT) reminder. 
- v3: explicit safety reminder.

3. Set the following parameters for optional scene information:
- draw_bbox_2d
- use_initial_setup
- use_self_caption

4. Set the following parameters for partial evaluation:
- not_eval_process_safety
- not_eval_termination_safety
- not_eval_awareness
- not_eval_execution

5. Since the performance of Omnigibson may vary depending on the hardware environment, you can run the following script to check whether the tasks in IS-Bench can be successfully executed in your environment.

```bash
bash entrypoints/validate_gt.sh
```

<h2 id="citation">🙏 Acknowledge</h2>

Leveraged part of data and code framework from[Behavior-1K dataset and Omnigibson simulator](https://github.com/StanfordVL/BEHAVIOR-1K).

<h2 id="citation">📑 Citation</h2>

```bibtex
@misc{lu2025isbench,
      title={IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks}, 
      author={Xiaoya Lu and Zeren Chen and Xuhao Hu and Yijin Zhou and Weichen Zhang and Dongrui Liu and Lu Sheng and Jing Shao},
      year={2025},
      eprint={2506.16402},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.16402}, 
}
```

<hr>
