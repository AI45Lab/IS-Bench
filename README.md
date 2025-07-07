# IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks

[Paper](https://www.arxiv.org/abs/2506.16402) | [Code](https://github.com/AI45Lab/IS-Bench) | [Datasets](https://huggingface.co/datasets/Ursulalala/IS_Bench_scenes)

![alt text](assets/motivation.png)

Existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. 

## Snapshot of Results
![alt text](assets/results.png)

## Installation

### System Requirements of Omnigibson

OS: Linux (Ubuntu 20.04+), Windows 10+

RAM: 32GB+ recommended

VRAM: 8GB+

GPU: NVIDIA RTX 2080+

1. Install Omnigibson
``` shell
conda create -n isbench python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 "numpy<2" -c pytorch -c nvidia
conda activate isbench
pip install omnigibson==1.1.1
python -m omnigibson.install    # install omnigibson assets and datasets
```

If you want to install in Docker or other environment, please see this [link](https://behavior.stanford.edu/omnigibson/getting_started/installation.html#__tabbed_1_1).

2. Download Source Code and BDDL of IS-Bench
``` shell
git clone https://github.com/AI45Lab/IS-Bench
cd IS-Bench/bddl
pip install -e .
```

3. Download Scene Dataset
``` shell
cd ../data
wget https://huggingface.co/datasets/Ursulalala/IS_Bench_scenes/resolve/main/scenes.tar.gz
tar -czxf scenes.tar.gz
```

## Evaluate
prompt_setting

task_list

sh scripts/launcher.sh

### Validate Golden Planning


### Evaluate Close-Source Models
1. Configure api_base and api_key at env.sh
2. add proxy at XX
3. sh 


### Evaluate Open-Source Models
vllm api ip

## Acknowledge
Leveraged part of data and code framework from[Behavior-1K dataset](https://github.com/StanfordVL/bddl-100) and [Omnigibson simulator](https://github.com/StanfordVL/BEHAVIOR-1K).

## Citation
```
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
