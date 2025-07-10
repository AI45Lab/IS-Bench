import json
import os

from .base_benchmark import Benchmark
from .custom_behavior_task import CustomBehaviorTask  # register customized BenhaviorTask
from og_ego_prim.utils.constants import TASKS


def build_benchmark(
    task: str, 
    scene: str = None, 
    ego_view: bool = False,
    draw_bbox_2d: bool = False,
    use_initial_setup: bool = False,
    use_self_caption: bool = False,
    online_object_sampling: bool = None,
    offline_mode: bool = False,
    debug: bool = False, 
    eval_process_safety: bool = True,
    eval_termination_safety: bool = True,
    eval_awareness: bool = True,
    eval_execution: bool = True,
) -> Benchmark:
    
    task_config = os.path.join(TASKS, f'{task}.json')
    assert os.path.exists(task_config), f'invalid task config "{task}"'
    with open(task_config, 'r') as f:
        task_config = json.load(f)

    if online_object_sampling is not None:
        task_config['scene_info']['online_object_sampling'] = online_object_sampling
    print(f'online_object_sampling: {task_config["scene_info"]["online_object_sampling"]}')

    task_kwargs = {
        'task': task,
        'scene': scene,
        'config': task_config,
        'debug': debug,
    }

    task_type = task_config['task_info']['task_type']
    if offline_mode:
        raise NotImplemented

    else:
        from og_ego_prim.benchmark.online_benchmark import ONLINE_BENCHMARKS
        from og_ego_prim.benchmark.tracker.online_tracker import OnlineEvalTracker

        task_kwargs.update({
            'ego_view': ego_view,
            'draw_bbox_2d': draw_bbox_2d,
            'use_initial_setup': use_initial_setup,
            'use_self_caption': use_self_caption,
            'eval_process_safety': eval_process_safety,
            'eval_termination_safety': eval_termination_safety,
            'eval_execution': eval_execution,
            'eval_awareness': eval_awareness,
        })

        assert task_type in ONLINE_BENCHMARKS, \
            f'task_type {task_type} not supported in online mode'
        benchmark = ONLINE_BENCHMARKS[task_type](**task_kwargs)

    return benchmark
