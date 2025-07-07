import argparse
import json
import os
from tqdm import tqdm

from og_ego_prim.utils.constants import WORK_DIR, TASKS
from og_ego_prim.utils.metric import Metric, read_benchmark_report

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default=None)
parser.add_argument('--model', type=str)


def main(args):
    benchmark_dir = args.benchmark
    if benchmark_dir is None:
        benchmark_dir = os.path.join(WORK_DIR, 'benchmark')
    if not os.path.exists(benchmark_dir):
        return
    work_dir = os.path.dirname(benchmark_dir)
    
    metric = Metric()
    tasks = os.listdir(TASKS)
    tasks = [os.path.join(TASKS, task) for task in tasks if task.endswith('.json')]

    for task in tqdm(tasks):
        with open(task, 'r') as f:
            try:
                task_config = json.load(f)
            except:
                import ipdb; ipdb.set_trace()
                print(task)
                raise
        task_name = task_config['task_info']['task_name']
        scene_name = task_config['scene_info']['default_scene_model']

        read_benchmark_report(task_name, scene_name, args.model, work_dir, metric)
    
    with open(os.path.join(benchmark_dir, 'report_all.json'), 'w') as f:
        json.dump(metric.summary(), f, indent=4)


if __name__ == '__main__':
    main(parser.parse_args())
