from og_ego_prim.utils.monkey_patch import add_monkey_patch
add_monkey_patch()

import argparse
import os
import sys

import omnigibson as og
from omnigibson.macros import gm
import shutil
import time
import torch

from og_ego_prim.benchmark import build_benchmark
from og_ego_prim.models import PlanningAgent

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True

parser = argparse.ArgumentParser()
parser.add_argument('--try_id', type=str, default=None)
# parser.add_argument('--try_id', type=bool, default=True)

parser.add_argument('--task', type=str, default=None)
parser.add_argument('--scene', type=str, default=None)
parser.add_argument('--model', type=str, default=None, help="If not local llm, referece to the model_id, if local_llm, referece to the local model path.")
parser.add_argument('--local_llm_serve', action='store_true')
parser.add_argument('--local_serve_ip', type=str, default="")
parser.add_argument('--local_serve_key', type=str, default="sk-123456")
parser.add_argument('--work_dir', type=str, default='./work_dir')

parser.add_argument('--draw_bbox_2d', action='store_true')
parser.add_argument('--use_initial_setup', action='store_true')
parser.add_argument('--use_self_caption', action='store_true')
parser.add_argument('--online_object_sampling', type=bool, default=None)
parser.add_argument('--debug', action='store_true')

parser.add_argument('--not_eval_process_safety', action='store_true')
parser.add_argument('--not_eval_termination_safety', action='store_true')
parser.add_argument('--not_eval_awareness', action='store_true')
parser.add_argument('--not_eval_execution', action='store_true')
parser.add_argument('--prompt_setting', type=str, default='default')


def online_benchmark_once(
    try_id,
    task: str,
    scene: str,
    model: str,
    local_llm_serve: str, 
    local_serve_ip: str,
    local_serve_key: str,
    work_dir: str,
    draw_bbox_2d: bool,
    use_initial_setup: bool,
    use_self_caption: bool,
    online_object_sampling: bool,
    debug: bool,
    eval_process_safety: bool,
    eval_termination_safety: bool,
    eval_awareness: bool,
    eval_execution: bool,
    prompt_setting: str
):
    benchmark = build_benchmark(
        task=task, 
        scene=scene, 
        ego_view=True,
        draw_bbox_2d=draw_bbox_2d,
        use_initial_setup=use_initial_setup,
        use_self_caption=use_self_caption,
        online_object_sampling=online_object_sampling, 
        debug=debug,
        eval_process_safety=eval_process_safety,
        eval_termination_safety=eval_termination_safety,
        eval_awareness=eval_awareness,
        eval_execution=eval_execution,
    )
    if debug and gm.HEADLESS is False:
        og.sim.enable_viewer_camera_teleoperation()

    benchmark_tag = f'{benchmark.task_name}___{benchmark.scene_name}'
    model_tag = args.model.replace('/', '__') if args.model is not None else 'example'
    
    if not try_id:
        output_dir = os.path.join(work_dir, 'benchmark', benchmark_tag, model_tag)
    else: 
        output_dir = os.path.join(work_dir, 'benchmark', benchmark_tag, f"{try_id}_{model_tag}")
    os.makedirs(output_dir, exist_ok=True)

    if online_object_sampling:
        fname = f'{scene}_task_{task}_0_0_template'
        sampled_scene_file = os.path.join(output_dir, f'{fname}.json')
        benchmark.env.task.save_task(path=sampled_scene_file)

    if model or local_llm_serve:
        agent = PlanningAgent(
            task_name=task, 
            scene_name=scene, 
            agent_name=model,
            work_dir=args.work_dir,
            local_llm_serve=local_llm_serve, 
            local_serve_ip=local_serve_ip,  
            local_serve_key=local_serve_key, 
            debug=debug,
            prompt_setting=prompt_setting,
            use_initial_setup=use_initial_setup,
            use_self_caption=use_self_caption,
        )
        agent.set_tracker(benchmark.tracker)
        planner = agent.step(use_obs=True, max_step=(len(benchmark._example_planning) + 10))
    else:
        planner = benchmark.get_example_planning()

    benchmark.get_surrounding_viewer_obs(save_img=os.path.join(output_dir, '0_init'))
    if use_self_caption:
        caption = agent.generate_caption(use_obs=True)
        benchmark.tracker.track_caption(
            content=caption
        )
    if eval_awareness and (model or local_llm_serve):
        awareness = agent.generate_awareness(use_obs=True)
        benchmark.evaluate_awareness(awareness)
    elif prompt_setting == 'v2':
        awareness = agent.generate_awareness(use_obs=True)
        benchmark.tracker.track_awareness(
            content=awareness,
            eval_results=None
        )
    if not (eval_process_safety or eval_termination_safety or eval_execution):
        benchmark.tracker.save_tracking(os.path.join(output_dir, 'report_awareness.json'))
        time.sleep(3)
        og.clear()
        return

    for i, plan in enumerate(planner):
        if benchmark.execute_plan(plan) is False:
            break
        step_tag = f'{i+1}_' + plan['action'].replace('(', '__').replace(')', '__')
        benchmark.get_surrounding_viewer_obs(save_img=os.path.join(output_dir, step_tag))

    benchmark.termination_evaluation()
    benchmark.tracker.save_tracking(os.path.join(output_dir, 'report.json'))
    
    if online_object_sampling:
        if benchmark.tracker.termination['reason'] == 'done' and benchmark.tracker.goal_condition['execution_goal_condition']['eval']: 
            normal_scene_file = os.path.join(work_dir, "..", "data", "scenes", scene, "json", f'{fname}.json')
            shutil.copyfile(sampled_scene_file, normal_scene_file)
        else:
            os.remove(sampled_scene_file)

    time.sleep(3)
    og.clear()


if __name__ == "__main__":
    args = parser.parse_args()
    print(f'args: {args}')
    sys.stdout.flush()

    online_benchmark_once(
        try_id=args.try_id,
        task=args.task,
        scene=args.scene,
        model=args.model,
        local_llm_serve=args.local_llm_serve,
        local_serve_ip=args.local_serve_ip,
        local_serve_key=args.local_serve_key,
        prompt_setting=args.prompt_setting,
        work_dir=args.work_dir,
        draw_bbox_2d=args.draw_bbox_2d,
        use_initial_setup=args.use_initial_setup,
        use_self_caption=args.use_self_caption,
        online_object_sampling=args.online_object_sampling,
        debug=args.debug,
        eval_process_safety=(not args.not_eval_process_safety),
        eval_termination_safety=(not args.not_eval_termination_safety),
        eval_awareness=(not args.not_eval_awareness),
        eval_execution=(not args.not_eval_execution),
    )
