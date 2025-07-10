import json
import os
import random
from typing import Dict, Generator, List, Literal, Optional
import yaml

import bddl
from numpy.typing import ArrayLike as NumpyArrayLike
import omnigibson as og
from omnigibson import object_states
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.bddl_utils import BEHAVIOR_ACTIVITIES
from PIL import Image
import torch

from .data_utils import (
    CUSTOMIZED_BEHAVIOR_ACTIVITIES, 
    get_customized_definition_filename,
    colorize_bboxes
)
from og_ego_prim.benchmark.base_benchmark import Benchmark
from og_ego_prim.benchmark.evaluator.evaluator import Evaluator
from og_ego_prim.benchmark.tracker.online_tracker import OnlineEvalTracker
from og_ego_prim.primitives import Executor
from og_ego_prim.primitives.object_states_utils import (
    is_target_object_predicate_with_obj, 
    find_task_related_object,
    get_visible_task_related_objects,
)
from og_ego_prim.utils.constants import CAMERAS, SCENES
from og_ego_prim.utils.types import PoseCoord, StepwisePlan


__all__ = ['ONLINE_BENCHMARKS']


class OnlineBenchmark(Benchmark):
    
    env: og.Environment
    ego_view: bool
    draw_bbox_2d: bool
    surrounding_poses: List[PoseCoord]

    executor: Executor
    evaluator: Evaluator
    tracker: OnlineEvalTracker

    def __init__(
        self,
        task: str, 
        scene: str, 
        config: Dict, 
        debug: bool,
        ego_view: bool, 
        draw_bbox_2d: bool,
        use_initial_setup: bool,
        use_self_caption: bool,
        eval_process_safety: bool,
        eval_termination_safety: bool,
        eval_awareness: bool, 
        eval_execution: bool,
    ):
        super().__init__(task, scene, config, debug, False)

        self.env = og.Environment(configs=self.env_config)
        self.ego_view = ego_view
        self.draw_bbox_2d = draw_bbox_2d
        self.use_initial_setup = use_initial_setup
        self.use_self_caption = use_self_caption
            
        camera_config = os.path.join(CAMERAS, 'camera.json')
        with open(camera_config, 'r') as f:
            camera_config = json.load(f)
        room = config['scene_info']['room']

        self.surrounding_poses = None
        if camera_config.get(f'{room}__{scene}', None):
            camera_config = camera_config[f'{room}__{scene}']
            
            self.surrounding_poses = []
            for pose_dict in camera_config:
                self.surrounding_poses.append(
                    (torch.tensor(pose_dict['pos']), torch.tensor(pose_dict['quat']))
                )

        self.tracker = OnlineEvalTracker()
        self.tracker.task = self.task_name
        self.tracker.scene = self.scene_name
        
        self.executor = Executor(self.env, primitive_type='ego', debug=debug)
        self.evaluator = Evaluator(
            self.env, config, self.tracker,
            eval_process_safety, 
            eval_termination_safety, 
            eval_awareness, 
            eval_execution
        )
        
        self.task_instruction = self._get_task_information(config)[0]
        self.initial_setup = self._get_task_information(config)[1]

        self.set_viewer()
        self._add_extra_init_states()

    def get_example_planning(self) -> Generator[str, None, None]:
        for i, plan in enumerate(self._example_planning):
            self.tracker.track_plan(step=i, plan=plan)
            yield plan
            if plan['action'].lower().startswith('done'):
                return

    def set_viewer(self):
        if self.ego_view:
            for i in range(len(self.env.robots)):
                self.env.robots[i].visible = False
            self.executor._simulator_loop(5)
        
        if self.draw_bbox_2d:
            og.sim.viewer_camera.add_modality('bbox_2d_tight')

    def _add_extra_init_states(self):
        # set objects in refrigerator to frozen
        refrigerator = find_task_related_object(self.env, 'refrigerator')
        if refrigerator is None:
            return
        
        for _, obj_ref in self.env.task.object_scope.items():
            obj = obj_ref.wrapped_obj
            if obj is None:
                continue
            if not hasattr(obj, 'states'):
                continue
            if object_states.Frozen not in obj.states:
                continue
            if not is_target_object_predicate_with_obj(obj, refrigerator, object_states.Inside):
                continue

            obj.states[object_states.Frozen].set_value(True)
        
        self.executor._simulator_loop(5)
    
    def _get_task_information(self, config: Dict):
        cond_configs = config["planning_context"]
        if not cond_configs:
            return None

        task_instruction = cond_configs['task_instruction']
        initial_setup = cond_configs['initial_setup']

        return task_instruction, initial_setup

    def execute_plan(self, plan: StepwisePlan | str) -> bool:
        if isinstance(plan, str):
            plan: StepwisePlan = dict(action=plan, caution=None)
        
        if plan['action'].upper().startswith('NAVIGATE'):
            return True

        self.evaluator.evaluate_process_safety_goal_condition(plan, 'before')
        
        if self.debug:
            self.executor.execute_plan(plan['action'])
        else:
            try:
                self.executor.execute_plan(plan['action'])
            except Exception as e:
                self.tracker.track_error(
                    action=plan['action'],
                    err_type=e.__class__.__name__,
                    msg=str(e)
                )

        self.evaluator.evaluate_process_safety_goal_condition(plan, 'after')
        return True

    def evaluate_awareness(self, awareness: str):
        self.evaluator.evaluate_awareness(
            self.task_instruction,
            self.initial_setup,
            awareness
        )

    def termination_evaluation(self):
        self.evaluator.evaluate_execution_goal_condition()
        self.evaluator.evaluate_non_executed_process_safety_goal_condition()
        self.evaluator.evaluate_termination_safety_goal_condition()
        if self.tracker.termination is None:
            self.tracker.track_termination(
                reason='done'
            )

    def reset_viewer_camera(self, pose: PoseCoord):
        if not isinstance(pose[0], torch.Tensor):
            pos, quat = pose
            pos = torch.Tensor(pos)
            quat = torch.Tensor(quat)
            pose = (pos, quat)

        og.sim.viewer_camera.set_position_orientation(*pose)
        self.executor._simulator_loop(5)

    def _preprocess_obs(self) -> NumpyArrayLike:
        obs, info = og.sim.viewer_camera.get_obs()
        rgb = obs['rgb'].cpu().numpy()
        if not self.draw_bbox_2d:
            return rgb

        bbox_2d_data = obs['bbox_2d_tight']
        bbox_2d_info = info['bbox_2d_tight']
        visible_task_related_objects = get_visible_task_related_objects(self.env)

        visible_task_related_bbox_2d_id = []
        for bbox_2d_id, bbox_name in bbox_2d_info.items():
            for obj in visible_task_related_objects:
                if bbox_name in obj.name:
                    visible_task_related_bbox_2d_id.append(bbox_2d_id)
                    break
        visible_task_related_bbox_2d_data = [
            data for data in bbox_2d_data if data[0] in visible_task_related_bbox_2d_id
        ]
        rgb_with_bbox_2d = colorize_bboxes(visible_task_related_bbox_2d_data, rgb, bbox_2d_info, num_channels=4)
        return rgb_with_bbox_2d

    def get_viewer_obs(
        self, 
        pose: Optional[PoseCoord] = None, 
        save_img: Optional[str] = None
    ) -> NumpyArrayLike:
        if pose is not None:
            self.reset_viewer_camera(pose) 
        
        obs = self._preprocess_obs()
        if save_img is not None:
            if os.path.isdir(save_img):
                save_img = os.path.join(save_img, 'obs.png')
            else:
                os.makedirs(os.path.dirname(save_img), exist_ok=True)

            img = Image.fromarray(obs)
            img.save(save_img)

        return obs

    def get_surrounding_viewer_obs(
        self, save_img: Optional[str] = None
    ) -> Optional[List[NumpyArrayLike]]:
        if self.surrounding_poses is None:
            return None

        if save_img is not None:
            if not os.path.exists(save_img):
                os.makedirs(save_img)
            elif not os.path.isdir(save_img):
                raise ValueError(f'surrounding_obs must be saved in a directory')
            
        surrounding_obs = []
        for i, pose in enumerate(self.surrounding_poses):
            save_img_i = None if save_img is None else os.path.join(save_img, f'obs_{i}.png')
            obs_i = self.get_viewer_obs(pose, save_img_i)
            surrounding_obs.append(obs_i)
        return surrounding_obs


class OnlineBehaviorBenchmark(OnlineBenchmark):
    
    def init_env_config(self, task: str, scene: str, config: Dict):
        env_config = os.path.join(og.example_config_path, config['_base_config'])
        with open(env_config, 'r') as f:
            env_config = yaml.load(f, Loader=yaml.FullLoader)

        task_info = config['task_info']
        scene_info = config['scene_info']                
        
        # task customization
        task_name = task_info['task_name']
        assert task_name in BEHAVIOR_ACTIVITIES or task_name in CUSTOMIZED_BEHAVIOR_ACTIVITIES
        if task_name not in BEHAVIOR_ACTIVITIES:
            og.tasks.behavior_task.BEHAVIOR_ACTIVITIES.append(task_name)
            bddl.parsing.get_definition_filename = get_customized_definition_filename

        task_type = task_info['task_type'] if not scene_info['online_object_sampling'] \
            else 'CustomBehaviorTask'
        print(f'Using task type: {task_type}')

        env_config['task'] = {
            'type': task_type,
            'activity_name': task_name,
            'activity_definition_id':  task_info['activity_definition_id'],
            'activity_instance_id':  task_info['activity_instance_id'],
            'predefined_problem': None,
            'online_object_sampling': scene_info['online_object_sampling'],
        }

        if scene is None:
            if 'default_scene_model' in scene_info and scene_info['default_scene_model']:
                scene = scene_info['default_scene_model']
            else:
                scene = random.choice(scene_info['scene_models'])
        assert scene in scene_info['scene_models'], f'task "{task}" is not supported in scene "{scene}"'

        env_config['scene'].update({
            'scene_model': scene,
            # 'load_task_relevant_only': True if self.debug else False,
            'load_task_relevant_only': True,
            'not_load_object_categories': ['ceilings', 'roof']
        })

        # scene customization
        activity_definition_id = task_info['activity_definition_id']
        activity_instance_id = task_info['activity_instance_id']
        scene_file = BehaviorTask.get_cached_activity_scene_filename(
            scene_model=scene,
            activity_name=task_name,
            activity_definition_id=activity_definition_id,
            activity_instance_id=activity_instance_id,
        )
        # use customized scene if scene_file exists
        scene_file = os.path.join(SCENES, scene, 'json', f'{scene_file}.json')
        if not scene_info['online_object_sampling'] and os.path.exists(scene_file):
            env_config['scene']['scene_file'] = scene_file

        return env_config


ONLINE_BENCHMARKS = {
    'BehaviorTask': OnlineBehaviorBenchmark,
}
