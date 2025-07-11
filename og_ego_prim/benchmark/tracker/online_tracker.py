import json
import os
from typing import List, Literal

import imageio

from og_ego_prim.benchmark.tracker import EvalTracker
from og_ego_prim.utils.types import StepwisePlan


class OnlineEvalTracker(EvalTracker):

    def __init__(self):
        super().__init__()

        self.plans = []
        self.raw_outputs = []
        self.awareness = None
        self.caption = None

        self.goal_condition = {}
        self.termination = None

        self.error_stack = []
        self.video_cache = []
    
    def track_plan(self, **kwargs):
        self.plans.append(dict(**kwargs))
    
    def track_raw_output(self, **kwargs):
        self.raw_outputs.append(dict(**kwargs))

    def track_error(self, **kwargs):
        self.error_stack.append(dict(**kwargs))

    def track_process_safety_goal_condition(self, **kwargs):
        if 'process_safety_goal_condition' not in self.goal_condition:
            self.goal_condition['process_safety_goal_condition'] = []
        self.goal_condition['process_safety_goal_condition'].append(dict(**kwargs))
    
    def track_termination_safety_goal_condition(self, **kwargs):
        if 'termination_safety_goal_condition' not in self.goal_condition:
            self.goal_condition['termination_safety_goal_condition'] = []
        self.goal_condition['termination_safety_goal_condition'].append(dict(**kwargs))
    
    def track_execution_goal_condition(self, **kwargs):
        self.goal_condition['execution_goal_condition'] = dict(**kwargs)
    
    def track_awareness(self, **kwargs):
        self.awareness = dict(**kwargs)
    
    def track_caption(self, **kwargs):
        self.caption = dict(**kwargs)
        
    def track_termination(self, **kwargs):
        self.termination = dict(**kwargs)

    def track_video_rgb(self, rgb):
        self.video_cache.append(rgb)

    def save_video(self, save_path: str):
        if not self.video_cache:
            return

        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'video.mp4')
        else:
            assert save_path.endswith('.mp4')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        video_writer = imageio.get_writer(save_path, fps=30)
        for rgb in self.video_cache:
            video_writer.append_data(rgb)
        video_writer.close()

    def save_tracking(self, save_path: str):
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        report = {
            'task': self.task,
            'scene': self.scene,
            'model': self.model,
            'awareness': self.awareness,
            'plans': [
                {
                    'step': plan['step'], 
                    'action': plan['plan']['action'], 
                    'caution': plan['plan']['caution']
                }
                for plan in self.plans
            ],
            'termination': self.termination,
            'error_stack': self.error_stack,
        }

        if 'process_safety_goal_condition' in self.goal_condition:
            report['process_safety_goal_condition'] = self.goal_condition['process_safety_goal_condition']
        if 'termination_safety_goal_condition' in self.goal_condition:
            report['termination_safety_goal_condition'] = self.goal_condition['termination_safety_goal_condition']
        if 'execution_goal_condition' in self.goal_condition:
            report['execution_goal_condition'] = self.goal_condition['execution_goal_condition']

        report['raw_outputs'] = self.raw_outputs
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        self.save_video(save_dir)
