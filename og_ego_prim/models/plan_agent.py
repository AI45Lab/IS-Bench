import json
import os   
import re
import sys
from typing import Generator, List, Tuple, Optional

from og_ego_prim.models.hf_inference import HFClient
from og_ego_prim.models.server_inference import ServerClient
from og_ego_prim.models.base_client import BaseClient
from og_ego_prim.primitives import VALID_PRIMITIVES
from og_ego_prim.benchmark.tracker import EvalTracker
from og_ego_prim.utils.constants import WORK_DIR
from og_ego_prim.utils.prompts import *
from og_ego_prim.utils.types import StepwisePlan

from og_ego_prim.utils.constants import TASKS

class BadAgentPlanError(Exception):
    pass


def parse_output(output: str) -> Optional[StepwisePlan]:
    pattern = r'```json(.*?)```'
    result = re.findall(pattern, output, re.DOTALL)
    
    if len(result) >= 1:
        result = result[0].strip()
        try:
            result = json.loads(result)
        except:
            result = None
        return result
    return None


def get_obs_from_dir(obs_dir: str) -> List[str]:
    obs_path_list = []
    for img in os.listdir(obs_dir):
        img_path = os.path.join(obs_dir, img)
        if img.endswith(".png"):
            obs_path_list.append(img_path)
    return obs_path_list


class PlanningAgent: 
    
    def __init__(
        self, 
        task_name: str, 
        scene_name: str, 
        agent_name: str, 
        work_dir: str,
        local_llm_serve: str,
        local_serve_ip: str,
        local_serve_key: str,
        prompt_setting: str,
        use_initial_setup: bool = False,
        use_self_caption: bool = False,
        retry: int = 3,
        verbose: bool = True,
        debug: bool = False,
    ) -> None:
        if work_dir is None:
            work_dir = WORK_DIR
        self.working_dir = os.path.join(work_dir, "benchmark")
        assert os.path.exists(self.working_dir)

        self.task_name = task_name
        self.scene_name = scene_name
        self.agent_name = agent_name
        self.current_step = 0

        self.retry = retry
        self.verbose = verbose
        self.debug = debug
        
        self.local_llm_serve = local_llm_serve
        self.local_serve_ip = local_serve_ip
        self.local_serve_key = local_serve_key
        self.prompt_setting = prompt_setting
        self.use_initial_setup = use_initial_setup
        self.use_self_caption = use_self_caption

        # initialize data 
        self.task_instruction, self.objects_str, self.initial_setup_str, self.object_abilities_str, self.wash_rules_str, self.goal_bddl_str, self.safety_tips_str = self.load_info_data()
        if self.verbose:
            print(f'[agent] instruction: {self.task_instruction}')
            print(f'[agent] objects:\n{self.objects_str}')
            print(f'[agent] initial setup:\n{self.initial_setup_str}')
            print(f'[agent] object abilities:\n{self.object_abilities_str}')
            print(f'[agent] wash rules:\n{self.wash_rules_str}')
            print(f'[agent] goal bddl:\n{self.goal_bddl_str}')
            sys.stdout.flush()
        
        self.client = self._get_agent(agent_name)
    
    def set_tracker(self, tracker: EvalTracker):
        self.tracker = tracker
        model_name = self.agent_name.split("/")[-1]
        self.tracker.model = model_name

    def _get_agent(self, agent_name: str) -> BaseClient:
        if self.local_llm_serve: 
            return ServerClient(
                model_type="local", 
                model_name=agent_name,
                api_key=self.local_serve_key, 
                api_base=self.local_serve_ip
            ) 
        else: 
            return ServerClient(
                model_type="close_source",
                model_name=agent_name, 
                api_key=os.environ['OPENAI_API_KEY'], 
                api_base=os.environ['OPENAI_API_BASE']
            ) 

    def _get_last_execution_info(self, use_obs=True):
        last_step, last_plan = 0, 'init'
        for plan in reversed(self.tracker.plans):
            if not plan['plan']['action'].startswith('navigate'):
                last_step = plan['step']
                last_plan = plan['plan']['action']
                break
        
        if not use_obs:
            observations = None
        else:
            benchmark_tag = f'{self.task_name}___{self.scene_name}'
            model_tag = self.agent_name.replace('/', '__')
            step_tag = f'{last_step}_' + last_plan.replace('(', '__').replace(')', '__')
            obs_dir = os.path.join(self.working_dir, benchmark_tag, model_tag, step_tag)
            observations = get_obs_from_dir(obs_dir)

            print(f'read obs from {obs_dir}')
            sys.stdout.flush()
        
        return last_plan, observations

    def _prepare_prompt(self) -> str:
        history_plans = "None"
        if self.current_step > 0:
            history_plans = '\n'.join(
                [history['history_text'] for history in self.tracker.plans]
            )
            
        if not self.use_initial_setup and not self.use_self_caption:
            if self.prompt_setting == 'v0': # v0: no safety reminder
                prompt = V0StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans
                )
            elif self.prompt_setting == 'v1': # v0 + implicit safety reminder
                prompt = V1StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans
                )
            elif self.prompt_setting == 'v2': # v0 + cot safety reminder
                assert self.tracker.awareness is not None and 'content' in self.tracker.awareness
                awareness = self.tracker.awareness['content']
                prompt = V2StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans,
                    awareness=awareness
                )
            elif self.prompt_setting == 'v3': # v0 + explicit safety reminder
                prompt = V3StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    safety_tips=self.safety_tips_str,
                    history_actions=history_plans
                )
            else:  
                raise Exception('Wrong prompt setting.') 
        else:
            if self.use_initial_setup:
                scene_description = self.initial_setup_str
            else:
                assert self.tracker.caption is not None and 'content' in self.tracker.caption
                scene_description = self.tracker.caption['content']

            if self.prompt_setting == 'v0': 
                prompt = T0StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans,
                    scene_description=scene_description, 
                )
            elif self.prompt_setting == 'v1': 
                prompt = T1StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans,
                    scene_description=scene_description, 
                )
            elif self.prompt_setting == 'v2':
                assert self.tracker.awareness is not None and 'content' in self.tracker.awareness
                awareness = self.tracker.awareness['content'] 
                prompt = T2StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans,
                    scene_description=scene_description, 
                    awareness=awareness
                )
            elif self.prompt_setting == 'v3': 
                prompt = T3StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans,
                    safety_tips=self.safety_tips_str,
                    scene_description=scene_description 
                )
            else:  
                raise Exception('Wrong prompt setting.') 
        
        return prompt
    

    def _verify_plan(self, plan: Optional[StepwisePlan]) -> Optional[Tuple[str, str, str]]:
        if plan is None:
            return None
        if 'action' not in plan:
            return None
        
        action = plan['action'].strip()
        if action.upper().startswith('DONE'):
            caution = plan.get('caution', None)
            return 'done', '', caution

        pattern = r'(?:\d+\.\s+)?([a-zA-Z_]+)\(([^)]+)\)'
        matches = re.findall(pattern, action)
        if len(matches) >= 1:
            operator, params = matches[-1]
        else:
            return None
        
        operator = operator.strip()
        if operator.upper() not in VALID_PRIMITIVES.keys():
            return None

        params = params.strip().lower()
        objects = [obj.strip() for obj in params.split(',')]
        if len(objects) != VALID_PRIMITIVES[operator.upper()]:
            return None
        for obj in objects:
            if obj not in self.objects_str:
                return None

        if 'caution' not in plan:
            caution = None
        else:
            caution = plan['caution']
        return operator.lower(), params, caution
    
    def generate_caption(self, use_obs=True) -> str:
        _, obs = self._get_last_execution_info(use_obs)
        prompt_cp = GenerateCaptionPrompt.format(
                objects_str=self.objects_str, 
                task_instruction=self.task_instruction, 
                object_abilities_str=self.object_abilities_str, 
                task_goals=self.goal_bddl_str,
                wash_rules_str=self.wash_rules_str,
            )
        output_caption = self.client.model(prompt_cp, image_file=obs)
        return output_caption
        
    def generate_awareness(self, use_obs=True) -> str:
        _, obs = self._get_last_execution_info(use_obs)
        if self.use_initial_setup or self.use_self_caption: 
            if self.use_initial_setup:
                scene_description = self.initial_setup_str
            else:
                assert self.tracker.caption is not None and 'content' in self.tracker.caption
                scene_description = self.tracker.caption['content']
            prompt_sa = T2GenerateAwarenessPrompt.format(
                objects_str=self.objects_str, 
                task_instruction=self.task_instruction, 
                object_abilities_str=self.object_abilities_str, 
                task_goals=self.goal_bddl_str,
                wash_rules_str=self.wash_rules_str,
                scene_description=scene_description, 
            )
        else:
            prompt_sa = GenerateAwarenessPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                )
        output = self.client.model(prompt_sa, image_file=obs)
        return output
        
        

    def step(self, use_obs=True, max_step=None) -> Generator[str, None, None]:
        retry = 0
        while True:
            # get obs after last execution
            last_plan, obs = self._get_last_execution_info(use_obs)
            prompt = self._prepare_prompt()

            if self.debug:
                print(f'[agent] last_step: {last_plan}, Continue (y/Y): ')
                sys.stdout.flush()

                while cmd := input().upper() != 'Y':
                    print(f'[agent] last_step: {last_plan}, Continue (y/Y): ')
                    sys.stdout.flush()
            
            output = self.client.model(prompt, image_file=obs)
            next_plan = parse_output(output)
            if self.verbose:
                print(f"[agent] raw output:\n{output}")
                print(f"[agent] next plan:\n{next_plan}")
                sys.stdout.flush()

            # verification the next step of generated plan is correct
            results = self._verify_plan(next_plan)
            if results is None:
                retry += 1
                if retry < self.retry:
                    print(f"[agent] retry...")
                    sys.stdout.flush()
                    continue
                else:
                    self.tracker.track_termination(
                        reason='plan_error',
                        type='BadAgentPlanError',
                        msg=f'plan ``{next_plan if next_plan else "None"}`` not applicable'
                    )
                    return
            else:
                retry = 0
            
                operator, params, caution = results
                self.current_step += 1
                next_plan: StepwisePlan = dict(
                    action=f'{operator}({params})',
                    caution=caution
                )
                self.tracker.track_plan(
                    step=self.current_step,
                    plan=next_plan,
                    history_text=f'{self.current_step}. {operator.upper()}({params.lower()})'
                )
                self.tracker.track_raw_output(
                    step=self.current_step,
                    content=output,
                )

                yield next_plan
                if operator == 'done':
                    return
                if max_step is not None and self.current_step > max_step:
                    self.tracker.track_termination(
                        reason='exceeding_max_steps',
                        type='BadAgentPlanError',
                        msg=f'exceeding max steps {max_step}'
                    )
                    return
        
    def load_info_data(self):
        with open(os.path.join(TASKS, f"{self.task_name}.json"), 'r', encoding='utf-8') as f:
            task_json_data = json.load(f)
        task_instruction = task_json_data['planning_context']['task_instruction']
        objects_list = task_json_data['planning_context']['object_list']
        objects_str = '\n'.join(f"{i+1}. {item.strip()}" for i, item in enumerate(objects_list))
        intial_setup_list = task_json_data['planning_context']['initial_setup']
        initial_setup_str = '\n'.join(f"{item.strip()}" for i, item in enumerate(intial_setup_list))
        
        object_abilities = task_json_data['planning_context']['object_abilities']
        if object_abilities is None:
            object_abilities_str = ""
        else:
            object_abilities_str = '\n'.join([f"{key}: " + str(value) for key, value in object_abilities.items()])
            
        wash_rules = task_json_data['planning_context']['wash_rules']
        if wash_rules is None:
            wash_rules_str = ""
        else: 
            wash_rules_str = json.dumps(wash_rules, indent=4, ensure_ascii=False)

        safety_tips = []
        for tip in task_json_data['evaluation_cautions']:
            safety_tips.append(tip['safety_tip'])
        for tip in task_json_data['evaluation_goal_conditions']['process_safety_goal_condition']:
            safety_tips.append(tip['safety_tip'])
        for tip in task_json_data['evaluation_goal_conditions']['termination_safety_goal_condition']:
            safety_tips.append(tip['safety_tip'])
        safety_tips_str = json.dumps(safety_tips, indent=4, ensure_ascii=False)
        
        goal_condition_bddl_str = task_json_data['evaluation_goal_conditions']['execution_goal_condition']
        
        return task_instruction, objects_str, initial_setup_str, object_abilities_str, wash_rules_str, goal_condition_bddl_str, safety_tips_str
