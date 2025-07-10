import ast
import json
import re
import os
from typing import Dict, List, Literal, Optional

from omnigibson.envs import Environment
from omnigibson.termination_conditions.predicate_goal import PredicateGoal
import openai

from og_ego_prim.benchmark.evaluator.bddl_goal_condition import compile_bddl_goal_condition
from og_ego_prim.benchmark.tracker import EvalTracker
from og_ego_prim.primitives.ego_primitives import VALID_PRIMITIVES
from og_ego_prim.utils.prompts import *
from og_ego_prim.utils.types import GoalCondition, StepwisePlan


class Evaluator:

    process_safety_goal_condition: Optional[Dict[str, GoalCondition]]
    termination_safety_goal_condition: Optional[List[GoalCondition]]
    execution_goal_condition: Optional[PredicateGoal]

    def __init__(
        self,
        env: Environment,
        config: Dict,
        tracker: EvalTracker,
        eval_process_safety: bool,
        eval_termination_safety: bool,
        eval_awareness: bool, 
        eval_execution: bool,
    ):
        self.env = env
        self.tracker = tracker

        self.process_safety_goal_condition = None if not eval_process_safety \
            else self._get_process_safety_goal_condition(config)
        self.termination_safety_goal_condition = None if not eval_termination_safety \
            else self._get_termination_safety_goal_condition(config)
        self.execution_goal_condition = None if not eval_execution \
            else self._get_execution_goal_condition(config)
        self.awareness_list = None if not eval_awareness \
            else self._get_awareness_list(config)
        
        self.judger_client = openai.OpenAI(
            api_key=os.environ['OPENAI_API_KEY'], 
            base_url=os.environ['OPENAI_API_BASE']
        )

    def _get_process_safety_goal_condition(self, config: Dict) -> Optional[Dict[str, GoalCondition]]:
        cond_configs = config['evaluation_goal_conditions']['process_safety_goal_condition']
        if not cond_configs:
            return None
        
        process_safety_goal_condition = {}
        for cond_config in cond_configs:
            condition_type = cond_config['type'].strip().lower()
            assert condition_type in ['after', 'before']
            action = cond_config['action'].strip().lower()
            primitive = action.split('(')[0].strip()
            assert primitive.upper() in VALID_PRIMITIVES.keys()
            param = action.split('(')[1].strip().split(')')[0].strip().replace(' ', '')
            action = f'{primitive}({param})'
            
            assert 'safety_bddl' in cond_config
            evaluator = compile_bddl_goal_condition(self.env.task, cond_config['safety_bddl']) 
            
            goal_condition: GoalCondition = dict(
                risk_type=cond_config['risk_type'],
                safety_principle=cond_config['safety_principle'],
                safety_tip=cond_config['safety_tip'],
                bddl=cond_config['safety_bddl'],
                bddl_evaluator=evaluator
            )
            process_safety_goal_condition[(condition_type, action)] = goal_condition
        
        return process_safety_goal_condition

    def _get_termination_safety_goal_condition(self, config: Dict) -> Optional[List[GoalCondition]]:
        cond_configs = config['evaluation_goal_conditions']['termination_safety_goal_condition']
        if not cond_configs:
            return None

        termination_safety_goal_condition = []
        for cond_config in cond_configs:
            assert 'safety_bddl' in cond_config
            evaluator = compile_bddl_goal_condition(self.env.task, cond_config['safety_bddl']) 

            goal_condition: GoalCondition = dict(
                risk_type=cond_config['risk_type'],
                safety_principle=cond_config['safety_principle'],
                safety_tip=cond_config['safety_tip'],
                action=cond_config['action'],
                bddl=cond_config['safety_bddl'], 
                bddl_evaluator=evaluator
            )
            termination_safety_goal_condition.append(goal_condition)

        return termination_safety_goal_condition

    def _get_goal_text_from_tokens(self, goal_text: str, goal_conds: List) -> str:
        goal_text += '('
        for i, token in enumerate(goal_conds):
            if isinstance(token, List):
                goal_text = self._get_goal_text_from_tokens(goal_text, token)
            else:
                goal_text += token
                if i != len(goal_conds) - 1:
                    goal_text += ' '
        goal_text += ')'
        return goal_text

    def _get_execution_goal_condition(self, config: Dict) -> GoalCondition:
        goal_condition = config['evaluation_goal_conditions']['execution_goal_condition']
        if not goal_condition:
            parsed_goal_conditions = self.env.task.activity_conditions.parsed_goal_conditions
            if len(parsed_goal_conditions) == 1:
                parsed_goal_conditions = parsed_goal_conditions[0]
            goal_condition_bddl = self._get_goal_text_from_tokens('', parsed_goal_conditions)

            goal_condition: GoalCondition = dict(
                bddl=goal_condition_bddl,
                bddl_evaluator=PredicateGoal(goal_fcn=lambda: self.env.task.activity_goal_conditions)
            )

        else:
            evaluator = compile_bddl_goal_condition(
                self.env.task, goal_condition
            )
            goal_condition: GoalCondition = dict(
                bddl=goal_condition,
                bddl_evaluator=evaluator
            )

        return goal_condition

    def _get_awareness_list(self, config: Dict) -> Optional[List[GoalCondition]]:
        awareness_list = config['evaluation_cautions']
        for goal_cond_key in ['process_safety_goal_condition', 'termination_safety_goal_condition']:
            awareness_list.extend([
                dict(
                    risk_type=goal_cond['risk_type'], 
                    safety_principle=goal_cond['safety_principle'],
                    safety_tip=goal_cond['safety_tip']
                )
                for goal_cond in config['evaluation_goal_conditions'][goal_cond_key]
            ])
        return awareness_list

    def llm_response_to_json(self, response: str) -> Dict:
        response = response.replace("\n", "")
        
        # Attempt to parse directly as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting content wrapped with ```json
        json_pattern = r"```json\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, response)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Try extracting content wrapped with any ``` block
        code_block_pattern = r"```\s*([\s\S]*?)\s*```"
        match = re.search(code_block_pattern, response)
        if match:
            potential_json = match.group(1)
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError:
                pass

        # Try to extract content between the first '{' and the last '}'
        brace_pattern = r"\{[\s\S]*\}"
        match = re.search(brace_pattern, response)
        if match:
            json_str = match.group(0)
            try:
                # Attempt parsing with ast.literal_eval for JSON-like structures
                return ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                pass

        # Try parsing key-value pairs for simpler JSON structures
        json_data = {}
        for line in response.split(","):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().strip('"')
                value = value.strip().strip('"')
                json_data[key] = value
        if json_data:
            return json_data
        
        # If all attempts fail, return None or raise an error
        raise ValueError(f"Could not parse response as JSON: {response}")   
    
    def llm_as_a_safety_judger(self, task_instruction, initial_setup, pred_cautions, gt_cautions) -> bool:
        os.environ['HTTP_PROXY']='http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@10.1.20.51:23128/'
        os.environ['HTTPS_PROXY']='http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@10.1.20.51:23128/'
        os.environ['http_proxy']='http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@10.1.20.51:23128/'
        os.environ['https_proxy']='http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@10.1.20.51:23128/'
        
        prompt = DefaultSafetyJudgePrompt.format(
            task_instruction=task_instruction,
            initial_setup=initial_setup,                                                 
            safety_tips=gt_cautions,
            sample_planning=pred_cautions
        )
        
        completion = self.judger_client.chat.completions.create(
            model='gpt-4o',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0
        )
        print(completion.choices[0].message.content) 
        res = self.llm_response_to_json(str(completion.choices[0].message.content))
        if "openai.com" in os.environ.get("OPENAI_API_BASE"):
            os.environ["http_proxy"] = "http://10.1.20.57:23128"
            os.environ["https_proxy"] = "http://10.1.20.57:23128"
            os.environ["HTTP_PROXY"] = "http://10.1.20.57:23128"
            os.environ["HTTPS_PROXY"] = "http://10.1.20.57:23128"

        return res
    
    def llm_as_a_safety_awareness_judger(self, task_instruction, initial_setup, pred_cautions, gt_cautions) -> bool:
        os.environ['HTTP_PROXY']='http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@10.1.20.51:23128/'
        os.environ['HTTPS_PROXY']='http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@10.1.20.51:23128/'
        os.environ['http_proxy']='http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@10.1.20.51:23128/'
        os.environ['https_proxy']='http://luxiaoya:kwMUZpsjfkRdN6rANEJp45sBoXK9gP1uLzQbwgerNbixbWFj3iOQMjTynOq8@10.1.20.51:23128/'
        
        prompt = SafetyAwarenessJudgePrompt.format(
            task_instruction=task_instruction,
            initial_setup=initial_setup,                                                 
            gt_cautions=gt_cautions,
            pred_cautions=pred_cautions
        )
        
        completion = self.judger_client.chat.completions.create(
            model='gpt-4o',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.6
        )
        print(completion.choices[0].message.content) 
        res = self.llm_response_to_json(str(completion.choices[0].message.content))            

        return res

    def evaluate_awareness(self, task_instruction: str, initial_setup: str, awareness: str):
        retry = 0
        results = None
        while retry < 3:
            results = self.llm_as_a_safety_awareness_judger(
                task_instruction, 
                initial_setup, 
                awareness, 
                self.awareness_list
            )
            if type(results) == list:
                break
            elif type(results) == dict:
                results = [results]
                break
            retry += 1
        # eval_awareness = [item["eval"] for item in results] # List[bool]
        self.tracker.track_awareness(
            content=awareness,
            eval_results=results
        )

    def evaluate_process_safety_goal_condition(
        self, plan: StepwisePlan, condition_type: Literal['before', 'after'], verbose=True,
    ):
        if self.process_safety_goal_condition is None:
            return
        
        action = plan['action'].strip().lower().replace(' ', '')
        condition_key = (condition_type.lower(), action)
        if condition_key not in self.process_safety_goal_condition:
            return

        goal_condition = self.process_safety_goal_condition[condition_key]
        assert 'bddl_evaluator' in goal_condition
        _, success = goal_condition['bddl_evaluator'].step(self.env.task, self.env, None)

        self.tracker.track_process_safety_goal_condition(
            action=plan['action'],
            type=condition_type,
            eval_mode='bddl',
            risk_type = goal_condition['risk_type'],
            safety_principle=goal_condition['safety_principle'],
            condition=goal_condition['bddl'],
            eval=success,
        )

        # only evaluate once at first
        del self.process_safety_goal_condition[condition_key]

        if not success and verbose:
            condtion = goal_condition['bddl']
            print(
                f'[goal-condition] Proccess Safety Goal Condition not met.\n'
                f'[goal-condition] -> ❎ <{condition_type}> <{action}> {condtion}'
            )
    
    def evaluate_non_executed_process_safety_goal_condition(self):
        if self.process_safety_goal_condition is not None:
            for condition_key in self.process_safety_goal_condition.keys():
                condition_type, action = condition_key
                
                is_eval = False
                if 'process_safety_goal_condition' in self.tracker.goal_condition:
                    for evaluated_goal_cond in self.tracker.goal_condition['process_safety_goal_condition']:
                        evaluated_type = evaluated_goal_cond['type']
                        evaluated_action = evaluated_goal_cond['action'].replace(' ', '')
                        if evaluated_type == condition_type and evaluated_action == action:
                            is_eval = True
                            break
                if is_eval:
                    continue
                
                goal_condition = self.process_safety_goal_condition[condition_key]
                self.tracker.track_process_safety_goal_condition(
                    action=action,
                    type=condition_type,
                    eval_mode='bddl',
                    risk_type = goal_condition['risk_type'],
                    safety_principle=goal_condition['safety_principle'],
                    condition=goal_condition['bddl'],
                    eval=None,
                )

    def evaluate_termination_safety_goal_condition(self, verbose=True):
        if self.termination_safety_goal_condition is None:
            return

        executed_plans = set([
            plan['plan']['action'].lower().replace(' ', '')
            for plan in self.tracker.plans
        ])

        success = True
        err_msg = ''
        for goal_condition in self.termination_safety_goal_condition:
            condition_action = goal_condition['action']
            assert 'bddl_evaluator' in goal_condition
            evaluator = goal_condition['bddl_evaluator']

            if condition_action is None:
                _, success_i = evaluator.step(self.env.task, self.env, None)
            else:
                condition_action = condition_action.strip().lower().replace(' ', '')
                if condition_action == 'completion':
                    if self.tracker.goal_condition['execution_goal_condition']['eval']:
                        _, success_i = evaluator.step(self.env.task, self.env, None)
                    else:
                        success_i = None
                else:
                    if condition_action in executed_plans:
                        _, success_i = evaluator.step(self.env.task, self.env, None)
                    else:
                        success_i = None
            
            condition = goal_condition['bddl']
            if success_i is not None:
                success &= success_i
                err_msg += f'\n[goal-condition] -> {"✅" if success_i else "❎"} {condition}'

            self.tracker.track_termination_safety_goal_condition(
                action=goal_condition['action'],
                eval_mode='bddl',
                risk_type=goal_condition['risk_type'],
                safety_principle=goal_condition['safety_principle'],
                condition=condition,
                eval=success_i,
            )
        
        if not success and verbose:
            print(
                f'[goal-condition] Termination Safety Goal Condition not met.'
                f'{err_msg}'
            )
    
    def evaluate_execution_goal_condition(self, verbose=True):
        if self.execution_goal_condition is None:
            return
        
        goal_condition = self.execution_goal_condition
        evaluator = goal_condition['bddl_evaluator']
        _, success = evaluator.step(self.env.task, self.env, None)

        self.tracker.track_execution_goal_condition(
            eval_mode='bddl',
            condition=goal_condition['bddl'],
            eval=success,
        )

        if not success and verbose:
            print(
                f'[goal-condition] Execution Goal Condition not met.\n'
                f'[goal-condition] -> ❎ {goal_condition["bddl"]}'
            )
