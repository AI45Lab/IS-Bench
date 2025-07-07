import re
import sys
from typing import List, Literal, Optional, Generator

import omnigibson as og
from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveError,
    ActionPrimitiveErrorGroup,
)
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitiveSet,
    StarterSemanticActionPrimitives,
)
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitiveSet,
    SymbolicSemanticActionPrimitives,
)
from omnigibson.envs import Environment
from omnigibson.macros import gm
import torch

from .ego_primitives import (
    EgoSemanticActionPrimitiveSet, 
    EgoSemanticActionPrimitives,
    VALID_PRIMITIVES
)
from .primitive_utils import find_task_related_object


class BadExecutionPlanError(Exception):
    pass


PRIMITIVE_SET = {
    'ego': EgoSemanticActionPrimitiveSet,
    'starter': StarterSemanticActionPrimitiveSet,
    'symbolic': SymbolicSemanticActionPrimitiveSet,
}

PRIMITIVES = {
    'ego': EgoSemanticActionPrimitives,
    'starter': StarterSemanticActionPrimitives,
    'symbolic': SymbolicSemanticActionPrimitives,
}


class Executor:

    def __init__(
        self, 
        env: Environment, 
        primitive_type: Literal['ego', 'starter', 'symbolic'] = 'ego', 
        verbose: bool = True,
        debug: bool = False,
    ):
        self.env = env
        self.verbose = verbose
        self.debug = debug

        self.primitive_set = PRIMITIVE_SET[primitive_type]

        controller_kwargs = {}
        if primitive_type == 'starter':
            controller_kwargs.update(dict(enable_head_tracking=False))
        self.controller = PRIMITIVES[primitive_type](env, **controller_kwargs)

    def execute_plans(self, plans: List[str]):
        for plan in plans:
            self.execute_plan(plan)
        
    def execute_plan(self, plan: str):
        """
            plan format: OPERATOR(OBJ@DESCRIPTOR, ...)
            e.g., 
                grasp(vegetables@inside the refrigerator)
                close(regrigerator)
        """
        if self.verbose:
            print(f'[executor] -> executing {plan}')
            sys.stdout.flush()

        if self.debug:
            debug_prompt = '[executor] Continue (y/Y)'
            if not gm.HEADLESS:
                debug_prompt += ' or Simulator (s/S)'
            print(f'{debug_prompt}: ')
            sys.stdout.flush()
             
            while cmd := input().upper() != "Y":
                if cmd == 'S':
                    if gm.HEADLESS:
                        print('[executor] Simulator (s/S) is not supported in HEADLESS mode.')
                        sys.stdout.flush()
                    else:   
                        self._simulator_loop()
                else:
                    print(f'{debug_prompt}: ')
                    sys.stdout.flush()

        action_seqs = self._parse_plan_to_action_seqs(plan)
        if action_seqs is None:  # Done
            return
        
        try:
            self._execute(action_seqs)
        except (ActionPrimitiveError, ActionPrimitiveErrorGroup) as e:
            if self.debug and gm.HEADLESS is False:
                print(f'[executor] catch error: {e}')
                sys.stdout.flush()
                self._simulator_loop()
            else:
                raise e
        
    def _execute(self, action_seqs: Generator[torch.Tensor, None, None]):
        for action in action_seqs:
            self.env.step(action)

    def _parse_plan_to_action_seqs(self, plan: str) -> Optional[Generator[torch.Tensor, None, None]]:
        pattern = r'([\w\W_]+)\((.*)\)'
        result = re.search(pattern, plan.strip())
        if result is None:
            raise BadExecutionPlanError(f'invalid plan "{plan}", expected "OPERATOR(OBJ@DESCRIPTOR)"')        
        operator, params = result.group(1).lower(), result.group(2).lower()

        if operator == 'done':
            return None
        
        if operator.upper() not in self.primitive_set._member_names_:
            raise BadExecutionPlanError(f'invalid operator "{operator}", expected {self.primitive_set._member_names_}')
        primitive = self.primitive_set._member_map_[operator.upper()]

        primitive_params = [param.strip() for param in params.split(',')]
        if len(primitive_params) != VALID_PRIMITIVES[operator.upper()]:
            raise BadExecutionPlanError(f'invalid params "{params}" for operator "{operator}"')

        object_refs = []
        for prim_param in primitive_params:
            if '@' in prim_param:
                obj, _ = prim_param.strip().split('@')
            else:
                obj = prim_param
            
            obj_ref = find_task_related_object(self.env, obj.strip())
            object_refs.append(obj_ref)

        try:
            action_seqs = self.controller.apply_ref(primitive, *object_refs)
        except TypeError:
            raise BadExecutionPlanError(f'invalid params "{params}" for operator "{operator}"')

        return action_seqs
            
    def _simulator_loop(self, interval=None):
        if interval is not None and isinstance(interval, int) and interval > 0:
            for _ in range(interval):
                self.env.step(torch.zeros(self.env.robots[0].action_dim))
        else:
            while True:
                self.env.step(torch.zeros(self.env.robots[0].action_dim))
