from typing import List, Literal, Optional, Tuple, TypeAlias, TypedDict, Union

from numpy.typing import ArrayLike as NumpyArrayLike
from omnigibson.termination_conditions.predicate_goal import PredicateGoal
import torch


ArrayLike: TypeAlias = Union[List, torch.Tensor, NumpyArrayLike]
PoseCoord: TypeAlias = Tuple[ArrayLike]


class GoalCondition(TypedDict):
    risk_type: Optional[str]
    safety_principle: Optional[str]
    safety_tip: Optional[str]
    action: Optional[str]
    type: Optional[str]

    bddl: Optional[str]
    bddl_evaluator: Optional[PredicateGoal]


class StepwisePlan(TypedDict):
    action: str
    caution: Optional[str]
