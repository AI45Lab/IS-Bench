from bddl.condition_evaluation import compile_state
from bddl.parsing import scan_tokens, package_predicates
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.predicate_goal import PredicateGoal


def compile_bddl_goal_condition(task: BaseTask, goal_conds: str) -> PredicateGoal:
    tokens = scan_tokens(string=goal_conds)
    assert tokens[0] == ':goal'

    goal_conds = []
    package_predicates(tokens[1], goal_conds, '', 'goals')

    goal_conds = compile_state(
        goal_conds,
        task.backend,
        scope=task.object_scope,
        object_map=task.activity_conditions.parsed_objects
    )
    evaluator = PredicateGoal(goal_fcn=lambda: goal_conds)
    return evaluator
