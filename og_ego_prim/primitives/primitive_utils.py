from typing import Optional

from omnigibson.envs import Environment
from omnigibson.objects import BaseObject


def find_task_related_object(
    env: Environment, 
    target_name: str, 
    retain_wrapper: bool = False,
) -> Optional[BaseObject]:
    task_related_objects = sorted(
        list(env.task.object_scope.keys()), 
        key=lambda name: len(name), 
        reverse=True
    )

    if target_name in task_related_objects:
        ref = env.task.object_scope[target_name]
        target_obj = ref if retain_wrapper else ref.wrapped_obj
        return target_obj

    for name in task_related_objects:
        ref = env.task.object_scope[name]
        if 'agent' in name:
            continue

        target_name = target_name.strip()
        candidate_name = name.split('.')[0].strip()
        if target_name in candidate_name:
            target_obj = ref if retain_wrapper else ref.wrapped_obj
            return target_obj
    
    return None
