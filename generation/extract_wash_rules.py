from og_ego_prim.utils.monkey_patch import add_monkey_patch
add_monkey_patch()

import argparse
import json
import os
import time
from typing import Dict, List

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.objects.object_base import BaseObject

from og_ego_prim.benchmark import build_benchmark
from og_ego_prim.primitives import find_task_related_object

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = True
gm.HEADLESS = True
# gm.ENABLE_FLATCACHE = True

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default=None)
parser.add_argument('--scene', type=str, default=None)
parser.add_argument('--work_dir', type=str, default='./work_dir')


def get_name(name: str) -> str:
    return name.strip().split('.')[0].strip()

def get_object_abilities(env: og.Environment) -> List[BaseObject]:
    object_scope = env.task.object_scope
    object_list = []

    for obj_name, obj in object_scope.items():
        if 'agent' in obj_name:
            continue
        abilities = None
        if hasattr(obj, 'wrapped_obj'):
            obj_wrapped = obj.wrapped_obj
            if hasattr(obj_wrapped, 'abilities'):
                abilities = obj_wrapped.abilities
        object_list.append({obj_name: abilities})
    return object_list

def find_cleaning_tools(env: og.Environment) -> List[BaseObject]:
    object_scope = env.task.object_scope

    cleaning_tools = []
    for obj_name in object_scope.keys():
        obj = find_task_related_object(env, obj_name)
        if obj is None:
            continue
        if object_states.ParticleRemover in obj.states:
            cleaning_tools.append((obj, obj_name))
    
    return cleaning_tools


def get_conditions(tool: BaseObject, system: str) -> Dict:
    # find conditions how tool support (remove) system
    abilities = tool.abilities['particleRemover']
    if not 'conditions' in abilities:
        valid_conditions = None
    elif not system in tool.abilities['particleRemover']['conditions']:
        valid_conditions = None
    else:
        valid_conditions = tool.abilities['particleRemover']['conditions'][system]

    if valid_conditions is not None:
        return {'type': 'valid', 'conditions': valid_conditions}
        
    if tool.scene.is_fluid_system(system):
        default_conditions = tool.abilities['particleRemover']['default_fluid_conditions']
    elif tool.scene.is_physical_particle_system(system):
        default_conditions = tool.abilities['particleRemover']['default_non_fluid_conditions']
    elif tool.scene.is_visual_particle_system(system):
        default_conditions = tool.abilities['particleRemover']['default_visual_conditions']
    else:
        default_conditions = None

    if default_conditions is None:
        return None

    return {'type': 'default', 'conditions': default_conditions}


def is_obj_produce_detergent(obj: BaseObject, detergent: str) -> bool:
    if 'particleSource' not in obj.abilities:
        return False
    abilities = obj.abilities['particleSource']
    if 'conditions' not in abilities:
        return False
    return detergent in abilities['conditions']


def is_obj_contain_detergent(obj: BaseObject, detergent: str) -> bool:
    if object_states.Contains not in obj.states:
        return False
    if detergent not in obj.scene.available_systems:
        return False

    detergent_system = obj.scene.get_system(detergent, force_init=False)
    return obj.states[object_states.Contains].get_value(detergent_system)


def find_detergent_source(detergent: str, env: og.Environment) -> List[str]:
    object_scope = env.task.object_scope
    detergent = get_name(detergent)

    sources = []
    for obj_name in object_scope.keys():
        obj = find_task_related_object(env, obj_name)
        if obj is None:
            continue 

        # if obj can produce detergent or obj contains detergent
        if is_obj_produce_detergent(obj, detergent) or is_obj_contain_detergent(obj, detergent):
            sources.append(get_name(obj_name))

    return sources


def get_rules_for_tool(tool: BaseObject, tool_name: str, env: og.Environment) -> List[Dict]:
    object_scope = env.task.object_scope

    rules = []
    for system_name in object_scope.keys():
        system = get_name(system_name)
        if not system in tool.scene.available_systems:
            continue
        if not tool.scene.is_system_active(system):
            continue
        if not tool.states[object_states.ParticleRemover].supports_system(system):
            continue

        conditions = get_conditions(tool, system)
        if conditions is None:
            continue

        rule = {
            'cleaning_tool': get_name(tool_name),
            'support_system': system,
        }

        if not conditions['conditions']:
            rules.append(rule)
            continue

        wash_conditions = []
        for condition in conditions['conditions']:
            condition_type, condition_value = condition
            wash_condition = {
                'type': condition_type,
                'value': condition_value,
            }
            
            if condition_type == 'saturated':
                source = find_detergent_source(condition_value, env)
                if not source:
                    continue
                wash_condition['source'] = source

            wash_conditions.append(wash_condition)

        if not wash_conditions:
            continue

        rule['conditions'] = wash_conditions
        rules.append(rule)

    return rules


def extract_wash_rules(
    task: str,
    scene: str,
    work_dir: str,
):
    benchmark = build_benchmark(
        task=task, 
        scene=scene, 
        ego_view=True, 
        online_object_sampling=False, 
        debug=True
    )

    output_dir = os.path.join(work_dir, 'rules', 'wash')
    obj_output_dir = os.path.join(work_dir, 'rules', 'obj_states')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(obj_output_dir, exist_ok=True)

    object_list = get_object_abilities(benchmark.env)
    cleaning_tools = find_cleaning_tools(benchmark.env)

    wash_rules = []
    for tool in cleaning_tools:
        rules = get_rules_for_tool(tool[0], tool[1], benchmark.env)
        wash_rules.extend(rules)

    save_path = os.path.join(output_dir, f'{benchmark.task_name}.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(wash_rules, f, indent=4, ensure_ascii=False)

    obj_save_path = os.path.join(obj_output_dir, f'{benchmark.task_name}.json')
    with open(obj_save_path, 'w', encoding='utf-8') as f:
        json.dump(object_list, f, indent=4, ensure_ascii=False)

    time.sleep(3)
    og.clear()


if __name__ == "__main__":
    args = parser.parse_args()
    # task_list = []
    # scene_list = []
    # task_path = os.path.join(args.work_dir, 'wash_task_list')
    # for file in os.listdir(task_path):
    #     with open(os.path.join(task_path, file)) as f:
    #         task_config = json.load(f)
    #     task_list.append(task_config['task_name'])
    #     scene_list.append(task_config['scene_models'][0])
    
    # for i in range(len(task_list)):
    extract_wash_rules(
        task=args.task,
        scene=args.scene,
        work_dir=args.work_dir,
    )
