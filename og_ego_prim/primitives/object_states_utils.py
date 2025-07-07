
from typing import List, Optional

from omnigibson import object_states
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from omnigibson.envs import Environment
from omnigibson.object_states.object_state_base import BaseObjectState, RelativeObjectState
from omnigibson.object_states.particle_modifier import ParticleModifier
from omnigibson.objects import StatefulObject
from omnigibson.scenes import Scene
from omnigibson.systems.system_base import BaseSystem
from omnigibson.utils.constants import PrimType
import torch

from .attachment import (
    Attachment,
    FluidAttachment,
    StainAttachment,
    PlacementAttachment,
    Placement
)
from .primitive_utils import find_task_related_object
    

def is_visual_or_physical_particle_system(scene: Scene, system: BaseSystem) -> bool:
    if scene.is_visual_particle_system(system_name=system.name):
        return True
    if scene.is_physical_particle_system(system_name=system.name):
        return True
    return False


def get_obj_with_state(
    obj: StatefulObject | str, 
    state: BaseObjectState, 
    env: Optional[Environment] = None
) -> Optional[StatefulObject]:

    if isinstance(obj, str):
        assert env is not None
        obj = find_task_related_object(env, obj)
    
    if obj is None:
        return None
    if not hasattr(obj, 'states'):
        return None
    if state not in obj.states:
        return None
    return obj


def get_visible_task_related_objects(env: Environment) -> List[StatefulObject]:
    visible_task_related_objects = []

    for obj_name, obj_ref in env.task.object_scope.items():
        obj_ref = obj_ref.wrapped_obj
        if obj_name.strip().split('.')[0].strip() in ['agent', 'floor', 'ceiling', 'roof']:
            continue
        if obj_ref is None:
            continue
        if isinstance(obj_ref, BaseSystem) and not env.scene.is_visual_particle_system(system_name=obj_ref.name):
            continue

        is_obj_visible = True
        for placement_obj_name, placement_obj_ref in env.task.object_scope.items():
            placement_obj_ref = placement_obj_ref.wrapped_obj
            if placement_obj_name.strip().split('.')[0].strip() in ['agent', 'floor', 'ceiling', 'roof']:
                continue
            if placement_obj_ref is None:
                continue
            placement_obj = get_obj_with_state(placement_obj_ref, object_states.Open)
            if placement_obj is None:
                continue
            if placement_obj.states[object_states.Open].get_value():
                continue
            if is_target_object_predicate_with_obj(obj_ref, placement_obj_ref, object_states.Inside):
                is_obj_visible = False
                break

        if is_obj_visible:
            visible_task_related_objects.append(obj_ref)
    
    return visible_task_related_objects


def get_covered_systems(
    obj: StatefulObject | str, 
    env: Optional[Environment] = None
) -> Optional[List[BaseSystem]]:

    covering_systems = set()
    obj = get_obj_with_state(obj, object_states.Covered, env)
    if obj is None:
        return None
    
    for system in obj.scene.system_registry.objects:
        if not is_visual_or_physical_particle_system(obj.scene, system):
            continue
        if obj.states[object_states.Covered].get_value(system):
            covering_systems.add(system)
    
    return list(covering_systems)


def get_contained_systems(
    obj: StatefulObject | str,
    env: Optional[Environment] = None
) -> Optional[List[BaseSystem]]:

    contained_systems = set()
    obj = get_obj_with_state(obj, object_states.Contains, env) 
    if obj is None:
        return None
    
    for system in obj.scene.system_registry.objects:
        if not is_visual_or_physical_particle_system(obj.scene, system):
            continue
        if obj.states[object_states.Contains].get_value(system):
            contained_systems.add(system)

    return list(contained_systems)


def get_container(
    system: BaseSystem,
    env: Environment,
) -> Optional[StatefulObject]:
    for container_name in env.task.object_scope.keys():
        if 'agent' in container_name:
            continue

        container_obj = get_obj_with_state(container_name, object_states.Contains, env)
        if container_obj is None:
            continue

        if container_obj.states[object_states.Contains].get_value(system):
            return container_obj
    
    return None


def get_produced_systems(
    obj: StatefulObject | str,
    env: Optional[Environment] = None
) -> Optional[List[BaseSystem]]:

    producing_systems = set()
    obj = get_obj_with_state(obj, object_states.ParticleSource, env)
    if obj is None:
        return None

    for system in obj.scene.system_registry.objects:
        if obj.states[object_states.ParticleSource].check_conditions_for_system(system.name):
            producing_systems.add(system)
    
    return list(producing_systems)


def get_saturated_systems(
    obj: StatefulObject | str,
    env: Optional[Environment] = None
) -> Optional[List[BaseSystem]]:

    saturated_systems = set()
    obj = get_obj_with_state(obj, object_states.Saturated, env)
    if obj is None:
        return None
    
    for system in obj.scene.system_registry.objects:
        if not is_visual_or_physical_particle_system(obj.scene, system):
            continue
        if obj.states[object_states.Saturated].get_value(system):
            saturated_systems.add(system)
    
    return list(saturated_systems)


def get_supported_systems(
    tool: StatefulObject,
    systems: List[BaseSystem],
    modifier: ParticleModifier,
) -> List[BaseSystem]:
    supported_systems = set()

    for system in systems:
        if tool.states[modifier].supports_system(system.name):
            supported_systems.add(system)

    return list(supported_systems)


def get_modified_systems(
    tool: StatefulObject,
    systems: List[BaseSystem],
    modifier: ParticleModifier,
) -> List[BaseSystem]:
    modified_systems = set()

    for system in systems:
        if tool.states[modifier].check_conditions_for_system(system.name):
            modified_systems.add(system)

    return list(modified_systems)


def is_target_object_predicate_with_obj(
    target_obj: StatefulObject, 
    obj: StatefulObject, 
    predicate: RelativeObjectState
) -> bool:
    # Maybe a FluidSystem like water
    if not hasattr(target_obj, 'states'):
        return False
    if not predicate in target_obj.states:
        return False
    return target_obj.states[predicate].get_value(obj)


def get_placement_objects(
    obj: StatefulObject, 
    env: Environment, 
    predicates: Optional[RelativeObjectState | List[RelativeObjectState]] = None,
) -> Optional[List[Placement]]:
    if predicates is None:
        predicates = [object_states.Inside, object_states.OnTop]
    if not isinstance(predicates, list):
        predicates = [predicates]
    
    if obj.prim_type == PrimType.CLOTH:
        return None
    
    placements = []
    for target_obj_name in env.task.object_scope.keys():
        if 'agent' in target_obj_name:
            continue

        target_obj = find_task_related_object(env, target_obj_name.strip())
        if target_obj is None:
            continue 

        for predicate in predicates:
            if is_target_object_predicate_with_obj(target_obj, obj, predicate):
                placements.append(Placement(target_obj, predicate))
    
    return placements


def get_cooked_system(cooked_system: str, env: Environment) -> Optional[BaseSystem]:
    for system_name in env.task.object_scope.keys():
        if system_name.startswith('agent'):
            continue
        
        if cooked_system in system_name and cooked_system in env._scene.available_systems:
            return env._scene.get_system(cooked_system)
    
    return None


def is_cloth_place_on_other(target_obj: StatefulObject, placement_obj: StatefulObject) -> bool:
    if not hasattr(target_obj, 'prim_type') or not hasattr(placement_obj, 'prim_type'):
        return False

    return target_obj.prim_type == PrimType.CLOTH \
        and placement_obj.prim_type == PrimType.RIGID


def check_open_before_grasp(
    obj: StatefulObject, 
    env: Environment
):
    for parent_obj_name in env.task.object_scope.keys():
        parent_obj = get_obj_with_state(parent_obj_name, object_states.Open, env)
        if parent_obj is None:
            continue

        if is_target_object_predicate_with_obj(obj, parent_obj, object_states.Inside) \
              and parent_obj.states[object_states.Open].get_value() is False:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                f"You should open {parent_obj.name} first, because currently the operated object is place inside {parent_obj.name}.",
                {"operated object": obj.name, "parent object should be opened first": parent_obj.name},
            )


def check_open_before_placement(
    obj: StatefulObject,
    env: Optional[Environment] = None,
):
    obj = get_obj_with_state(obj, object_states.Open, env)
    if obj is None:
        return
    
    if obj.states[object_states.Open].get_value() is False:
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            f"You should open {obj.name} first before placing other object in {obj.name}.",
            {"object": obj.name},
        )
    

def check_close_before_toggle_on(
    obj: StatefulObject,
    env: Optional[Environment] = None,
):
    obj = get_obj_with_state(obj, object_states.Open, env)
    if obj is None:
        return
    
    if obj.states[object_states.Open].get_value():
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            "The machine must be closed and then toggled on",
            {"target object": obj.name},
        )


def check_toggle_off_before_open(
    obj: StatefulObject,
    env: Optional[Environment] = None,
):
    obj = get_obj_with_state(obj, object_states.ToggledOn, env)
    if obj is None:
        return
    
    if obj.states[object_states.ToggledOn].get_value():
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            "The machine must be toggled off before open",
            {"target object": obj.name},
        )


def check_heat_source_before_cook(
    obj: StatefulObject, 
    env: Environment
):
    adjacency = obj.states[object_states.VerticalAdjacency].get_value()
    
    placed_heat_source = None
    for heat_source_name in env.task.object_scope.keys():
        heat_source = get_obj_with_state(heat_source_name, object_states.HeatSourceOrSink, env)
        if heat_source is None:
            continue

        if heat_source.states[object_states.HeatSourceOrSink].requires_inside:
            if not is_target_object_predicate_with_obj(obj, heat_source, object_states.Inside):
                continue
        else:
            if heat_source not in adjacency.negative_neighbors or heat_source in adjacency.positive_neighbors:
                continue
        
        placed_heat_source = heat_source
        if object_states.ToggledOn in heat_source.states \
              and heat_source.states[object_states.ToggledOn].get_value() is False:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                f"You should toggle the heat source on when cooking.",
                {"target object": obj.name, "heat source": heat_source.name},
            )

    if placed_heat_source is None:
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            f"You should first place the target inside or on top of a heat source before cooking.",
            {"target object": obj.name},
        )


def capture_attachments(obj: StatefulObject, env: Environment) -> List[Attachment]:
    attachments: List[Attachment] = []

    contained_systems = get_contained_systems(obj)
    if contained_systems is not None:
        for system in contained_systems:
            if hasattr(system, 'is_fluid') and system.is_fluid:  # 
                attachment = FluidAttachment(obj, system)
                attachment.remove_attachment()
                attachments.append(attachment)
    
    # covering_systems = get_covered_systems(obj)
    # if covering_systems is not None:
    #     for system in covering_systems:
    #         if not system.is_fluid:
    #             attachment = StainAttachment(obj, system)
    #             attachment.remove_attachment()
    #             attachments.append(attachment)

    placement_objects = get_placement_objects(obj, env)
    if placement_objects is not None:
        for placement in placement_objects:
            attachment = PlacementAttachment(obj, placement)
            attachment.remove_attachment()
            attachments.append(attachment)

    return attachments


def recover_attachments(attachments: List[Attachment]):
    for attachment in attachments:
        attachment.recover_attachment()

    del attachments
