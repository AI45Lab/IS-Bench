from typing import List, Optional

from aenum import IntEnum, auto
from omnigibson import object_states
from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveError, 
    ActionPrimitiveErrorGroup
)
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
)
from omnigibson.envs import Environment
from omnigibson.objects import StatefulObject
from omnigibson.object_states.object_state_base import RelativeObjectState 
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.systems import BaseSystem
from omnigibson.transition_rules import SlicingRule
from omnigibson.utils.constants import PrimType
import torch

from .object_states_utils import (
    get_covered_systems, 
    get_contained_systems,
    get_produced_systems,
    get_supported_systems,
    get_modified_systems,
    get_container,
    capture_attachments,
    recover_attachments,
    check_open_before_grasp,
    check_open_before_placement,
    check_close_before_toggle_on,
    check_toggle_off_before_open,
    check_heat_source_before_cook,
    get_cooked_system,
    is_cloth_place_on_other,
    get_placement_objects,
    is_visual_or_physical_particle_system,
    find_task_related_object,
    get_obj_with_state,
    is_target_object_predicate_with_obj
)


class EgoSemanticActionPrimitiveSet(IntEnum):
    _init_ = "value __doc__"
    PLACE_ON_TOP = auto(), "Place the target_obj on top of placement_obj"
    PLACE_INSIDE = auto(), "Place the target_obj inside placement_obj"
    OPEN = auto(), "Open an target_obj"
    CLOSE = auto(), "Close an target_obj"
    TOGGLE_ON = auto(), "Toggle an target_obj on"
    TOGGLE_OFF = auto(), "Toggle an target_obj off"
    WIPE = auto(), "Wipe the target_obj with the cleaning_tool"
    CUT = auto(), "Cut (slice or dice) the target_obj with the cutting_tool"
    SOAK_UNDER = auto(), "Soak the target_obj with particles produced by the fluid_source"
    SOAK_INSIDE = auto(), "Soak the target_obj with particles in the fluid_container"
    FILL_WITH = auto(), "Fill the target_obj with particles produced by the fluid source"
    POUR_INTO = auto(), "Pour the particle in the fluid_container into the target_obj (usually a container)"
    WAIT_FOR_COOKED = auto(), "Wait for the cook process of the object to final"
    WAIT_FOR_WASHED = auto(), "Wait for the wash process fo the wash machine to final"
    WAIT = auto(), "Wait for the object to change, such as waiting for the object to rise to room temperature."
    SPREAD = auto(), "Spread some particles onto some object, make object covered with these particles"
    WAIT_FOR_FROZEN = auto(), "Wait something in the refridge to frozen"


VALID_PRIMITIVES = {
    "PLACE_ON_TOP": 2,
    "PLACE_INSIDE": 2,
    "OPEN": 1,
    "CLOSE": 1,
    "TOGGLE_ON": 1,
    "TOGGLE_OFF": 1,
    "WIPE": 2,
    "CUT": 2,
    "SOAK_INSIDE": 2,
    "SOAK_UNDER": 2,
    "FILL_WITH": 2,
    "POUR_INTO": 2,
    "SPREAD": 2,
    "WAIT": 1,
    "WAIT_FOR_COOKED": 1,
    "WAIT_FOR_WASHED": 1,
    "WAIT_FOR_FROZEN": 2,
}


class EgoSemanticActionPrimitives(StarterSemanticActionPrimitives):

    def __init__(self, env: Environment):
        super().__init__(env)
        self.controller_functions = {
            EgoSemanticActionPrimitiveSet.PLACE_ON_TOP: self._place_on_top,
            EgoSemanticActionPrimitiveSet.PLACE_INSIDE: self._place_inside,
            EgoSemanticActionPrimitiveSet.OPEN: self._open,  # done
            EgoSemanticActionPrimitiveSet.CLOSE: self._close,  # done
            EgoSemanticActionPrimitiveSet.TOGGLE_ON: self._toggle_on,  # done
            EgoSemanticActionPrimitiveSet.TOGGLE_OFF: self._toggle_off,  # done
            EgoSemanticActionPrimitiveSet.WIPE: self._wipe,
            EgoSemanticActionPrimitiveSet.CUT: self._cut,
            EgoSemanticActionPrimitiveSet.SOAK_INSIDE: self._soak_inside,
            EgoSemanticActionPrimitiveSet.SOAK_UNDER: self._soak_under,
            EgoSemanticActionPrimitiveSet.FILL_WITH: self._fill_with,
            EgoSemanticActionPrimitiveSet.POUR_INTO: self._pour_into,
            EgoSemanticActionPrimitiveSet.WAIT_FOR_COOKED: self._wait_for_cooked,
            EgoSemanticActionPrimitiveSet.WAIT_FOR_WASHED: self._wait_for_washed,
            EgoSemanticActionPrimitiveSet.WAIT: self._wait,
            EgoSemanticActionPrimitiveSet.SPREAD: self._spread,
            EgoSemanticActionPrimitiveSet.WAIT_FOR_FROZEN: self._wait_for_frozen,
        }
        self.env = env
        self.attachments = []

    def apply_ref(self, primitive, *args):
        if any(isinstance(arg, BaseRobot) for arg in args):
            raise ActionPrimitiveErrorGroup(
                [
                    ActionPrimitiveError(
                        ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                        "Cannot call a symbolic semantic action primitive with a robot as an argument.",
                    )
                ]
            )
        
        try:
            yield from self.controller_functions[primitive](*args)
        except ActionPrimitiveError as e:
            raise ActionPrimitiveErrorGroup([e])

        # Settle before returning.
        try:
            yield from self._settle_robot()
        except ActionPrimitiveError:
            pass

    def _open_or_close(self, target_obj: StatefulObject, should_open: bool):
        if object_states.Open not in target_obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not openable.",
                {"target object": target_obj.name},
            )
        
        # Don't do anything if the object is already closed and we're trying to close.
        if should_open == target_obj.states[object_states.Open].get_value():
            return
        
        if should_open is True:
            check_toggle_off_before_open(target_obj)

        inside_placements = get_placement_objects(target_obj, self.env, object_states.Inside)
        
        # Set the value
        target_obj.states[object_states.Open].set_value(should_open, fully=True)
        yield from self._settle_robot()

        if inside_placements:
            for placement in inside_placements:
                obj = placement.object
                if not is_target_object_predicate_with_obj(obj, target_obj, object_states.Inside):
                    yield from self._place_inside(obj, target_obj, skip_check=True)

        if target_obj.states[object_states.Open].get_value() != should_open:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "The object did not open or close as expected. Maybe try again",
                {"target object": target_obj.name, "is it currently open": target_obj.states[object_states.Open].get_value()},
            )

    def _place_on_top(self, target_obj: StatefulObject, placement_obj: StatefulObject, **kwargs):
        yield from self._place_with_predicate(target_obj, placement_obj, object_states.OnTop, **kwargs)

    def _place_inside(self, target_obj: StatefulObject, placement_obj: StatefulObject, **kwargs):
        yield from self._place_with_predicate(target_obj, placement_obj, object_states.Inside, **kwargs)

    def _sample_on_top_heat_source(self, target_obj: StatefulObject, heat_source: StatefulObject):
        heating_element = heat_source.states[object_states.HeatSourceOrSink].link
        heating_element_positions = heating_element.get_position_orientation()[0] + torch.tensor([0, 0, 0.1])
        target_obj.set_position_orientation(position=heating_element_positions)
        yield from self._settle_robot()

    def _sample_placement_with_predicate(
        self, 
        target_obj: StatefulObject, 
        placement_obj: StatefulObject, 
        predicate: RelativeObjectState,
    ):
        placement_obj_pose = placement_obj.get_position_orientation()

        attempts = 0
        while attempts < 5:
            attempts += 1
            try:
                # Find a spot to put it
                predicated_pose = self._sample_pose_with_object_and_predicate(
                    predicate, target_obj, placement_obj
                )
            except Exception as e:
                print(f'Attempt {attempts}: {e}')
                continue

            # Actually move the target object to the spot and step a bit to settle it.
            target_obj.set_position_orientation(*predicated_pose)
            yield from self._settle_robot()

            if target_obj.states[predicate].get_value(placement_obj):
                break
            else:
                # recover if failed
                placement_obj.set_position_orientation(*placement_obj_pose)
                placement_obj.keep_still()
                yield from self._settle_robot()

    def _place_with_predicate(
        self, 
        target_obj: StatefulObject | BaseSystem, 
        placement_obj: StatefulObject, 
        predicate: RelativeObjectState,
        skip_check: bool = False,
    ):
        if isinstance(target_obj, BaseSystem) and is_visual_or_physical_particle_system(target_obj.scene, target_obj):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Cannot place system to desired position, perhaps place its container like bottle to the position",
                {"system (target object)": target_obj.name},
            )

        if predicate == object_states.OnTop and is_cloth_place_on_other(target_obj, placement_obj):
            predicate = object_states.Overlaid
        if predicate not in [object_states.OnTop, object_states.Inside, object_states.Overlaid]:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Only support place the target object OnTop, OverLaid or Inside the placement object.",
                {"provided_predicate": predicate.__class__.__name__},
            )

        if not skip_check:
            check_open_before_grasp(target_obj, self.env)
            if predicate == object_states.Inside:
                check_open_before_placement(placement_obj)

        attachments = capture_attachments(target_obj, self.env)
        
        if object_states.HeatSourceOrSink in placement_obj.states and predicate == object_states.OnTop and \
              not placement_obj.states[object_states.HeatSourceOrSink].requires_inside:
            yield from self._sample_on_top_heat_source(target_obj, placement_obj)

        elif predicate == object_states.Overlaid:
            placement_pose = placement_obj.get_position_orientation()[0]
            position = placement_pose + torch.tensor([0, 0, 0.1])
            target_obj.set_position_orientation(position=position)
            yield from self._settle_robot()

        else:
            yield from self._sample_placement_with_predicate(target_obj, placement_obj, predicate)

            # Last attempt to directly place the target object with predicate
            if not target_obj.states[predicate].get_value(placement_obj):
                if predicate == object_states.OnTop:  # ontop 
                    placement_pose = placement_obj.get_position_orientation()[0]
                    position = placement_pose + torch.tensor([0, 0, 0.1])
                else: # others, inside
                    placement_obj_center = placement_obj.get_base_aligned_bbox()[0]
                    position = placement_obj_center

                target_obj.set_position_orientation(position=position)
                yield from self._settle_robot()

        if attachments:
            recover_attachments(attachments)
            yield from self._settle_robot()

        # check
        error = ActionPrimitiveError(
            ActionPrimitiveError.Reason.EXECUTION_ERROR,
            "Failed to place target object at the desired place (probably dropped).",
            {"dropped object": target_obj.name, "placement object": placement_obj.name, "predicate": predicate.__class__.__name__},
        )
        if predicate == object_states.OnTop:
            adjacency = target_obj.states[object_states.VerticalAdjacency].get_value()
            if not placement_obj in adjacency.negative_neighbors or placement_obj in adjacency.positive_neighbors:
                raise error
        elif predicate == object_states.Inside and not target_obj.states[predicate].get_value(placement_obj):
                raise error
        elif predicate == object_states.Overlaid and not target_obj.states[object_states.Touching].get_value(placement_obj):
                raise error

    def _toggle(self, target_obj: StatefulObject, value: bool):
        if object_states.ToggledOn not in target_obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not toggleable.",
                {"target object": target_obj.name},
            )
        
        if target_obj.states[object_states.ToggledOn].get_value() == value:
            return
        
        if value is True:
            check_close_before_toggle_on(target_obj)

        # Call the setter
        target_obj.states[object_states.ToggledOn].set_value(value)
        yield from self._settle_robot()

        # Check that it actually happened
        if target_obj.states[object_states.ToggledOn].get_value() != value:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "The object did not toggle as expected - maybe try again",
                {
                    "target object": target_obj.name,
                    "is it currently toggled on": target_obj.states[object_states.ToggledOn].get_value(),
                },
            )
        
    def _wipe(self, target_obj: StatefulObject, cleaning_tool: StatefulObject):
        # Check that the cleaning tool can remove those particles
        if object_states.ParticleRemover not in cleaning_tool.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The cleaning tool is not a particle remover.",
                {"cleaning tool": cleaning_tool.name},
            )
        
        covered_systems = get_covered_systems(target_obj)
        # Check that the target object is coverable
        if covered_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not coverable by any particles, so there is no need to wipe it.",
                {"target object": target_obj.name},
            )
        
        check_open_before_grasp(target_obj, self.env)
        check_open_before_grasp(cleaning_tool, self.env)

        # Check if the target object has any particles on it
        if not covered_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not covered by any particles.",
                {"target object": target_obj.name},
            )
        
        supported_systems = get_supported_systems(
            cleaning_tool, covered_systems, object_states.ParticleRemover
        )
        if not supported_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is covered only by particles that this cleaning tool cannot remove.",
                {
                    "target object": target_obj.name,
                    "cleaning tool": cleaning_tool.name,
                    "particles the target object is covered by": sorted(x.name for x in covered_systems),
                    "particles the cleaning tool can remove": sorted(
                        [x for x in cleaning_tool.states[object_states.ParticleRemover].conditions.keys()]
                    ),
                },
            )
        
        removed_systems = get_modified_systems(
            cleaning_tool, supported_systems, object_states.ParticleRemover
        )
        if not removed_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is covered by some particles that this cleaning tool can normally remove, but needs to be in a different state to do so (e.g. toggled on, soaked by another fluid first, etc.).",
                {
                    "target object": target_obj.name,
                    "cleaning tool": cleaning_tool.name,
                    "particles the target object is covered by": sorted(x.name for x in covered_systems),
                },
            )
        
        # If so, remove the particles on the target object
        MAX_WIPE_NUMS = 3
        for i in range(MAX_WIPE_NUMS):  # 最多wipe 三次
            print(f"######[INFO] Try to Wipe at times {i}")
            for system in removed_systems:
                target_obj.states[object_states.Covered].set_value(system, False)
                yield from self._settle_robot()
            if not get_covered_systems(target_obj):  # wipe OK，提前退出
                break
                    

    def _cut(self, target_obj: StatefulObject, cutting_tool: StatefulObject):
        # Check that cutting tool is a slicer
        if "slicer" not in cutting_tool._abilities:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The cutting tool cannot slice object.",
                {"cutting tool": cutting_tool.name},
            )
        
        # Check that the target object is sliceable
        if "sliceable" not in target_obj._abilities and "diceable" not in target_obj._abilities:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not sliceable or diceable.",
                {"target object": target_obj.name},
            )

        check_open_before_grasp(target_obj, self.env)
        check_open_before_grasp(cutting_tool, self.env)

        added_obj_attrs, removed_objs = [], []
        (slicing_rule,) = [
            rule 
            for rule in target_obj.scene.transition_rule_api.active_rules 
            if isinstance(rule, SlicingRule)
        ]
        output = slicing_rule.transition({"sliceable": [target_obj]})

        added_obj_attrs += output.add
        removed_objs += output.remove
        target_obj.scene.transition_rule_api.execute_transition(
            added_obj_attrs=added_obj_attrs, removed_objs=removed_objs
        )
        yield from self._settle_robot()
    
    def _soak_with_fluid_systems(self, target_obj: StatefulObject, systems: List[BaseSystem]) -> Optional[List[BaseSystem]]:
        # Check that the target object can saturated (remove) with particles in container or producer
        supported_systems = get_supported_systems(
            target_obj, systems, object_states.ParticleRemover
        )
        if not supported_systems:
            return None
        
        removed_systems = get_modified_systems(
            target_obj, supported_systems, object_states.ParticleRemover
        )
        if not removed_systems:
            return None
        
        return removed_systems

    def _soak_under(self, target_obj: StatefulObject, fluid_source: StatefulObject):
        # Check that the target object can saturated (remove) with particles
        if object_states.Saturated not in target_obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The currently grasped object cannot soak particles.",
                {"object in hand": target_obj.name},
            )

        check_open_before_grasp(target_obj, self.env)

        # Check that the fluid source should either be a particle producer or a particle container
        produced_systems = get_produced_systems(fluid_source)
        if produced_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The fluid source is not a particle producer, so you can not soak target object.",
                {"fluid source": fluid_source.name},
            )
        if not produced_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The fluid source currently does not produce particles, may be some conditions for producing particles not met, e.g., the fluid source should be toggled on.",
                {"fluid source": fluid_source.name}
            )
        
        removed_produced_systems = self._soak_with_fluid_systems(target_obj, produced_systems)
        if removed_produced_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object cannot soak with particles from fluid source, maybe target object cannot support the particles or saturation reaches upper limit",
                {
                    "target object": target_obj.name,
                    "fluid source": fluid_source.name,
                    "particles the target object can soak:": sorted(
                        [x for x in target_obj.states[object_states.ParticleRemover].conditions.keys()]
                    ),
                    "particles the fluid source produces": sorted(x.name for x in produced_systems) if produced_systems else None,
                },
            )
        
        # Remove the particles.
        for system in removed_produced_systems:
            target_obj.states[object_states.Saturated].set_value(system, True)
        
        yield from self._settle_robot()
    
    def _soak_inside(self, target_obj: StatefulObject, fluid_container: StatefulObject):
        # Check that the target object can saturated (remove) with particles
        if object_states.Saturated not in target_obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The currently grasped object cannot soak particles.",
                {"target object": target_obj.name},
            )

        check_open_before_grasp(target_obj, self.env)
        check_open_before_grasp(fluid_container, self.env)

        contained_systems = get_contained_systems(fluid_container)
        if contained_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The fluid container is not a particle container, so you can not soak target object.",
                {"fluid container": fluid_container.name},
            )
        
        if not contained_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The fluid source currently does not contain any particles.",
                {"fluid container": fluid_container.name}
            )
        
        removed_contained_systems = self._soak_with_fluid_systems(target_obj, contained_systems)
        if removed_contained_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object cannot soak with particles from fluid container, maybe target object cannot support the particles or saturation reaches upper limit",
                {
                    "target object": target_obj.name,
                    "fluid container": fluid_container.name,
                    "particles the target object can soak:": sorted(
                        [x for x in target_obj.states[object_states.ParticleRemover].conditions.keys()]
                    ),
                    "particles in the fluid container": sorted(x.name for x in contained_systems) if contained_systems else None,
                },
            )
        
        # Remove the particles.
        for system in removed_contained_systems:
            target_obj.states[object_states.Saturated].set_value(system, True)
        
        yield from self._settle_robot()

    def _fill_with(self, target_obj: StatefulObject, fluid_source: StatefulObject):
        # Check that target object is fillable
        contained_systems = get_contained_systems(target_obj)
        if contained_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not fillable by particles, so you can not fill anything in it.",
                {"target object": target_obj.name},
            )

        check_open_before_grasp(target_obj, self.env)

        # Check that the fluid source should be a particle producer
        produced_systems = get_produced_systems(fluid_source)
        if produced_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The fluid source is not a particle producer, so you can not fill target object.",
                {"fluid source": fluid_source.name},
            )
        if not produced_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The fluid source currently does not produce any particles, may be some conditions for producing particles not met, e.g., the fluid source should be toggled on.",
                {"fluid source": fluid_source.name}
            )
        
        # If so, fill the target object with all the particles from fluid source
        for system in produced_systems:
            target_obj.states[object_states.Filled].set_value(system, True)
            yield from self._settle_robot()

        # for system in produced_systems:
        #     if not target_obj.states[object_states.Contains].get_value(system):
        #         raise ActionPrimitiveError(
        #             ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
        #             "The container does not contain target particle, maybe the container fall over and the particles are scattered.",
        #             {"container": target_obj.name, "particle": system.name}
        #         )

    def _pour_into(self, fluid_container: StatefulObject, target_obj: StatefulObject):
        # Check that target object is fillable
        contained_systems = get_contained_systems(target_obj)
        if contained_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not fillable by particles, so you can not fill anything in it.",
                {"target object": target_obj.name},
            )

        check_open_before_grasp(target_obj, self.env)
        check_open_before_grasp(fluid_container, self.env)

        # Check that the fluid container contains particles
        contained_systems = get_contained_systems(fluid_container)
        if contained_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The fluid container is not a particle container, so you can not fill target object.",
                {"fluid container": fluid_container.name},
            )
        if not contained_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The fluid container currently does not contain any particles.",
                {"fluid container": fluid_container.name}
            )

        # If so, fill the target object with all the particles from fluid source
        for system in contained_systems:
            target_obj.states[object_states.Filled].set_value(system, True)
            yield from self._settle_robot()

        # for system in contained_systems:
        #     if not target_obj.states[object_states.Contains].get_value(system):
        #         raise ActionPrimitiveError(
        #             ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
        #             "The container does not contain target particle, maybe the container fall over and the particles are scattered.",
        #             {"container": target_obj.name, "particle": system.name}
        #         )
    
    def _spread(self, fluid_container: StatefulObject, target_obj: StatefulObject):
        contained_systems = get_contained_systems(fluid_container)
        # check current object is a particle container
        if contained_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The current container object is not fillable by particles, so you can not use it to spread",
                {"target object": fluid_container.name},
            )
        # Check if the current object has any particles in it
        if not contained_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The current container object does not contain any particles.",
                {"target object": fluid_container.name},
            )

        check_open_before_grasp(fluid_container, self.env)
        check_open_before_grasp(target_obj, self.env)

        for system in contained_systems:
            if target_obj.prim_type != PrimType.CLOTH:
                target_obj.states[object_states.Covered].set_value(system, True)
            else: 
                target_obj.states[object_states.Saturated].set_value(system, True)
            yield from self._settle_robot()

    def _cook_particle_system(self, target_system: BaseSystem):
        container = get_container(target_system, self.env)
        if container is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                f"The target particle waiting for cooked should be contained in a container",
                {"target particle": target_system.name},
            )

        check_heat_source_before_cook(container, self.env)

        system_name = target_system.name
        cooked_system_name = f'cooked__{system_name}'
        cooked_system = get_cooked_system(cooked_system_name, self.env)
        if cooked_system is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                f"Cannot cook target particle.",
                {"target particle": target_system.name},
            )

        contained_particles = container.states[object_states.ContainedParticles].get_value(target_system)
        in_volume_idx = torch.where(contained_particles.in_volume)[0]
        if len(in_volume_idx) < 1:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                f"Container does not contain any target particles.",
                {"target particle": target_system.name, "container": container.name},
            )
        
        # Remove uncooked particles
        target_system.remove_particles(idxs=in_volume_idx)

        # Generate cooked particles
        particle_positions = contained_particles.positions[in_volume_idx]
        cooked_system.generate_particles(positions=particle_positions)

        yield from self._settle_robot()

        cooked_system = find_task_related_object(
            self.env, cooked_system_name, retain_wrapper=True
        )
        if not cooked_system.exists:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                f"Fail to add cooked particles.",
                {"target system": target_system.name, "cooked system": cooked_system_name},
            )

    def _cook_non_particle_object(self, target_obj: StatefulObject):
        heating_temperature = -1.0
        if object_states.Cooked in target_obj.states:
            heating_temperature = max(
                heating_temperature, target_obj.states[object_states.Cooked].cook_temperature
            )
        if object_states.Heated in target_obj.states:
            heating_temperature = max(
                heating_temperature, target_obj.states[object_states.Heated].heat_temperature
            )

        # Check that the current object is cookable or heatable
        if heating_temperature < 0:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Target object is not cookable or heatable.",
                {"target object": target_obj.name},
            )
        heating_temperature = heating_temperature + 15

        check_heat_source_before_cook(target_obj, self.env)

        target_obj.states[object_states.Temperature].set_value(heating_temperature)
        target_obj.states[object_states.MaxTemperature].set_value(heating_temperature)
        yield from self._settle_robot()

        if not ((object_states.Cooked in target_obj.states and target_obj.states[object_states.Cooked].get_value()) \
              or (object_states.Heated in target_obj.states and target_obj.states[object_states.Heated].get_value())):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Target object is not cooked and heated.",
                {"target object": target_obj.name},
            )

    def _wait_for_cooked(self, target_obj: StatefulObject | BaseSystem):
        if isinstance(target_obj, BaseSystem):
            yield from self._cook_particle_system(target_obj)
        else:
            # check if cooked proberty
            if object_states.Cooked in target_obj.states:
                yield from self._cook_non_particle_object(target_obj)
            else: 
                # target_obj is a cook_tool
                contained_systems = get_contained_systems(target_obj)
                if contained_systems:
                    # cook the fluid system inside the container
                    for particle_system in contained_systems:
                        if hasattr(particle_system, 'is_fluid') and particle_system.is_fluid:
                            yield from self._cook_particle_system(particle_system)

                placement_objects = get_placement_objects(target_obj, self.env)
                if placement_objects:
                    for placement_object in placement_objects:
                        if object_states.Cooked in placement_object.states:
                            yield from self._cook_non_particle_object(placement_object)

    def _wait_for_washed(self, wash_machine: StatefulObject):
        if not (wash_machine.name.split("_")[0] == 'washer' or wash_machine.name.split("_")[0] == 'dishwasher'):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Wait_for_washed api must input with an wach_machine named 'washer'.",
                {"target object": wash_machine.name},
            )
        
        if wash_machine.states[object_states.Open].get_value():
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The washer is still open on so you are not able to toggled on to wash",
                {"target object": wash_machine.name},
            )

        if not wash_machine.states[object_states.ToggledOn].get_value():
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The washer is not toggled on so you are not able to wash",
                {"target object": wash_machine.name},
            )
        
        inside_placements = get_placement_objects(wash_machine, self.env, object_states.Inside)
        if not inside_placements:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "There are no objects inside the wash machine, so there is no need to wash",
                {"target object": wash_machine.name},
            )
     
        for inside_placement in inside_placements:
            inside_obj = inside_placement.object
            covered_systems = get_covered_systems(inside_obj)
            if covered_systems is None:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "The target object inside the washer is not coverable by any particles, so there is no need to wipe it.",
                    {"target object": inside_obj.name},
                )
            if not covered_systems:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "The target object inside the washer is not covered by any particles, so there is no need to wipe it.",
                    {"target object": inside_obj.name},
                )
            for system in covered_systems:
                inside_obj.states[object_states.Covered].set_value(system, False)
                yield from self._settle_robot()

    def _wait(self, target_obj: StatefulObject):
        if hasattr(target_obj, 'states'):
            if object_states.Frozen in target_obj.states:
                target_obj.states[object_states.Frozen].set_value(False)
            if object_states.Heated in target_obj.states:
                target_obj.states[object_states.Heated].set_value(False)

        yield from self._settle_robot()
        
    def _wait_for_frozen(self, target_obj: StatefulObject, refrigerator_obj: StatefulObject):
        if not 'fridge' in refrigerator_obj.name:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The second arguments should be refriderator",
                {"refrigerator_obj": refrigerator_obj.name},
            )
            
        if hasattr(target_obj, 'states') and object_states.Frozen in target_obj.states:
            if target_obj.states[object_states.Inside].get_value(refrigerator_obj):
                target_obj.states[object_states.Frozen].set_value(True)
                yield from self._settle_robot()
            else: 
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "The target object is not inside the refrigerator",
                    {"target_obj": target_obj.name},
                )
        else:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object have no frozen states to make if frozen",
                {"target_obj": target_obj.name},
            )
                        
