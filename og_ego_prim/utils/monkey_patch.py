import sys

import omnigibson as og
from omnigibson.systems.micro_particle_system import PhysxParticleInstancer
from omnigibson.utils.physx_utils import create_physx_particleset_pointinstancer
from omnigibson.utils.python_utils import torch_delete
from omnigibson.utils.usd_utils import absolute_prim_path_to_scene_relative
import torch


# fix bug, skip converting List[str] to tensor
def patched__python_utils__recursively_convert_to_torch(state):
    # For all the lists in state dict, convert to torch tensor
    for key, value in state.items():
        if isinstance(value, dict):
            state[key] = patched__python_utils__recursively_convert_to_torch(value)
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], str):
                state[key] = value
            else:
                state[key] = torch.tensor(value, dtype=torch.float32)
    return state


# fix bug, int("0.0")
def patched__MicroPhysicalParticleSystem__particle_instancer_name_to_idn(self, name):
    value = name.split(f"{self.name}Instancer")[-1]
    if '.' in value:
        value = float(value)
    return int(value)


# fix bug, covert n_particles/particle_group (converted into tensor in recursively_convert_to_torch) into int
def patched__MicroPhysicalParticleSystem__generate_particle_instancer(
    self,
    n_particles,
    idn=None,
    particle_group=0,
    positions=None,
    velocities=None,
    orientations=None,
    scales=None,
    prototype_indices=None,
):
    # Run sanity checks
    assert self.initialized, "Must initialize system before generating particle instancers!"

    # Multiple particle instancers is NOT supported currently, since there is no clear use case for multiple
    assert self.n_instancers == 0, (
        f"Cannot create multiple instancers for the same system! "
        f"There is already {self.n_instancers} pre-existing instancers."
    )

    # Automatically generate an identification number for this instancer if none is specified
    if idn is None:
        idn = self.next_available_instancer_idn

    assert idn not in self.instancer_idns, f"instancer idn {idn} already exists."

    # Generate standardized prim path for this instancer
    name = self.particle_instancer_idn_to_name(idn=idn)

    if isinstance(n_particles, torch.Tensor):
        n_particles = int(n_particles.item())
    if isinstance(particle_group, torch.Tensor):
        particle_group = int(particle_group.item())

    # Create the instancer
    instance = create_physx_particleset_pointinstancer(
        name=name,
        particle_system_path=self.prim_path,
        physx_particle_system_path=self.system_prim_path,
        particle_group=particle_group,
        positions=torch.zeros((n_particles, 3)) if positions is None else positions,
        self_collision=self.self_collision,
        fluid=self.is_fluid,
        particle_mass=None,
        particle_density=self.particle_density,
        orientations=orientations,
        velocities=velocities,
        angular_velocities=None,
        scales=self.sample_scales(n=n_particles) if scales is None else scales,
        prototype_prim_paths=[pp.prim_path for pp in self.particle_prototypes],
        prototype_indices=prototype_indices,
        enabled=not self.visual_only,
    )

    # Create the instancer object that wraps the raw prim
    instancer = PhysxParticleInstancer(
        relative_prim_path=absolute_prim_path_to_scene_relative(self.scene, instance.GetPrimPath().pathString),
        name=name,
        idn=idn,
    )
    instancer.load(self.scene)
    instancer.initialize()
    self.particle_instancers[name] = instancer

    return instancer


# fix bug, potential mismatched object uuid between state_info and scene.object_registry
def patched__MacroVisualParticleSystem__load_state(self, state):
    group_objects = []
    particle_idns = []
    particle_attached_references = []

    def _get_object(attached_obj_name, info):
        obj = self.scene.object_registry("uuid", info["particle_attached_obj_uuid"])
        if obj is not None:
            return obj
        
        objects_info = self.scene._objects_info['init_info']
        if attached_obj_name in objects_info:
            attached_obj_info = objects_info[attached_obj_name]
            attached_obj_uuid = attached_obj_info['args']['uuid']
            if attached_obj_uuid != info["particle_attached_obj_uuid"]:
                obj = self.scene.object_registry("uuid", attached_obj_uuid)
        
        if obj is not None:
            og.systems.macro_particle_system.log.warning(
                "Mismatched uuid between objects_info and system_info, using uuid in objects_info instead."
            )
            return obj

        if attached_obj_name in self.scene.object_registry.object_names:
            obj = self.scene.object_registry("name", attached_obj_name)
        
        if obj is not None:
            og.systems.macro_particle_system.log.warning(
                "Cannot find object in object_registry using uuid, using name instead."
            )
            return obj
        
        return None

    indices_to_remove = torch.empty(0, dtype=int)
    for attached_obj_name, info in state["groups"].items():
        obj = _get_object(attached_obj_name, info)

        # obj will be None if an object with an attachment group is removed between dump_state() and load_state()
        if obj is not None:
            group_objects.append(obj)
            particle_idns.append(info["particle_idns"])
            particle_attached_references.append(info["particle_attached_references"])
        else:
            indices_to_remove = torch.cat((indices_to_remove, torch.tensor(info["particle_indices"], dtype=int)))
    self._sync_particle_groups(
        group_objects=group_objects,
        particle_idns=particle_idns,
        particle_attached_references=particle_attached_references,
    )
    state["n_particles"] -= len(indices_to_remove)
    state["positions"] = torch_delete(state["positions"], indices_to_remove, dim=0)
    state["orientations"] = torch_delete(state["orientations"], indices_to_remove, dim=0)
    state["scales"] = torch_delete(state["scales"], indices_to_remove, dim=0)

    # Run super
    super(og.systems.macro_particle_system.MacroVisualParticleSystem, self)._load_state(state=state)


# fix bug, all(conditions) -> any(valid_conditions) & all(limit_conditions) & all(nonempty_conditions)
def patched__ParticleModifier__check_conditions_for_system(self, system_name):
    if not self.supports_system(system_name):
        return False
    
    valid_conditions, limit_conditions, nonempty_conditions = [], [], []
    for condition in self.conditions[system_name]:
        if '._generate_condition.' in condition.__qualname__:
            valid_conditions.append(condition)
        elif '._generate_limit_condition.' in condition.__qualname__:
            limit_conditions.append(condition)
        elif '._generate_nonempty_system_condition.' in condition.__qualname__:
            nonempty_conditions.append(condition)
        else:
            raise ValueError(f'invalid condition type {condition.__qualname__}')

    valid_check = any(condition(self.obj) for condition in valid_conditions) if valid_conditions else True
    limit_check = all(condition(self.obj) for condition in limit_conditions) if limit_conditions else True
    nonempty_check = all(condition(self.obj) for condition in nonempty_conditions) if nonempty_conditions else True
    return valid_check & limit_check & nonempty_check


def add_monkey_patch():
    og.utils.python_utils.recursively_convert_to_torch = patched__python_utils__recursively_convert_to_torch
    og.object_states.particle_modifier.ParticleModifier.check_conditions_for_system = patched__ParticleModifier__check_conditions_for_system
    og.systems.micro_particle_system.MicroPhysicalParticleSystem.particle_instancer_name_to_idn = patched__MicroPhysicalParticleSystem__particle_instancer_name_to_idn
    og.systems.micro_particle_system.MicroPhysicalParticleSystem.generate_particle_instancer = patched__MicroPhysicalParticleSystem__generate_particle_instancer
    og.systems.macro_particle_system.MacroVisualParticleSystem._load_state = patched__MacroVisualParticleSystem__load_state
    
    patched_funcs = [var for var in globals() if var.startswith('patched__')]
    print(f'patched omnigibson: {patched_funcs}')
    sys.stdout.flush()
