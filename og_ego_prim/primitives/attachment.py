from abc import ABC, abstractmethod
from collections import namedtuple
import math

from omnigibson import object_states
from omnigibson.objects import StatefulObject
from omnigibson.object_states.object_state_base import RelativeObjectState
from omnigibson.systems.micro_particle_system import FluidSystem
from omnigibson.systems.system_base import BaseSystem
import torch

Placement = namedtuple("Placement", ("object", "predicate"))


class Attachment(ABC):

    def __init__(self, obj: StatefulObject):
        self.obj = obj
        self.obj_pose = obj.get_position_orientation()
    
    @abstractmethod
    def remove_attachment(self):
        pass

    @abstractmethod
    def recover_attachment(self):
        pass
    
    @property
    def is_fluid(self):
        return False

    @property
    def name(self):
        pass


class FluidAttachment(Attachment):

    def __init__(self, obj: StatefulObject, system: FluidSystem):
        super().__init__(obj)

        self.system = system
        self.particle_instancer = self.system.default_particle_instancer

        contained_particles = obj.states[object_states.ContainedParticles].get_value(self.system)
        _, _, in_volume = contained_particles
        indices = in_volume.nonzero().squeeze()

        self.attached_particle_positions = self.particle_instancer.particle_positions[indices].clone()
        self.attached_particle_velocities = self.particle_instancer.particle_velocities[indices].clone()
        self.attached_particle_orientations = self.particle_instancer.particle_orientations[indices].clone()
        self.attached_particle_scales = self.particle_instancer.particle_scales[indices].clone()
        self.attached_particle_prototype_ids = self.particle_instancer.particle_prototype_ids[indices].clone()

        self.indices = indices

    def remove_attachment(self):
        if self.indices is not None:
            self.particle_instancer.remove_particles(self.indices)
            self.indices = None
        
    def recover_attachment(self):
        obj_pos, obj_quat = self.obj_pose
        diff_pos = self.attached_particle_positions - obj_pos
        diff_quat = self.attached_particle_orientations - obj_quat
        
        new_obj_pose = self.obj.get_position_orientation()
        new_obj_pos, new_obj_quat = new_obj_pose
        new_particle_positions = diff_pos + new_obj_pos
        new_particle_orientation = diff_quat + new_obj_quat

        self.particle_instancer.add_particles(
            positions=new_particle_positions,
            velocities=self.attached_particle_velocities,
            orientations=new_particle_orientation,
            scales=self.attached_particle_scales,
            prototype_indices=self.attached_particle_prototype_ids,
        )
    
    @property
    def is_fluid(self):
        return True
    
    @property
    def name(self):
        return self.system.name


class StainAttachment(Attachment):
    
    def __init__(self, obj: StatefulObject, system: BaseSystem):
        super().__init__(obj)
        
        self.system = system
        self.particle_template = self.system._particle_template
        self.particles = self.system.particles
        self.particle_counter = 0

        self.old_pos, self.old_quat = self.system.get_particles_local_pose()

    def remove_attachment(self):
        for _, value in self.particles: 
            value.visible = False

    def recover_attachment(self):
        obj_pos, obj_quat = self.obj_pose

        diff_pos = self.old_pos - obj_pos
        diff_quat = self.old_quat - obj_quat

        new_obj_pos, new_obj_quat = self.obj.get_position_orientation()

        new_particle_pos = diff_pos + new_obj_pos
        new_particle_quat = diff_quat + diff_pos

        self.system.set_particles_local_pose(new_particle_pos, new_particle_quat)

        for _, value in self.particles: 
            value.visible = True


class PlacementAttachment(Attachment):

    def __init__(self, obj: StatefulObject, placement: Placement):
        super().__init__(obj)

        self.placed_obj: StatefulObject = placement.object
        self.placed_predicate: RelativeObjectState = placement.predicate

        self.placed_obj_pose = self.placed_obj.get_position_orientation()

    def remove_attachment(self):
        self.placed_obj.visible = False

    def recover_attachment(self):
        obj_pos, obj_quat = self.obj_pose
        placed_obj_pos, placed_obj_quat = self.placed_obj_pose
        diff_pos = placed_obj_pos - obj_pos
        diff_quat = placed_obj_quat - obj_quat

        new_obj_pose = self.obj.get_position_orientation()
        new_obj_pos, new_obj_quat = new_obj_pose
        new_placed_obj_pos = diff_pos + new_obj_pos
        new_placed_obj_quat = diff_quat + new_obj_quat

        # keep new quat as a unit vector
        unit_quat = torch.norm(new_placed_obj_quat)
        if not math.isclose(unit_quat.item(), 1, abs_tol=1e-3):
            new_placed_obj_quat = new_placed_obj_quat / unit_quat

        self.placed_obj.set_position_orientation(
            position=new_placed_obj_pos,
            orientation=new_placed_obj_quat,
        )
        self.placed_obj.keep_still()

        self.placed_obj.visible = True
    
    @property
    def name(self):
        return self.placed_obj.name
