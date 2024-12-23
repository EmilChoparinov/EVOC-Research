import numpy as np
from revolve2.modular_robot.body import Module
from revolve2.simulation.scene import MultiBodySystem, Pose, SimulationState
from revolve2.simulation.scene import MultiBodySystem, SimulationState, UUIDKey
from revolve2.modular_robot import ModularRobot
from ._build_multi_body_systems import BodyToMultiBodySystemMapping
from pyrr import Vector3, Quaternion

class ModularRobotSimulationState:
    """The state of a modular robot at some moment in a simulation."""

    _simulation_state: SimulationState
    _multi_body_system: MultiBodySystem
    _modular_robot_to_module_map: BodyToMultiBodySystemMapping
    """The multi-body system corresponding to the modular robot."""

    def __init__(
        self, simulation_state: SimulationState, multi_body_system: MultiBodySystem,
        modular_robot_to_module_map: dict[UUIDKey[ModularRobot], BodyToMultiBodySystemMapping]
    ) -> None:
        """
        Initialize this object.

        :param simulation_state: The simulation state corresponding to this modular robot state.
        :param multi_body_system: The multi-body system this modular robot corresponds to.
        """
        self._simulation_state = simulation_state
        self._multi_body_system = multi_body_system
        self._modular_robot_to_module_map = modular_robot_to_module_map        

    def get_pose(self) -> Pose:
        """
        Get the pose of the modular robot.

        :returns: The retrieved pose.
        """
        return self._simulation_state.get_multi_body_system_pose(
            self._multi_body_system
        )

    def get_module_relative_pose(self, module: Module) -> Pose:
        """
        Get the pose of a module, relative to its parent module's reference frame.

        In case there is no parent(the core), this is equal to getting the absolute pose.

        :param module: The module to get the pose for.
        :returns: The relative pose.
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError()
        return Pose()

    def get_module_absolute_pose(self, module: Module) -> Pose:
        """
        Get the pose of this module, relative the global reference frame.

        :param module: The module to get the pose for.
        :returns: The absolute pose.
        :raises NotImplementedError: Always.
        """
        # This selection is ORDERED such that it matches the _xpos array
        geometry_selection =  [
            self._simulation_state._geometries[geo_name]
            for geo_name in
            [ 
            "mbs1/mbs1_geom0", # ORIGIN(?) can be anything that runs
            "mbs1/mbs1_geom0", # ORIGIN(?) can be anything that runs
            "mbs1/mbs1_geom0", # (2) Head
            "mbs1/mbs1_link0_geom2", # (3) Middle
            "mbs1/mbs1_link0_link1_geom2", # (4) Rear
            "mbs1/mbs1_link0_link1_link2_geom2", # (5) Right Hind
            "mbs1/mbs1_link0_link1_link1_geom2",  # (6) Left Hind 
            "mbs1/mbs1_link1_geom2", # (7) Left Front
            "mbs1/mbs1_link2_geom2", # (8) Right Front
        ]]


        # Encode the corrected rotated coordinates in a matrix. Suppose z,y 
        # are flipped for LEFT HIND, then array would look like [0,2,1]
        correction_matrix = np.array([
            [0,1,2], # ORIGIN
            [0,1,2], # ORIGIN
            [0,1,2], # HEAD
            [0,1,2], # MIDDLE
            [0,1,2], # RIGHT FRONT
            [0,1,2], # LEFT FRONT
            [0,1,2], # REAR
            [0,1,2], # RIGHT HIND (z -> x), (x -> y)
            [0,1,2], # LEFT HIND (z -> x), (x -> y)
        ])

        true_xpos = np.array(
            [point[correction] 
            for point, correction in 
            zip(geometry_selection, correction_matrix)])

        rigid_body = self._modular_robot_to_module_map.module_to_rigid_body[UUIDKey(module)]
        
        # HACK!! this comes out of an abstract thats implemented. Instead of
        #        if-checking if the property exists, I left this intentionally
        #        to blow up so YOU can fix it.
        rigid_body_idx = self._simulation_state._abstraction_to_mujoco_mapping.rigid_body[UUIDKey(rigid_body)]


        return Pose(
            Vector3(true_xpos[rigid_body_idx.id]),
            Quaternion(self._simulation_state._xquat[rigid_body_idx.id])
        )