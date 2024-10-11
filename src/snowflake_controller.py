"""
Goal:
The goal of this python file is to provide a brain for snowflake.py. This is 
just a test example for playing around with the revolve2 API and is not 
intended to be used other than as an example.
"""

from revolve2.modular_robot.body.v2 import ActiveHingeV2
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot import ModularRobotControlInterface
from math import cos

class SnowflakeBrain(Brain):

    left_arm_joint: ActiveHingeV2
    right_arm_joint: ActiveHingeV2
    left_hip_joint: ActiveHingeV2
    right_hip_joint: ActiveHingeV2
    
    actuate_100: list[ActiveHingeV2]

    def __init__(
        self,    
        left_arm_joint: ActiveHingeV2,
        right_arm_joint: ActiveHingeV2,
        left_hip_joint: ActiveHingeV2,
        right_hip_joint: ActiveHingeV2,
        actuate_100: list[ActiveHingeV2]
    ) -> None:
        self.left_arm_joint = left_arm_joint
        self.right_arm_joint = right_arm_joint
        self.left_hip_joint = left_hip_joint
        self.right_hip_joint = right_hip_joint
        self.actuate_100 = actuate_100

    def make_instance(self) -> BrainInstance:
        return SnowflakeBrainInstance(
            left_arm_joint=self.left_arm_joint,
            left_hip_joint=self.left_hip_joint,
            right_arm_joint=self.right_arm_joint,
            right_hip_joint=self.right_hip_joint,
            actuate_100 = self.actuate_100
        )

class SnowflakeBrainInstance(BrainInstance): 
    left_arm_joint: ActiveHingeV2
    right_arm_joint: ActiveHingeV2
    left_hip_joint: ActiveHingeV2
    right_hip_joint: ActiveHingeV2
    
    actuate_100: list[ActiveHingeV2]

    _sim_time = 0.0
    
    def __init__(self,
            left_arm_joint: ActiveHingeV2,
            right_arm_joint: ActiveHingeV2,
            left_hip_joint: ActiveHingeV2,
            right_hip_joint: ActiveHingeV2,
            actuate_100: list[ActiveHingeV2]
        ):
            self.left_arm_joint = left_arm_joint
            self.right_arm_joint = right_arm_joint
            self.left_hip_joint = left_hip_joint
            self.right_hip_joint = right_hip_joint
            self.actuate_100 = actuate_100
        
    def control(self, dt: float, sensor_state: ModularRobotSensorState, control_interface: ModularRobotControlInterface):
        self._sim_time += dt
    
        for joint in self.actuate_100:
            control_interface.set_active_hinge_target(joint, 1)
        
        right_reflect = 2 * cos(self._sim_time)
        control_interface.set_active_hinge_target(self.left_arm_joint, right_reflect)
        control_interface.set_active_hinge_target(self.left_hip_joint, right_reflect)
        
        control_interface.set_active_hinge_target(self.right_arm_joint, -right_reflect)
        control_interface.set_active_hinge_target(self.right_hip_joint, -right_reflect)