"""
Goal:
The goal of this python file is to test the connection between a physical robot.

The connection is successful if:
- The robot connects
- A video feed opens in a new window
- A servo connected to a robots pin actuates
"""

# === Program variables ========================================================
hostname="10.15.3.103"
enable_video_feed = True
SERVO_PIN = 13
# ==============================================================================

from dataclasses import dataclass

from pyrr import Vector3

from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.body.sensors import CameraSensor
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot import ModularRobot, ModularRobotControlInterface
from revolve2.modular_robot_physical import Config, UUIDKey
from revolve2.modular_robot_physical.remote import run_remote

from math import cos

# === Build body ===============================================================
def body_representation() -> tuple[BodyV2, ActiveHingeV2]:
    body = BodyV2()
    body.core_v2.back_face.bottom = ActiveHingeV2(0.0)
    body.core.add_sensor(
        CameraSensor(
            position=Vector3([0, 0, 0]), 
            camera_size=(480, 640)
        )
    )
    return (body, body.core_v2.back_face.bottom)
# ==============================================================================

# === Build brain ==============================================================
@dataclass
class RemoteBrain(Brain):
    joint: ActiveHingeV2

    def make_instance(self) -> BrainInstance:
        return RemoteBrainInstance(joint=self.joint)

@dataclass
class RemoteBrainInstance(BrainInstance): 
    joint: ActiveHingeV2
    _sim_time = 0.0

    def control(
            self, 
            dt: float, 
            sensor_state: ModularRobotSensorState, 
            control_interface: ModularRobotControlInterface
        ):
        self._sim_time += dt
        
        right_reflect = 2 * cos(self._sim_time)
        control_interface.set_active_hinge_target(self.joint, right_reflect)
# ==============================================================================

def on_prepared() -> None:
    print("Robot is ready. Press enter to start the brain.")
    input()

def main() -> None:
    body, joint = body_representation()

    brain = RemoteBrain(joint)
    robot = ModularRobot(body, brain)
    
    config = Config(
        modular_robot=robot,
        hinge_mapping={UUIDKey(joint): SERVO_PIN},
        run_duration=30,
        control_frequency=20,
        initial_hinge_positions={UUIDKey(joint): 1},
        inverse_servos={},
    )

    print("Initializing robot..")
    run_remote(
        config=config,
        hostname=hostname,
        debug=True,
        on_prepared=on_prepared,
        display_camera_view=enable_video_feed
    )
if __name__ == "__main__": main()