"""
Goal:
The goal of this python file is to test the connection between a physical robot.

The connection is successful if:
- The robot connects
- A video feed opens in a new window
- A servo connected to a robots pin actuates
"""

# === Program variables ========================================================
hostname="10.15.3.59"
enable_video_feed = True
SERVO_PIN = 31
# ==============================================================================

from dataclasses import dataclass

from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.body.sensors import CameraSensor
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot import ModularRobot, ModularRobotControlInterface
from revolve2.modular_robot_physical import Config, UUIDKey
from revolve2.modular_robot_physical.remote import run_remote

from revolve2.standards.modular_robots_v2 import gecko_v2

from experiment.revolve_layer import remote_control_with_polling_rate

from math import cos

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

def on_prepared() -> None:
    print("Robot is ready. Press enter to start the brain.")
    input()

def main() -> None:
    body = gecko_v2()
    joint = body.core_v2.right_face.bottom

    brain = RemoteBrain(joint)
    robot = ModularRobot(body, brain)
    
    config = Config(
        modular_robot=robot,
        hinge_mapping={UUIDKey(joint): SERVO_PIN},
        run_duration=30,
        control_frequency=30,
        initial_hinge_positions={UUIDKey(joint): 0},
        inverse_servos={},
    )

    print("Initializing robot..")
    remote_control_with_polling_rate(
        config=config,
        hostname=hostname,
        port=20812,
        rate=5
    )

    # run_remote(
    #     config=config,
    #     hostname=hostname,
    #     debug=False,
    #     on_prepared=on_prepared,
    #     # display_camera_view=enable_video_feed
    # )
if __name__ == "__main__": main()