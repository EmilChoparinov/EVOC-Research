"""
Goal:
Provide an example simulation of a customized robot performing a "snowflake" or
"swimming" motion.
"""
from snowflake_controller import SnowflakeBrain

from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.standards import terrains
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.modular_robot import ModularRobot

# === Build robot =============================================================
rng = make_rng_time_seed()

robot_root = BodyV2()
robot_head = robot_root.core_v2

robot_head.back_face.bottom = ActiveHingeV2(0.0)
robot_head.back_face.bottom.attachment = BrickV2(0.0)
robot_body = robot_head.back_face.bottom.attachment

robot_body.left = ActiveHingeV2(RightAngles.RAD_HALFPI)      # left arm
robot_body.left.attachment = BrickV2(0.0) # left hand

robot_body.right = ActiveHingeV2(RightAngles.RAD_HALFPI) # right arm
robot_body.right.attachment = BrickV2(0.0) # right hand

robot_body.front = ActiveHingeV2(0.0) # add torso to body
robot_body.front.attachment = BrickV2(0.0)
robot_torso = robot_body.front.attachment

robot_torso.left = ActiveHingeV2(RightAngles.RAD_HALFPI) # left leg
# left knee joint
robot_torso.left.attachment = BrickV2(RightAngles.DEG_270)
robot_torso.left.attachment.right = ActiveHingeV2(RightAngles.RAD_HALFPI) 
robot_torso.left.attachment.right.attachment = BrickV2(0.0) # left foot


robot_torso.right = ActiveHingeV2(RightAngles.RAD_HALFPI) # left leg
# left knee joint
robot_torso.right.attachment = BrickV2(RightAngles.DEG_270)
robot_torso.right.attachment.left = ActiveHingeV2(RightAngles.RAD_ONEANDAHALFPI)
robot_torso.right.attachment.left.attachment = BrickV2(0.0) # left foot
# ==============================================================================

robot_brain = SnowflakeBrain(
    left_arm_joint=robot_body.left,
    right_arm_joint=robot_body.right,
    left_hip_joint=robot_torso.left,
    right_hip_joint=robot_torso.right,
    actuate_100=[robot_torso.right.attachment.left, robot_torso.left.attachment.right]
)

robot = ModularRobot(robot_root, robot_brain)

def main() -> None:
    setup_logging()
    scene = ModularRobotScene(terrain=terrains.flat())
    scene.add_robot(robot)
    
    simulate_scenes(
        simulator=LocalSimulator(viewer_type="native"),
        batch_parameters=make_standard_batch_parameters(simulation_time=60),
        scenes=scene
    )

if __name__ == "__main__": main()