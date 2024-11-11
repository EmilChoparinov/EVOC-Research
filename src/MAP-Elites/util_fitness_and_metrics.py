from revolve2.standards.fitness_functions import xy_displacement
from util_get_metrics import get_metrics

def calculate_fitness_and_metrics(robots, behaviors):
    fitness_and_metrics = []
    for i in range(len(robots)):
        fitness_and_metrics.append(calculate(robots[i], behaviors[i]))

    return fitness_and_metrics

def get_fitness_and_positions(robot, behavior):
    modules = [robot.body.core_v2,  # head
               robot.body.core_v2.left_face.bottom.attachment,  # left_front
               robot.body.core_v2.right_face.bottom.attachment,  # right_front
               robot.body.core_v2.back_face.bottom.attachment,  # middle
               robot.body.core_v2.back_face.bottom.attachment.front.attachment,  # rear
               robot.body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment,  # left_hind
               robot.body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment,  # right_hind
               ]
    fitness = xy_displacement(behavior[0].get_modular_robot_simulation_state(robot),
                              behavior[-1].get_modular_robot_simulation_state(robot))

    head_positions = []
    left_front_positions = [];
    right_front_positions = [];
    middle_positions = []
    rear_positions = [];
    left_hind_positions = [];
    right_hind_positions = []
    for i in range(len(behavior)):
        element = behavior[i].get_modular_robot_simulation_state(robot)

        # Head
        xyz_position = element.get_module_absolute_pose(modules[0]).position
        head_positions.append((xyz_position[0], xyz_position[1]))

        # Left Front
        xyz_position = element.get_module_absolute_pose(modules[1]).position
        left_front_positions.append((xyz_position[0], xyz_position[1]))

        # Right Front
        xyz_position = element.get_module_absolute_pose(modules[2]).position
        right_front_positions.append((xyz_position[0], xyz_position[1]))

        # Middle
        xyz_position = element.get_module_absolute_pose(modules[3]).position
        middle_positions.append((xyz_position[0], xyz_position[1]))

        # Rear
        xyz_position = element.get_module_absolute_pose(modules[4]).position
        rear_positions.append((xyz_position[0], xyz_position[1]))

        # Left Hind
        xyz_position = element.get_module_absolute_pose(modules[5]).position
        left_hind_positions.append((xyz_position[0], xyz_position[1]))

        # Right Hind
        xyz_position = element.get_module_absolute_pose(modules[6]).position
        right_hind_positions.append((xyz_position[0], xyz_position[1]))

    return (fitness, head_positions, left_front_positions, right_front_positions, middle_positions,
                          rear_positions, left_hind_positions, right_hind_positions)

def calculate(robot, behavior):
    (fitness, head_positions, left_front_positions, right_front_positions, middle_positions,
    rear_positions, left_hind_positions, right_hind_positions) = get_fitness_and_positions(robot, behavior)
    metrics = get_metrics(head_positions, left_front_positions, right_front_positions, middle_positions,
                          rear_positions, left_hind_positions, right_hind_positions)

    return (fitness,) + metrics