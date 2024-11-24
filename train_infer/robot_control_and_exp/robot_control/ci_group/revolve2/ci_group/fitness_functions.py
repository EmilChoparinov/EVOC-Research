"""Standard fitness functions for modular robots."""

import math

from revolve2.modular_robot_simulation import ModularRobotSimulationState


def xy_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the distance traveled on the xy-plane by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position
    # print("============begin_position.z:", begin_position.z)

    # print("============begin_position:", begin_position)
    # print("============end_position:", end_position)

    return math.sqrt(
        (begin_position.x - end_position.x) ** 2
        + (begin_position.y - end_position.y) ** 2
    )

def max_z_displacement(
    state_list: list[ModularRobotSimulationState]
) -> float:
    """
    Calculate the maximum z displacement of a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    max_z = 0
    # import pdb; pdb.set_trace()
    for state in state_list:
        position = state.get_pose().position
        if position.z > max_z:
            max_z = position.z
    return max_z
