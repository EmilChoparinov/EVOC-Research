from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import SceneSimulationState

import numpy as np
import numpy.typing as npt

from typing import Tuple, List


genotype = npt.NDArray[np.float_]
population = List[ModularRobot]
simulated_behavior = list[SceneSimulationState]