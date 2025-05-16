import typing
from collections import namedtuple

import numpy as np
import numpy.typing as npt

from revolve2.modular_robot_simulation import SceneSimulationState

objective_type = typing.Literal["DTW", "MSE", "All_Angles", "4_Angles", "2_Angles", "Distance"]

solution = npt.NDArray[np.float_]
behavior = list[SceneSimulationState]

EAState = namedtuple('EAState', 
                     ['generation', 'run', 'alpha', 
                       'animal_data'])

EAConfig = namedtuple('EAConfig',
                      ['ttl', 'freq'])