import typing
from collections import namedtuple

import numpy as np
import numpy.typing as npt

from revolve2.modular_robot_simulation import SceneSimulationState

similarity_type = typing.Literal["DTW", "MSE", "Angles", "All_Angles", "4_Angles", "2_Angles", "distance"]

solution = npt.NDArray[np.float_]
behavior = list[SceneSimulationState]

EAState = namedtuple('EAState', 
                     ['generation', 'run', 'alpha', 
                      'similarity_type', 'animal_data'])

EAConfig = namedtuple('EAConfig',
                      ['ttl', 'freq'])