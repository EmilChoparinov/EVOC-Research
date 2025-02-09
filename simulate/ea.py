from collections import namedtuple
from revolve2.standards import terrains
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, CpgNetworkStructure
import math
from revolve2.experimentation.rng import seed_from_time
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import active_hinges_to_cpg_network_structure_neighbor
from revolve2.standards.modular_robots_v2 import gecko_v2
from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

import simulate.stypes as stypes
import simulate.evaluate as evaluate
import pandas as pd
import simulate.data as data
import logging
import cma

# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /                 EA UTILITIES                 \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
def file_idempotent(state: stypes.EAState) -> str:
    return f"run-{state.run}-alpha-{state.alpha}-type-{state.similarity_type}.csv"


# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /           TUPLE CONSTRUCTORS                 \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
def create_state(
        generation: int, run: int, alpha: float,
        similarity_type: stypes.similarity_type, animal_data: pd.DataFrame):
    return stypes.EAState(
        generation=generation, run=run, alpha=alpha,
        similarity_type=similarity_type, animal_data=animal_data)

def create_config(
        ttl: int, freq: int):
    return stypes.EAConfig(ttl=ttl, freq=freq)

# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /                  EA                          \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
def iterate(state: stypes.EAState, config: stypes.EAConfig):
    
    # Base case: no more runs needed
    if state.run == 0: return

    # Stack `runs` total of this function
    iterate(state._replace(run=state.run - 1), config)

    # Define the shape we want to iterate on
    body_shape = gecko_v2()

    # Using the body, extract the generated CPG structure from Revolve alongside
    # the mapping between hinge objects -> neurons
    cpg_struct, mapping = active_hinges_to_cpg_network_structure_neighbor(
        body_shape.find_modules_of_type(ActiveHinge))
    
    # Create an evolution strategy (CMA-ES)
    cma_es_options = cma.CMAOptions()
    cma_es_options.set("bounds", [-2.5, 2.5])
    cma_es_options.set("seed", seed_from_time() % 2 ** 32)

    cma_es = cma.CMAEvolutionStrategy(
        cpg_struct.num_connections * [0.0], # initial CPG weights (all zero)
        0.5, # initial STD
        cma_es_options)

    logging.info(f"[iterate] starting run {state.run}")

    # Perform one generation
    for gen_i in range(state.generation):
        logging.info(
            f"[iterate] Performing generation {gen_i + 1} / {state.generation}")
        
        # Collect this iterations solutions to test
        solutions: list[stypes.solution] = cma_es.ask()

        # Send solution vector to simulation function and get the robot objects
        # alongside the simulation result
        robots, behaviors = simulate_solutions(solutions, 
                                               cpg_struct,
                                               body_shape, mapping,
                                               config)
        
        # Convert behavior data into dataframes to make further processing
        # easier
        df_behaviors = data.behaviors_to_dataframes(robots, behaviors, state)
        
        # Evaluate and tell CMA-ES the scores of each solution
        scores = evaluate.evaluate(df_behaviors, state)
        cma_es.tell(solutions, scores)

        # Select the best score this iteration and append the data used to get
        # the result into the dataframe
        best_score, best_df_behavior = evaluate\
            .most_fit(scores, df_behaviors)
        data.apply_statistics(best_df_behavior, best_score, state, gen_i)

        # Export the best this generation
        logging.info(
            f"[iterate] {gen_i} / {state.generation}\n{cma_es.result.xbest=}\n{cma_es.result.fbest=}")
        if gen_i == 0:
            best_df_behavior.to_csv(file_idempotent(state), index=False)
            continue
        best_df_behavior.to_csv(file_idempotent(state), 
                                mode='a', index=False, header=False)

    # For convenience, we process the last generation into a video
    data.create_video_state(state)
    logging.info(f"[iterate] EA Iteration {state.run} complete")

def simulate_solutions(solution_set: list[stypes.solution], 
                       cpg_struct: CpgNetworkStructure,
                       body_shape: BodyV2, body_map: any,
                       config: stypes.EAConfig):
    
    # Create a list of robots to simulate with uniform weights
    robots = [
        ModularRobot(
            body=body_shape,
            brain=BrainCpgNetworkStatic.uniform_from_params(
                params=solution,
                cpg_network_structure=cpg_struct,
                initial_state_uniform=math.sqrt(2) * 0.5, 
                output_mapping=body_map))
        for solution in solution_set]

    def new_robot_scene(robot: ModularRobot) -> ModularRobotScene:
        s = ModularRobotScene(terrain=terrains.flat())
        s.add_robot(robot)
        return s
    

    scenes = [new_robot_scene(robot) for robot in robots]

    # This returns the robots that were used. This is a dependency for the 
    # scene state object returned as the second value in this tuple.
    return (robots, 
            simulate_scenes(
                simulator=LocalSimulator(headless=True, num_simulators=8),
                batch_parameters=make_standard_batch_parameters(
                    simulation_time=config.ttl,
                    sampling_frequency=config.freq),
                scenes=scenes))
    