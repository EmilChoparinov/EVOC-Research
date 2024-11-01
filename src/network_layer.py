"""
Goal:
The goal of this file is to provide an alternative, faster layer to interface 
between the physical robot 
"""


import asyncio
import time
from typing import Callable

import capnp
import numpy as np
from numpy.typing import NDArray
from pyrr import Vector3

from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.body.sensors import CameraSensor, IMUSensor
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot.brain import Brain

from revolve2.modular_robot_physical._config import Config
from revolve2.modular_robot_physical._hardware_type import HardwareType
from revolve2.modular_robot_physical._protocol_version import PROTOCOL_VERSION
from revolve2.modular_robot_physical._standard_port import STANDARD_PORT
from revolve2.modular_robot_physical._uuid_key import UUIDKey
from revolve2.modular_robot_physical.robot_daemon_api import robot_daemon_protocol_capnp

from  revolve2.modular_robot_physical.remote._camera_sensor_state_impl import CameraSensorStateImpl
from  revolve2.modular_robot_physical.remote._imu_sensor_state_impl import IMUSensorStateImpl
from  revolve2.modular_robot_physical.remote._modular_robot_control_interface_impl import ModularRobotControlInterfaceImpl
from  revolve2.modular_robot_physical.remote._modular_robot_sensor_state_impl_v1 import ModularRobotSensorStateImplV1
from  revolve2.modular_robot_physical.remote._modular_robot_sensor_state_impl_v2 import ModularRobotSensorStateImplV2


from typing import Callable

from revolve2.modular_robot_physical.remote._remote import _active_hinge_targets_to_pin_controls

def remote_control_with_polling_rate(config:Config, hostname:str, port:int, rate: float):
    async def _process(config:Config, hostname:str, port:int, rate: float):
        controller = config.modular_robot.brain.make_instance()
        
        # Perform connection call
        client = None
        service = None
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host=hostname,
                port=port
            )
            client = capnp.TwoPartyClient(connection)
            service = client.bootstrap().cast_as(robot_daemon_protocol_capnp.RoboServer)
        except ConnectionRefusedError:
            raise ConnectionRefusedError("Can't connect to robot! Network Error!")
    
        # Perform Protocol Setup. Reject if NOT on v2 hardware
        r: robot_daemon_protocol_capnp.SetupResponse = (await service.setup(
            robot_daemon_protocol_capnp.SetupArgs(
                version=PROTOCOL_VERSION, activePins=[x for x in range(32)] #??
            )
        )).response
        
        if not r.versionOk: raise RuntimeError("Protocol Version mismatch!")
        if not r.hardwareType == "v2": raise NotImplementedError("Only available for V2 hardware")
        
        pin_controls = _active_hinge_targets_to_pin_controls(
            config,
            [(active_hinge, config.initial_hinge_positions[active_hinge])
             for active_hinge in config.hinge_mapping],
        )
        
        # Read battery level for help!
        sens_read: robot_daemon_protocol_capnp.SensorReadings = (
            await service.controlAndReadSensors(robot_daemon_protocol_capnp.ControlAndReadSensorsArgs(
                setPins=pin_controls, readPins=[]
            ))
        ).response
        print(f"Battery level is at {sens_read.battery * 100.0}%! :)")
        
        print("Press enter to begin brain controller")
        input()
        
        control_period = 1 / config.control_frequency
        
        start_time = time.time()
        last_update_time = start_time
        elapsed_time: float
        while(current_time := time.time()) - start_time < config.run_duration:
            
            # Sleep until next poll time operation
            wait_to_poll_again = last_update_time + control_period
            if current_time < wait_to_poll_again:
                await asyncio.sleep(wait_to_poll_again - current_time)
                last_update_time = wait_to_poll_again
                elapsed_time = control_period
            else:
                print(
                    f"Warning! Loop is lagging behind {wait_to_poll_again - current_time} seconds!"
                )
                elapsed_time = last_update_time - current_time
                last_update_time = current_time

            # Get interface between controller and brain
            control_interface = ModularRobotControlInterfaceImpl()
            controller.control(
                elapsed_time,
                sensor_state=None,
                control_interface=control_interface
            )
            
            # Select PINS
            pin_controller = _active_hinge_targets_to_pin_controls(
                config, control_interface._set_active_hinges
            )
            
            # Send Network Request
            service.control(
                robot_daemon_protocol_capnp.ControlArgs(setPins=pin_controller)
            )
        
        
    asyncio.run(capnp.run(
        _process(
            config=config,
            hostname=hostname,
            port=port,
            rate=rate
        )
    ))