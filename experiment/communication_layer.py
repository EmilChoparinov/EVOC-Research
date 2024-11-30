import logging
import socketio

from dataclasses import dataclass
from typing import Literal
from enum import Enum

event = Literal['new_experiment', 'pause', 'play', 'restart']

class Event(Enum):
    new_experiment = 'e'
    pause          = 'pa'
    play           = 'pl'
    restart        = 'r'

@dataclass
class Payload:
    run_id: int

@dataclass
class Command:
    type: Event
    payload: Payload | None = None

@dataclass
class Message:
    commands: list[Command]

def boot_sockets():
    sio = socketio.Client()
    command_buffer: list[Command] = []

    @sio.event
    def connect():
        logging.info(f"Established socket connection")

    @sio.event
    def disconnect():
        logging.info(f"Cannot reach other wire!")

    @sio.on('message')
    def on_message_recv(msg: Message):
        command_buffer += msg.commands

    return sio, command_buffer, lambda x: sio.emit('message', x)