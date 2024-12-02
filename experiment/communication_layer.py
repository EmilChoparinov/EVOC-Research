import logging
import socketio
import eventlet
from flask import Flask

from dataclasses import dataclass
from typing import Literal
from enum import Enum

import experiment.config as config

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

def boot_sockets(ctype: Literal['server','client']):
    sio = socketio.Client()
    app = Flask(__name__)

    if ctype == 'server':
        sio = socketio.Server()
        app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
        eventlet.wsgi.server(eventlet.listen(('', config.port)), app)

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
