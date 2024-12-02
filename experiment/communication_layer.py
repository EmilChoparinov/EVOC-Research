import logging
import socketio
import eventlet
from pprint import pprint
from flask import Flask
import threading
from queue import  Queue

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

        @sio.event
        def connect(sid, environ):
            logging.info(f"Established socket connection: {sid}")
            logging.info(f"{pprint(environ)}")

        @sio.event
        def disconnect():
            logging.info(f"Cannot reach other wire!")

        print(f"Opening server in new thread on port: {config.port}")
        threading.Thread(target=lambda: eventlet.wsgi.server(eventlet.listen(('', config.port)), app)).start()

    command_buffer = Queue()

    @sio.on('message')
    def on_message_recv(msg: Message):
        [command_buffer.put(cmd) for cmd in msg.commands]

    return sio, command_buffer, lambda x: sio.emit('message', x)
