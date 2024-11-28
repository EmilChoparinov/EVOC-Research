# Real World Experiment Collection Tools
The code provided in this folder is used to capture video footage of the robot
performing its CPG. Below is the architecture used. 

```
+----------+                  +----------+   +----------+
|          |<-----+    +----->|          |   |          |
|  Robot   |      |    |      | Recorder +-->| Local FS |
|          +----+ |    | +----+          |   |          |
+----+-----+    | |    | |    +----------+   +----------+
     |          v |    | v          ^                    
     |         +--+----+--+         |                    
     |         |          |         |                    
     |         |  Socket  |         |                    
     v         |          |         |                    
+----------+   +----------+   +-----+----+               
|          |                  |          |               
|  Motors  |                  |  Camera  |               
|          |                  |          |               
+----------+                  +----------+               
```
Note that all arrows except Recorder -> Local FS are network based. Therefore,
you must be on the same network as the robot to use this program.

# Usage
Ensure you have properly setup the correct ports and ip addresses for the robot
and your machine in `config.py`

You must SSH into the robot and run `python3 -m experiment.robot` on its own
hardware. You may run `python3 -m experiment.recorder` on any machine as long as
you are within the local network.

`robot.py` will provide a simple CLI interface where you can press the enter
button to start the next CPG. This will sync with the recorder to start a new
recording.

In the event the current CPG recording is poor, you may reset the robot to
by pressing 'r' in the recorder window. This will cause to robot to stop
moving and return to its initial CPG state.

You may also press play and pause to stop the recording midway which will also
stop/start the robot.
