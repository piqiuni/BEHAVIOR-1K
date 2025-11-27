# WebSocket-Enabled BEHAVIOR Environment

## Overview

`behavior_env_web.py` is a modified version of `behavior_env_demo.py` that enables remote control of the BEHAVIOR robot environment through WebSocket communication.

## Architecture

The system supports three main components that can run independently:

```
┌─────────────────┐     WebSocket     ┌─────────────────────┐
│  BEHAVIOR Env   │◄─────────────────►│  WebSocket Server   │
│ (behavior_env_  │     obs/action    │ (simple_websocket_  │
│  web.py)        │                   │  server.py)         │
└─────────────────┘                   └──────────┬──────────┘
                                                 │ (optional)
                                                 │ Forward
                                                 ▼
                                      ┌─────────────────────┐
                                      │  ROS Visualizer     │
                                      │ (ros_visualizer.py) │
                                      └──────────┬──────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────┐
                                      │  ROS Topics         │
                                      │  /behavior/...      │
                                      └─────────────────────┘
```

## Components

### 1. Environment Client (`behavior_env_web.py`)
- Runs the OmniGibson simulation
- Sends observations to WebSocket server
- Receives and executes actions

### 2. WebSocket Server (`simple_websocket_server.py`)
- Receives observations from environment
- Runs policy to generate actions
- Optionally forwards data to ROS visualizer

### 3. ROS Visualizer (`ros_visualizer.py`)
- Receives observations and actions via WebSocket
- Publishes to ROS topics for visualization
- Runs independently as a monitoring component

## Key Features

### 1. **Observation Preprocessing** (参考 `eval.py`)
- Flattens nested observation dictionaries
- Computes camera relative poses (3 cameras: head, left_wrist, right_wrist)
- Adds task ID encoding
- Converts all data to NumPy arrays for network transmission

### 2. **WebSocket Communication** (基于 `WebsocketPolicy`)
- Sends preprocessed observations to remote server
- Receives action commands from server
- Automatic reconnection on connection loss
- Uses msgpack for efficient serialization

### 3. **Observation Structure**

Sent to WebSocket server:
```python
{
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.ndarray,  # (480, 480, 3)
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.ndarray, # (480, 480, 3)
    "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.ndarray,             # (720, 720, 3)
    "robot_r1::proprio": np.ndarray,                                     # Proprioception
    "robot_r1::cam_rel_poses": np.ndarray,                               # (21,) 3×7 poses
    "task_id": np.ndarray,                                               # (1,) task index
}
```

### 4. **Expected Action Format**

Server should return:
```python
{
    "action": np.ndarray  # Shape: (23,) for R1Pro
}
```

Action dimensions (23-DoF):
- Base movement: 3 DoF (x, y, rotation)
- Torso: 1 DoF
- Left arm: 7 DoF
- Right arm: 7 DoF
- Left gripper: 1 DoF
- Right gripper: 1 DoF
- Head: 2 DoF (pan, tilt)

## Usage

### Basic Usage
```bash
# Default: localhost:8000, assembling_gift_baskets task
python behavior_env_web.py
```

### Custom Configuration
```bash
# Custom server address
python behavior_env_web.py --host 192.168.1.100 --port 9000

# Different task
python behavior_env_web.py --task cleaning_bathrooms

# Headless mode (no GUI)
python behavior_env_web.py --headless

# Quick test (1 episode only)
python behavior_env_web.py --short
```

### Combined Options
```bash
python behavior_env_web.py --host 0.0.0.0 --port 8080 --task assembling_gift_baskets --headless
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--host` | str | `localhost` | WebSocket server hostname or IP |
| `--port` | int | `8000` | WebSocket server port |
| `--task` | str | `assembling_gift_baskets` | BEHAVIOR task name |
| `--headless` | flag | `False` | Run without GUI |
| `--short` | flag | `False` | Run only 1 iteration |

## Server Implementation Example

A simple WebSocket server should:

1. Accept WebSocket connections on the specified port
2. Receive msgpack-encoded observations
3. Process observations and generate actions
4. Send back msgpack-encoded action dictionary
5. Handle reset signals: `{"reset": True}`

Example server structure:
```python
import asyncio
import websockets
import msgpack
import numpy as np

async def handle_client(websocket, path):
    async for message in websocket:
        obs = msgpack.unpackb(message, raw=False)
        
        if obs.get("reset"):
            # Handle reset
            continue
        
        # Your policy logic here
        action = your_policy(obs)
        
        response = {"action": action}
        await websocket.send(msgpack.packb(response))

asyncio.get_event_loop().run_until_complete(
    websockets.serve(handle_client, "0.0.0.0", 8000)
)
```

## Differences from `behavior_env_demo.py`

| Feature | `behavior_env_demo.py` | `behavior_env_web.py` |
|---------|------------------------|----------------------|
| Control | Random actions | WebSocket server |
| Observations | Not processed | Preprocessed (flattened, camera poses) |
| Task ID | Not included | Included in observations |
| Configuration | Hardcoded | Command-line arguments |
| Reset handling | Simple | Notifies policy server |
| Error handling | Minimal | Fallback to zero actions |

## Dependencies

All dependencies from `eval.py`:
- `omnigibson.learning.policies.WebsocketPolicy`
- `omnigibson.learning.utils.eval_utils`
- `omnigibson.learning.utils.array_tensor_utils`
- `omnigibson.utils.transform_utils`

## Troubleshooting

### Connection Issues
- Ensure WebSocket server is running before starting the environment
- Check firewall settings if connecting to remote server
- Verify correct host and port

### Performance
- Large RGB images may cause network bottlenecks
- Consider image compression or downsampling for remote deployment
- Use local network for best performance

### Task Names
If you get a warning about task name not found, check available tasks:
```python
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
print(TASK_NAMES_TO_INDICES.keys())
```

## Implementation Details

The implementation follows the architecture from `eval.py`:

1. **Observation Preprocessing** (`preprocess_obs()`)
   - Mirrors `Evaluator._preprocess_obs()` from `eval.py`
   - Ensures consistency with training/evaluation pipeline

2. **Camera Pose Computation**
   - Uses camera view transform matrices when available
   - Falls back to direct poses during initialization
   - Computes relative poses w.r.t. robot base

3. **WebSocket Policy**
   - Wraps `WebsocketClientPolicy` from `network_utils`
   - Handles serialization/deserialization automatically
   - Provides automatic reconnection

## Next Steps

- Implement your policy server
- Test with simple actions first
- Monitor network latency
- Consider implementing action smoothing for better control

---

## ROS Visualization Component

### Overview

The `ros_visualizer.py` script provides a visualization endpoint that receives observations and actions from the WebSocket server and publishes them through ROS topics.

### ROS Topics Published

| Topic | Message Type | Description |
|-------|--------------|-------------|
| `/behavior/left_realsense/rgb` | `sensor_msgs/Image` | Left realsense RGB camera |
| `/behavior/left_realsense/depth` | `sensor_msgs/Image` | Left realsense depth camera |
| `/behavior/right_realsense/rgb` | `sensor_msgs/Image` | Right realsense RGB camera |
| `/behavior/right_realsense/depth` | `sensor_msgs/Image` | Right realsense depth camera |
| `/behavior/zed/rgb` | `sensor_msgs/Image` | ZED RGB camera |
| `/behavior/zed/depth` | `sensor_msgs/Image` | ZED depth camera |
| `/behavior/action` | `std_msgs/Float32MultiArray` | Robot action (23-DoF) |
| `/behavior/proprioception` | `std_msgs/Float32MultiArray` | Robot proprioception |

### Usage

#### Step 1: Start the ROS Visualizer
```bash
# Start visualizer server on port 8001
python ros_visualizer.py --port 8001
```

#### Step 2: Start the WebSocket Server with Visualization Forwarding
```bash
# Start policy server with visualization forwarding enabled
python simple_websocket_server.py --port 8000 --viz-host localhost --viz-port 8001
```

#### Step 3: Start the Environment
```bash
# Start the BEHAVIOR environment
python OmniGibson/omnigibson/examples/environments/behavior_env_web.py --host localhost --port 8000
```

### Command Line Arguments (ros_visualizer.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--host` | str | `0.0.0.0` | Host address to bind to |
| `--port` | int | `8001` | Port to listen on |
| `--action-dim` | int | `23` | Action space dimension |
| `--verbose` | flag | `False` | Enable verbose logging |

### Integration with RViz

Once the visualizer is running, you can view the camera feeds in RViz:

1. Launch RViz: `rosrun rviz rviz`
2. Add Image displays for:
   - `/behavior/left_realsense/rgb`
   - `/behavior/right_realsense/rgb`
   - `/behavior/zed/rgb`
3. Monitor actions on the terminal or use `rostopic echo /behavior/action`

### Mock Mode

If ROS is not available, the visualizer will run in mock mode and print warnings. This allows testing the WebSocket communication without a full ROS installation.

### Dependencies

- Python 3.7+
- `websockets`
- `msgpack`
- `numpy`
- ROS (rospy, sensor_msgs, std_msgs, cv_bridge) - optional

### Example: Custom Visualization Processing

You can extend the `ROSVisualizerPublisher` class for custom visualization:

```python
from ros_visualizer import ROSVisualizerPublisher, VisualizationBridge

# Create a bridge
bridge = VisualizationBridge()

# Update with observation and action
bridge.update(obs=observation_dict, action=action_array)

# Or wrap an existing policy
from ros_visualizer import create_visualization_middleware

wrapped_policy = create_visualization_middleware(original_policy, bridge)
action = wrapped_policy.forward(obs)  # Automatically sends to visualizer
```
