# WebSocket-Enabled BEHAVIOR Environment

## Overview

`behavior_env_web.py` is a modified version of `behavior_env_demo.py` that enables remote control of the BEHAVIOR robot environment through WebSocket communication.

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
