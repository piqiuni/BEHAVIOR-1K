# :material-eye: **ROS Visualization for BEHAVIOR Environment**

This page describes the ROS visualization component that allows you to visualize observations and actions from the BEHAVIOR environment through ROS topics.

## Overview

The ROS visualization system provides a way to monitor and visualize data from the BEHAVIOR simulation environment in real-time using ROS (Robot Operating System). This is useful for:

- Debugging robot policies
- Monitoring simulation state during training or evaluation
- Integrating with existing ROS-based visualization tools like RViz
- Recording simulation data for offline analysis

## Architecture

The visualization system consists of three main components:

```
[BEHAVIOR Env] <-> [WebSocket Server] <-> [ROS Visualizer]
     |                    |                      |
(simulation)      (policy inference)     (visualization)
```

1. **BEHAVIOR Environment** (`behavior_env_web.py`): Runs the simulation and sends observations to the WebSocket server
2. **WebSocket Server** (`simple_websocket_server.py`): Receives observations, computes actions, and optionally forwards data to the visualizer
3. **ROS Visualizer** (`ros_visualizer.py`): Receives data and publishes to ROS topics for visualization

## Prerequisites

Before using the ROS visualization component, ensure you have:

- ROS (Noetic or later) installed
- `rospy`, `sensor_msgs`, `std_msgs` packages
- `cv_bridge` for image conversion
- Python packages: `websockets`, `msgpack`, `numpy`

## Quick Start

### Step 1: Start the ROS Visualizer

```bash
# Start the visualizer on default port (8001)
python ros_visualizer.py

# Or with custom settings
python ros_visualizer.py --host 0.0.0.0 --port 8001 --verbose
```

### Step 2: Start the WebSocket Server with Visualization

```bash
# Enable visualization forwarding
python simple_websocket_server.py --viz-host localhost --viz-port 8001
```

### Step 3: Run the BEHAVIOR Environment

```bash
python OmniGibson/omnigibson/examples/environments/behavior_env_web.py --host localhost --port 8000
```

## ROS Topics Published

The ROS visualizer publishes the following topics:

### Camera Images

| Topic | Message Type | Description |
|-------|--------------|-------------|
| `/behavior/left_realsense/rgb` | `sensor_msgs/Image` | Left RealSense RGB image |
| `/behavior/left_realsense/depth` | `sensor_msgs/Image` | Left RealSense depth image |
| `/behavior/right_realsense/rgb` | `sensor_msgs/Image` | Right RealSense RGB image |
| `/behavior/right_realsense/depth` | `sensor_msgs/Image` | Right RealSense depth image |
| `/behavior/zed/rgb` | `sensor_msgs/Image` | ZED camera RGB image |
| `/behavior/zed/depth` | `sensor_msgs/Image` | ZED camera depth image |

### Robot State

| Topic | Message Type | Description |
|-------|--------------|-------------|
| `/behavior/action` | `std_msgs/Float32MultiArray` | Robot action vector (23 dimensions for R1Pro) |
| `/behavior/proprioception` | `std_msgs/Float32MultiArray` | Robot proprioception data |

## Command Line Options

### ROS Visualizer (`ros_visualizer.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Host address to bind to |
| `--port` | `8001` | Port to listen on |
| `--action-dim` | `23` | Action space dimension |
| `--verbose` | `false` | Enable verbose logging |

### WebSocket Server (`simple_websocket_server.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Host address to bind to |
| `--port` | `8000` | Port to listen on |
| `--action-dim` | `23` | Action space dimension |
| `--viz-host` | `None` | Visualization server host (enables forwarding) |
| `--viz-port` | `8001` | Visualization server port |

## Viewing in RViz

To view the published images in RViz:

1. Launch RViz: `rosrun rviz rviz`
2. Add an Image display for each camera topic
3. Set the Image Topic to the desired topic (e.g., `/behavior/left_realsense/rgb`)

## Programmatic Usage

You can also use the visualization components programmatically in your own scripts:

```python
from ros_visualizer import VisualizationBridge, create_visualization_middleware

# Create a visualization bridge
visualizer = VisualizationBridge(server_host="localhost", server_port=8000)

# Wrap an existing policy with visualization
wrapped_policy = create_visualization_middleware(original_policy, visualizer)

# The wrapped policy will automatically send data to ROS
action = wrapped_policy.forward(observation)
```

## Running Without ROS

The visualizer can run in "mock mode" without ROS installed. In this mode, it will:

- Accept WebSocket connections normally
- Process incoming data
- Log statistics about data flow
- Skip actual ROS publishing

This is useful for testing the WebSocket communication without setting up a full ROS environment.

## Troubleshooting

### Connection Issues

If the visualizer cannot connect:

1. Ensure the visualization server is running before starting the WebSocket server
2. Check that ports are not blocked by firewall
3. Verify network connectivity between components

### No Data Appearing

If ROS topics are not receiving data:

1. Check that the WebSocket server was started with `--viz-host` option
2. Verify the visualization server is receiving connections (check logs)
3. Ensure ROS master is running (`roscore`)

### Image Format Issues

If images appear corrupted:

- The visualizer automatically handles common format conversions
- Supported formats: RGB8, RGBA, grayscale, float32 depth
- Check the camera observation keys match expected patterns
