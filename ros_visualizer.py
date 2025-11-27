"""
ROS Visualization Client for BEHAVIOR Environment

This script implements a visualization endpoint that:
- Connects to a WebSocket server (like simple_websocket_server.py)
- Receives observations (obs) and actions from the simulation
- Publishes observations and actions through ROS topics for visualization

The visualizer runs independently from the web (behavior_env_web.py) and server
(simple_websocket_server.py) components, acting as a third party that subscribes
to the data stream for monitoring and visualization purposes.

Architecture:
    [BEHAVIOR Env] <-> [WebSocket Server] <-> [ROS Visualizer]
         |                    |                      |
    (simulation)      (policy inference)     (visualization)

Usage:
    # Start with default settings (localhost:8001 for visualization port)
    python ros_visualizer.py
    
    # Custom host and port
    python ros_visualizer.py --host 192.168.1.100 --port 8001
    
    # With verbose logging
    python ros_visualizer.py --host localhost --port 8001 --verbose

ROS Topics Published:
    Observations (camera images):
        /behavior/left_realsense/rgb (sensor_msgs/Image)
        /behavior/left_realsense/depth (sensor_msgs/Image)
        /behavior/right_realsense/rgb (sensor_msgs/Image)
        /behavior/right_realsense/depth (sensor_msgs/Image)
        /behavior/zed/rgb (sensor_msgs/Image)
        /behavior/zed/depth (sensor_msgs/Image)
    
    Actions:
        /behavior/action (std_msgs/Float32MultiArray)
    
    Proprioception:
        /behavior/proprioception (std_msgs/Float32MultiArray)

Requirements:
    - ROS (rospy, sensor_msgs, std_msgs)
    - cv_bridge
    - websockets
    - msgpack
    - numpy
"""

import asyncio
import argparse
import functools
import logging
import msgpack
import numpy as np
import threading
import time
import websockets
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ros_visualizer")

# ROS imports with fallback
try:
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32MultiArray, Header
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    logger.warning("ROS not available. Running in mock mode (no actual publishing).")
    rospy = None
    Image = None
    Float32MultiArray = None
    Header = None
    CvBridge = None
    ROS_AVAILABLE = False


# NumPy array support for msgpack (compatible with network_utils.py)
def pack_array(obj):
    """Pack NumPy arrays for msgpack serialization."""
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    """Unpack NumPy arrays from msgpack."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


# Create packer/unpacker with NumPy support
packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def to_cv_img(tensor_or_arr, is_depth: bool = False) -> np.ndarray:
    """
    Convert tensor or numpy array to cv2-compatible image format.
    
    Args:
        tensor_or_arr: Input tensor or array
        is_depth: If True, treat as depth image (float32)
    
    Returns:
        np.ndarray: Image array suitable for cv_bridge
    """
    # Convert to numpy if needed
    try:
        import torch
        if isinstance(tensor_or_arr, torch.Tensor):
            arr = tensor_or_arr.detach().cpu().numpy()
        else:
            arr = np.array(tensor_or_arr)
    except ImportError:
        arr = np.array(tensor_or_arr)

    # Depth: ensure float32 single channel
    if is_depth:
        arr = arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return arr

    # RGB: ensure HxWx3 uint8
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.1:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    # If channel-first (CxHxW), transpose to channel-last (HxWxC)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # If RGBA, drop alpha channel
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # If grayscale, replicate to RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    return np.ascontiguousarray(arr)


class ROSVisualizerPublisher:
    """
    Handles ROS publishing of observations and actions.
    """
    
    def __init__(self, node_name: str = "behavior_visualizer"):
        """
        Initialize ROS publishers.
        
        Args:
            node_name: Name for the ROS node
        """
        self.bridge = None
        self.pubs = {}
        self.initialized = False
        
        if not ROS_AVAILABLE:
            logger.warning("ROS not available, publishers will not be created")
            return
        
        try:
            # Initialize ROS node
            try:
                if not rospy.core.is_initialized():
                    rospy.init_node(node_name, anonymous=True)
            except Exception:
                pass
            
            # Create cv_bridge for image conversion
            self.bridge = CvBridge()
            
            # Create publishers for camera observations
            self.pubs = {
                # Left realsense camera
                "left_rgb": rospy.Publisher("/behavior/left_realsense/rgb", Image, queue_size=1),
                "left_depth": rospy.Publisher("/behavior/left_realsense/depth", Image, queue_size=1),
                # Right realsense camera
                "right_rgb": rospy.Publisher("/behavior/right_realsense/rgb", Image, queue_size=1),
                "right_depth": rospy.Publisher("/behavior/right_realsense/depth", Image, queue_size=1),
                # ZED camera
                "zed_rgb": rospy.Publisher("/behavior/zed/rgb", Image, queue_size=1),
                "zed_depth": rospy.Publisher("/behavior/zed/depth", Image, queue_size=1),
                # Action publisher
                "action": rospy.Publisher("/behavior/action", Float32MultiArray, queue_size=1),
                # Proprioception publisher
                "proprioception": rospy.Publisher("/behavior/proprioception", Float32MultiArray, queue_size=1),
            }
            
            self.initialized = True
            logger.info("ROS publishers initialized successfully")
            logger.info("Publishing to topics:")
            for name, pub in self.pubs.items():
                logger.info(f"  - {pub.name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ROS publishers: {e}")
    
    def publish_observation(self, obs: Dict[str, Any]) -> None:
        """
        Publish observation data to ROS topics.
        
        Args:
            obs: Observation dictionary containing camera images and proprioception
        """
        if not self.initialized or not ROS_AVAILABLE:
            return
        
        try:
            # Camera link name patterns to search for in obs
            camera_patterns = {
                "left_realsense": ["left_realsense_link:Camera:0", "left_realsense"],
                "right_realsense": ["right_realsense_link:Camera:0", "right_realsense"],
                "zed": ["zed_link:Camera:0", "zed"],
            }
            
            # Try to find and publish camera images
            for cam_key, patterns in camera_patterns.items():
                rgb_img = None
                depth_img = None
                
                # Search for matching keys in obs
                for pattern in patterns:
                    for obs_key in obs.keys():
                        if pattern in str(obs_key):
                            if "rgb" in str(obs_key).lower():
                                rgb_img = obs[obs_key]
                            elif "depth" in str(obs_key).lower():
                                depth_img = obs[obs_key]
                
                # Publish if found
                if rgb_img is not None:
                    try:
                        img = to_cv_img(rgb_img, is_depth=False)
                        msg = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
                        self.pubs[f"{cam_key.split('_')[0]}_rgb"].publish(msg)
                    except Exception as e:
                        logger.debug(f"Failed to publish {cam_key} RGB: {e}")
                
                if depth_img is not None:
                    try:
                        img = to_cv_img(depth_img, is_depth=True)
                        msg = self.bridge.cv2_to_imgmsg(img, encoding="32FC1")
                        self.pubs[f"{cam_key.split('_')[0]}_depth"].publish(msg)
                    except Exception as e:
                        logger.debug(f"Failed to publish {cam_key} depth: {e}")
            
            # Publish proprioception if present
            proprio_keys = ["proprio", "proprioception", "joint_pos", "joint_vel"]
            for key in proprio_keys:
                if key in obs:
                    proprio_data = obs[key]
                    if isinstance(proprio_data, np.ndarray):
                        msg = Float32MultiArray()
                        msg.data = proprio_data.flatten().astype(np.float32).tolist()
                        self.pubs["proprioception"].publish(msg)
                        break
                        
        except Exception as e:
            logger.debug(f"Error publishing observation: {e}")
    
    def publish_action(self, action: np.ndarray) -> None:
        """
        Publish action data to ROS topic.
        
        Args:
            action: Action array
        """
        if not self.initialized or not ROS_AVAILABLE:
            return
        
        try:
            msg = Float32MultiArray()
            if isinstance(action, np.ndarray):
                msg.data = action.flatten().astype(np.float32).tolist()
            else:
                msg.data = list(action)
            self.pubs["action"].publish(msg)
        except Exception as e:
            logger.debug(f"Error publishing action: {e}")


class VisualizationWebSocketServer:
    """
    WebSocket server that receives observations and actions from the main
    WebSocket policy server and forwards them to ROS for visualization.
    
    This acts as a "visualization tap" into the data stream between the
    BEHAVIOR environment and the policy server.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8001,
        action_dim: int = 23,
    ):
        """
        Initialize the visualization WebSocket server.
        
        Args:
            host: Host address to bind to
            port: Port to listen on
            action_dim: Expected action dimension
        """
        self.host = host
        self.port = port
        self.action_dim = action_dim
        
        # ROS publisher
        self.ros_publisher = ROSVisualizerPublisher()
        
        # Statistics
        self.msg_count = 0
        self.last_log_time = time.time()
        
    async def handle_visualization_data(self, websocket):
        """
        Handle incoming visualization data from a client.
        
        Expected message format:
        {
            "obs": {...},      # Observation dictionary
            "action": [...]    # Action array (optional)
        }
        """
        client_address = websocket.remote_address
        logger.info(f"Visualization client connected from {client_address}")
        
        try:
            # Send server metadata
            metadata = {
                "type": "visualizer",
                "action_dim": self.action_dim,
                "ros_available": ROS_AVAILABLE,
            }
            await websocket.send(packb(metadata))
            
            async for message in websocket:
                try:
                    data = unpackb(message)
                    self.msg_count += 1
                    
                    # Extract and publish observation
                    if "obs" in data:
                        self.ros_publisher.publish_observation(data["obs"])
                    
                    # Extract and publish action
                    if "action" in data:
                        action = data["action"]
                        if isinstance(action, np.ndarray):
                            self.ros_publisher.publish_action(action)
                        else:
                            self.ros_publisher.publish_action(np.array(action))
                    
                    # Log statistics periodically
                    now = time.time()
                    if now - self.last_log_time > 10.0:
                        rate = self.msg_count / (now - self.last_log_time)
                        logger.info(f"Visualization rate: {rate:.1f} msg/s")
                        self.msg_count = 0
                        self.last_log_time = now
                    
                    # Send acknowledgment
                    await websocket.send(packb({"status": "ok"}))
                    
                except Exception as e:
                    logger.error(f"Error processing visualization message: {e}")
                    
        except Exception as e:
            logger.info(f"Visualization client {client_address} disconnected: {e}")
    
    def process_request(self, connection, request):
        """Handle HTTP requests (health checks)."""
        try:
            from websockets.http11 import Response
            from websockets.datastructures import Headers
        except ImportError:
            return None
        
        if request.path == "/healthz":
            return Response(
                status_code=200,
                reason_phrase="OK",
                headers=Headers([("Content-Type", "text/plain")]),
                body=b"OK\n"
            )
        
        # Check WebSocket upgrade
        conn_hdr = request.headers.get("Connection", "")
        upgrade_hdr = request.headers.get("Upgrade", "")
        if "upgrade" in conn_hdr.lower() and "websocket" in upgrade_hdr.lower():
            return None
        
        return Response(
            status_code=426,
            reason_phrase="Upgrade Required",
            headers=Headers([("Content-Type", "text/plain")]),
            body=b"426 Upgrade Required\n"
        )
    
    async def run(self):
        """Start the WebSocket server."""
        logger.info("=" * 60)
        logger.info("ROS Visualization WebSocket Server")
        logger.info("=" * 60)
        logger.info(f"Server address: ws://{self.host}:{self.port}")
        logger.info(f"ROS available: {ROS_AVAILABLE}")
        logger.info("=" * 60)
        logger.info("Waiting for visualization data...")
        
        async with websockets.serve(
            self.handle_visualization_data,
            self.host,
            self.port,
            process_request=self.process_request,
            max_size=100 * 1024 * 1024,  # 100 MB
            ping_interval=20,
            ping_timeout=10,
        ):
            await asyncio.Future()  # Run forever


class VisualizationBridge:
    """
    A bridge that connects to an existing WebSocket policy server and
    forwards all observations and actions to ROS for visualization.
    
    This is useful when you want to visualize data from an existing
    server without modifying the server code.
    """
    
    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 8000,
        action_dim: int = 23,
    ):
        """
        Initialize the visualization bridge.
        
        Args:
            server_host: Host of the WebSocket policy server to connect to
            server_port: Port of the WebSocket policy server
            action_dim: Expected action dimension
        """
        self.server_host = server_host
        self.server_port = server_port
        self.action_dim = action_dim
        
        # ROS publisher
        self.ros_publisher = ROSVisualizerPublisher()
        
        # State
        self._running = False
        self._latest_obs = None
        self._latest_action = None
        self._lock = threading.Lock()
    
    def update(self, obs: Optional[Dict[str, Any]] = None, action: Optional[np.ndarray] = None):
        """
        Update with new observation and/or action data.
        
        Args:
            obs: Observation dictionary
            action: Action array
        """
        with self._lock:
            if obs is not None:
                self._latest_obs = obs
            if action is not None:
                self._latest_action = action
        
        # Publish to ROS
        if obs is not None:
            self.ros_publisher.publish_observation(obs)
        if action is not None:
            self.ros_publisher.publish_action(action)


def create_visualization_middleware(original_policy, visualizer: VisualizationBridge):
    """
    Create a wrapper around a policy that sends data to the visualizer.
    
    This can be used to add visualization to an existing policy without
    modifying its code.
    
    Args:
        original_policy: The original policy to wrap
        visualizer: VisualizationBridge instance
    
    Returns:
        Wrapped policy with visualization
    """
    class VisualizationWrapper:
        def __init__(self, policy, viz):
            self.policy = policy
            self.viz = viz
        
        def forward(self, obs, *args, **kwargs):
            # Send observation to visualizer
            self.viz.update(obs=obs)
            
            # Get action from original policy
            action = self.policy.forward(obs, *args, **kwargs)
            
            # Send action to visualizer
            if hasattr(action, 'numpy'):
                self.viz.update(action=action.numpy())
            else:
                self.viz.update(action=np.array(action))
            
            return action
        
        def act(self, obs, *args, **kwargs):
            return self.forward(obs, *args, **kwargs)
        
        def reset(self):
            if hasattr(self.policy, 'reset'):
                self.policy.reset()
    
    return VisualizationWrapper(original_policy, visualizer)


async def main(host: str = "0.0.0.0", port: int = 8001, action_dim: int = 23):
    """
    Main entry point for the ROS visualization server.
    
    Args:
        host: Host address to bind to
        port: Port to listen on
        action_dim: Expected action dimension
    """
    server = VisualizationWebSocketServer(host=host, port=port, action_dim=action_dim)
    await server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ROS Visualization Server for BEHAVIOR Environment"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to listen on (default: 8001)"
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=23,
        help="Action space dimension (default: 23)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        asyncio.run(main(host=args.host, port=args.port, action_dim=args.action_dim))
    except KeyboardInterrupt:
        logger.info("\nVisualization server stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
