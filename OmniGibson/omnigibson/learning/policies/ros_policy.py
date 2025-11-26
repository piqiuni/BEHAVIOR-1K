"""
ROS-backed policy that publishes sensor observations and accepts/returns actions via ROS topics.
Adapted from behavior_env_ros.py's RosPolicy class for use in the learning framework.

This policy:
- Publishes camera RGB/depth images to fixed ROS topics (e.g., /robot/left_realsense/rgb)
- Subscribes to a control topic (/behavior/control by default) for action commands
- Can be used standalone or as a fallback that returns zero actions when ROS is unavailable
"""

import numpy as np
import threading
import logging
from typing import Optional, Dict, Any

try:
    import torch as th
except ImportError:
    th = None

try:
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32MultiArray
    from cv_bridge import CvBridge
except Exception:
    rospy = None
    Image = None
    Float32MultiArray = None
    CvBridge = None

logger = logging.getLogger(__name__)


def to_cv_img(tensor_or_arr, is_depth: bool = False) -> np.ndarray:
    """
    Convert torch tensor or numpy array to a cv2-compatible numpy image.
    
    - Depth images: returned as float32 2D arrays
    - RGB images: returned as uint8 HxWx3 arrays
    
    Args:
        tensor_or_arr: input tensor or array
        is_depth: if True, treat as depth (single channel, float32); else RGB
        
    Returns:
        np.ndarray: contiguous numpy array suitable for cv_bridge
    """
    try:
        import torch
        if isinstance(tensor_or_arr, torch.Tensor):
            arr = tensor_or_arr.detach().cpu().numpy()
        else:
            arr = np.array(tensor_or_arr)
    except Exception:
        arr = np.array(tensor_or_arr)

    # Depth: ensure float32 single channel
    if is_depth:
        arr = arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return arr

    # RGB: ensure HxWx3 uint8
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        if arr.max() <= 1.1:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    # If channel-first, transpose to channel-last
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # If image has 4 channels (RGBA/BGRA), drop alpha
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # If grayscale, replicate to RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # Ensure contiguous for cv_bridge
    arr = np.ascontiguousarray(arr)
    return arr


class RosPolicy:
    """
    ROS-backed policy that publishes observations and returns zero actions (or actions from ROS).
    
    Can be used as a LocalPolicy-like interface:
    - `act(obs)` / `forward(obs)` return a torch Tensor or numpy array of shape (action_dim,)
    - `reset()` clears internal buffers
    
    Fixed ROS topics:
    - Publishers (camera data): /robot/left_realsense/rgb, /robot/left_realsense/depth, etc.
    - Subscriber (control): /behavior/control (Float32MultiArray)
    """

    def __init__(
        self,
        action_dim: Optional[int] = 23,
        control_topic: str = "/behavior/control",
        queue_size: int = 1,
        return_zeros: bool = True,
        rate_hz: int = 10,
        **kwargs
    ) -> None:
        """
        Initialize RosPolicy.
        
        Args:
            action_dim: dimension of action vector (default 23 for R1Pro)
            control_topic: ROS topic to subscribe to for control commands
            queue_size: ROS subscription queue size
            return_zeros: if True, return zero actions instead of waiting for ROS; 
                         if False, block until action arrives or timeout
            rate_hz: publishing rate (Hz)
        """
        self.action_dim = int(action_dim) if action_dim is not None else 23
        self.control_topic = control_topic
        self.return_zeros = return_zeros
        self.rate_hz = rate_hz
        
        # Internal state for ROS subscription
        self._latest_action = np.zeros(self.action_dim, dtype=np.float32)
        self._action_lock = threading.Lock()
        self._new_action_event = threading.Event()
        
        # ROS infrastructure
        self.sub = None
        self.bridge = CvBridge() if CvBridge is not None else None
        self.pubs = {}
        
        # Camera link name substrings for observation extraction
        self.camera_link_names = {
            "left_realsense": "left_realsense_link:Camera:0",
            "right_realsense": "right_realsense_link:Camera:0",
            "zed": "zed_link:Camera:0",
        }
        
        # Initialize ROS if available
        self._init_ros(queue_size)

    def _init_ros(self, queue_size: int) -> None:
        """Initialize ROS node and subscriber."""
        if rospy is None or Float32MultiArray is None:
            logger.info("ROS not available; RosPolicy will operate in zero-action mode")
            return
        
        try:
            # Initialize ROS node if not already done
            try:
                if not rospy.core.is_initialized():
                    rospy.init_node("ros_policy_node", anonymous=True)
            except Exception:
                # Some rospy builds don't expose is_initialized
                pass
            
            # Subscribe to control topic
            self.sub = rospy.Subscriber(
                self.control_topic,
                Float32MultiArray,
                self._on_action_received,
                queue_size=queue_size
            )
            logger.info(f"RosPolicy subscribed to {self.control_topic}")
        except Exception as e:
            logger.warning(f"Failed to initialize ROS subscription: {e}")
            self.sub = None

    def setup_publishers_for_robot(self, robot) -> None:
        """
        Setup fixed publishers for camera topics.
        
        Publishers use fixed names (no robot name): /robot/left_realsense/rgb, etc.
        
        Args:
            robot: R1Pro robot instance (for reference; publishers are fixed-topic)
        """
        if rospy is None or Image is None:
            logger.debug("ROS Image publisher not available")
            return
        
        self.pubs = {
            "left_rgb": rospy.Publisher("/robot/left_realsense/rgb", Image, queue_size=1),
            "left_depth": rospy.Publisher("/robot/left_realsense/depth", Image, queue_size=1),
            "right_rgb": rospy.Publisher("/robot/right_realsense/rgb", Image, queue_size=1),
            "right_depth": rospy.Publisher("/robot/right_realsense/depth", Image, queue_size=1),
            "zed_rgb": rospy.Publisher("/robot/zed/rgb", Image, queue_size=1),
            "zed_depth": rospy.Publisher("/robot/zed/depth", Image, queue_size=1),
        }
        logger.info("RosPolicy publishers initialized for fixed topics (/robot/...)")

    def publish_observations(self, obs: Dict[str, Any], robot) -> None:
        """
        Extract and publish camera observations as ROS Image messages.
        
        Args:
            obs: observation dictionary (expected to be preprocessed/flattened)
            robot: R1Pro robot instance
        """
        if rospy is None or self.bridge is None or not self.pubs:
            return
        
        try:
            # Attempt to extract camera images from observation dict
            # Expected structure: obs contains keys like "robot_r1::<camera>::rgb", "robot_r1::<camera>::depth"
            
            images = {}
            for cam_key, link_name in self.camera_link_names.items():
                # Try to find RGB and depth in obs
                rgb_key = f"robot_r1:{link_name}:rgb" if f"robot_r1:{link_name}:rgb" in obs else None
                depth_key = f"robot_r1:{link_name}:depth" if f"robot_r1:{link_name}:depth" in obs else None
                
                if rgb_key:
                    images[f"{cam_key}_rgb"] = to_cv_img(obs[rgb_key], is_depth=False)
                if depth_key:
                    images[f"{cam_key}_depth"] = to_cv_img(obs[depth_key], is_depth=True)
            
            # Publish available images
            pub_map = {
                "left_realsense_rgb": "left_rgb",
                "left_realsense_depth": "left_depth",
                "right_realsense_rgb": "right_rgb",
                "right_realsense_depth": "right_depth",
                "zed_rgb": "zed_rgb",
                "zed_depth": "zed_depth",
            }
            
            for img_key, pub_key in pub_map.items():
                if img_key in images and pub_key in self.pubs:
                    img = images[img_key]
                    encoding = "32FC1" if "depth" in img_key else "rgb8"
                    msg = self.bridge.cv2_to_imgmsg(img, encoding=encoding)
                    self.pubs[pub_key].publish(msg)
        
        except Exception as e:
            logger.debug(f"Failed to publish observations: {e}")

    def _on_action_received(self, msg) -> None:
        """ROS subscriber callback for action messages.
        
        Args:
            msg: std_msgs/Float32MultiArray message
        """
        try:
            data = np.array(msg.data, dtype=np.float32)
            
            # Pad or truncate to action_dim
            if data.size != self.action_dim:
                if data.size < self.action_dim:
                    data = np.concatenate([data, np.zeros(self.action_dim - data.size, dtype=np.float32)])
                else:
                    data = data[:self.action_dim]
            
            with self._action_lock:
                self._latest_action = data.copy()
            
            # Signal that new action arrived
            self._new_action_event.set()
        except Exception as e:
            logger.debug(f"Error processing action message: {e}")

    def get_latest_action(self) -> np.ndarray:
        """Return the latest action as a numpy array."""
        with self._action_lock:
            return self._latest_action.copy()

    def act(self, obs: Dict[str, Any], **kwargs) -> Any:
        """Forward pass (alias for forward)."""
        return self.forward(obs, **kwargs)

    def forward(
        self,
        obs: Optional[Dict[str, Any]] = None,
        robot = None,
        timeout: float = 1.0,
        **kwargs
    ) -> Any:
        """
        Get action from ROS or return zeros.
        
        Args:
            obs: observation dictionary (optional; used to publish camera data)
            robot: robot instance (optional; used to publish observations)
            timeout: seconds to wait for ROS action (ignored if return_zeros=True)
            
        Returns:
            torch.Tensor or np.ndarray of shape (action_dim,) representing the action
        """
        # If return_zeros is True, just publish obs and return zeros without waiting
        if self.return_zeros:
            if obs is not None and robot is not None:
                try:
                    self.publish_observations(obs, robot)
                except Exception:
                    pass
            
            action = np.zeros(self.action_dim, dtype=np.float32)
            if th is not None:
                return th.from_numpy(action).to(dtype=th.float32)
            return action
        
        # Otherwise, attempt to wait for ROS action
        if obs is not None and robot is not None:
            try:
                self.publish_observations(obs, robot)
            except Exception:
                pass
        
        # Wait for new action (clear event first)
        try:
            self._new_action_event.clear()
        except Exception:
            pass
        
        got_action = False
        try:
            got_action = self._new_action_event.wait(timeout=timeout)
        except Exception:
            pass
        
        # Return latest action or zeros
        action = self.get_latest_action()
        if not got_action:
            logger.debug(f"Timeout waiting for ROS action (>{timeout}s); returning zeros")
        
        if th is not None:
            return th.from_numpy(action).to(dtype=th.float32)
        return action

    def reset(self) -> None:
        """Reset internal state."""
        with self._action_lock:
            self._latest_action = np.zeros(self.action_dim, dtype=np.float32)
        try:
            self._new_action_event.clear()
        except Exception:
            pass

    def close(self) -> None:
        """Cleanup ROS resources."""
        if self.sub is not None:
            try:
                self.sub.unregister()
            except Exception:
                pass
            self.sub = None
