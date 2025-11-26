import os

import yaml

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import choose_from_options

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False

import numpy as np
import threading
import time
from omnigibson.robots.r1pro import R1Pro
import torch as th
from typing import Optional

import torch as th
import omnigibson.utils.transform_utils as T
from omnigibson.learning.utils.eval_utils import (
    flatten_obs_dict,
    ROBOT_CAMERA_NAMES,
    TASK_NAMES_TO_INDICES,
)


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

# All ROS action handling is implemented inside RosPolicy below (no global callbacks)

def _preprocess_obs(obs: dict, robot: R1Pro, cfg=None) -> dict:
    """
    Preprocess the observation dictionary similarly to Evaluator._preprocess_obs.
    This flattens obs and appends `robot_r1::cam_rel_poses` (concatenated poses for each camera).

    Args:
        obs (dict): raw env observation
        robot (R1Pro): robot instance to query for camera poses
        cfg: optional config object; if provided and contains task name, `task_id` will be appended

    Returns:
        dict: flattened obs with added keys
    """
    return obs
    obs = flatten_obs_dict(obs)
    base_pose = robot.get_position_orientation()
    cam_rel_poses = []
    for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
        camera = robot.sensors[camera_name.split("::")[1]]
        direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
        if np.allclose(direct_cam_pose, np.zeros(16)):
            cam_rel_poses.append(
                th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
            )
        else:
            cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
            cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
    obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
    # append task id to obs when possible
    if cfg is not None and hasattr(cfg, "task") and hasattr(cfg.task, "name"):
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[cfg.task.name]], dtype=th.int64)
    return obs


def to_cv_img(tensor_or_arr, is_depth=False):
    """Convert torch tensor or numpy array from the environment to a cv2-style numpy image.

    - Depth images are returned as float32 2D arrays.
    - RGB images are returned as uint8 HxWx3 arrays.
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

    # If channel-first, transpose
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # If image has 4 channels (e.g., RGBA/BGRA), drop the alpha channel
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # If grayscale provided where RGB expected, replicate channels
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # Ensure contiguous array for cv_bridge
    arr = np.ascontiguousarray(arr)

    return arr


class RosPolicy:
    """
    ROS-backed policy that provides actions from a ROS topic or zeros if unavailable.

    Usage mirrors LocalPolicy: `act(obs)` / `forward(obs)` return a torch Tensor of shape `(action_dim,)`.
    """

    def __init__(self, *args, action_dim: Optional[int] = 23, topic: str = "/behavior/control", queue_size: int = 1, **kwargs) -> None:
        self.policy = None
        self.action_dim = int(action_dim) if action_dim is not None else None
        self._latest = np.zeros(self.action_dim, dtype=np.float32) if self.action_dim is not None else None
        self._lock = threading.Lock()
        self.topic = topic
        self.sub = None

        if rospy is not None and Float32MultiArray is not None:
            try:
                try:
                    if not rospy.core.is_initialized():
                        rospy.init_node("ros_policy_node", anonymous=True)
                except Exception:
                    # some rospy builds don't expose core.is_initialized
                    pass
                self.sub = rospy.Subscriber(self.topic, Float32MultiArray, self._cb, queue_size=queue_size)
                og.log.info(f"RosPolicy subscribed to {self.topic}")
            except Exception as e:
                og.log.warn(f"Failed to subscribe to ROS topic {self.topic}: {e}")
                self.sub = None
        else:
            og.log.info("ROS not available; RosPolicy will output zero actions.")

        # create cv bridge and publishers when available
        self.bridge = CvBridge() if CvBridge is not None else None
        self.pubs = {}
        self.left_link_name = None
        self.right_link_name = None
        self.zed_link_name = None
        self.rate_hz = kwargs.get("rate_hz", 10)
        self._last_pub_shapes = None
        # event to signal arrival of a new action
        self._new_action_event = threading.Event()

    def setup_publishers_for_robot(self, robot: R1Pro):
        """Initialize fixed publishers for the robot's camera topics.

        Uses fixed topic namespace (no robot name injected), e.g.:
        `/robot/left_realsense/rgb` etc.
        """
        if rospy is None or Image is None:
            return
        # fixed topic names (no robot name variable)
        self.pubs = {
            "left_rgb": rospy.Publisher("/robot/left_realsense/rgb", Image, queue_size=1),
            "left_depth": rospy.Publisher("/robot/left_realsense/depth", Image, queue_size=1),
            "right_rgb": rospy.Publisher("/robot/right_realsense/rgb", Image, queue_size=1),
            "right_depth": rospy.Publisher("/robot/right_realsense/depth", Image, queue_size=1),
            "zed_rgb": rospy.Publisher("/robot/zed/rgb", Image, queue_size=1),
            "zed_depth": rospy.Publisher("/robot/zed/depth", Image, queue_size=1),
        }
        # keep camera link keys for compatibility (not used when publishing from preprocessed obs)
        self.left_link_name = "left_realsense_link:Camera:0"
        self.right_link_name = "right_realsense_link:Camera:0"
        self.zed_link_name = "zed_link:Camera:0"

    def publish_state(self, state: dict, robot: R1Pro) -> None:
        """Extract camera images from environment state and publish as ROS Image messages.

        This is non-blocking and will silently skip if rospy or CvBridge is not available.
        """
        if rospy is None or self.bridge is None or not self.pubs:
            return

        # The `state` argument here is expected to be the preprocessed obs dict
        # (flattened) produced by Evaluator._preprocess_obs. Search for keys containing
        # the camera link substrings and 'rgb'/'depth' indicators.

        

        try:
            obs = state
            robot_obs = obs.get(robot.name)
            
            left_rgb = robot_obs.get(f"{robot.name}:{self.left_link_name}").get("rgb")
            left_depth = robot_obs.get(f"{robot.name}:{self.left_link_name}").get("depth")
            right_rgb = robot_obs.get(f"{robot.name}:{self.right_link_name}").get("rgb")
            right_depth = robot_obs.get(f"{robot.name}:{self.right_link_name}").get("depth")
            zed_rgb = robot_obs.get(f"{robot.name}:{self.zed_link_name}").get("rgb")
            zed_depth = robot_obs.get(f"{robot.name}:{self.zed_link_name}").get("depth")
            
            

            if left_rgb is None and left_depth is None and right_rgb is None and right_depth is None and zed_rgb is None and zed_depth is None:
                # nothing found
                return

            # convert found arrays to cv images when present
            left_rgb_img = to_cv_img(left_rgb, is_depth=False) if left_rgb is not None else None
            left_depth_img = to_cv_img(left_depth, is_depth=True) if left_depth is not None else None
            right_rgb_img = to_cv_img(right_rgb, is_depth=False) if right_rgb is not None else None
            right_depth_img = to_cv_img(right_depth, is_depth=True) if right_depth is not None else None
            zed_rgb_img = to_cv_img(zed_rgb, is_depth=False) if zed_rgb is not None else None
            zed_depth_img = to_cv_img(zed_depth, is_depth=True) if zed_depth is not None else None

            # publish available images
            try:
                if left_rgb_img is not None:
                    self.pubs["left_rgb"].publish(self.bridge.cv2_to_imgmsg(left_rgb_img, encoding="rgb8"))
                if left_depth_img is not None:
                    self.pubs["left_depth"].publish(self.bridge.cv2_to_imgmsg(left_depth_img, encoding="32FC1"))
                if right_rgb_img is not None:
                    self.pubs["right_rgb"].publish(self.bridge.cv2_to_imgmsg(right_rgb_img, encoding="rgb8"))
                if right_depth_img is not None:
                    self.pubs["right_depth"].publish(self.bridge.cv2_to_imgmsg(right_depth_img, encoding="32FC1"))
                if zed_rgb_img is not None:
                    self.pubs["zed_rgb"].publish(self.bridge.cv2_to_imgmsg(zed_rgb_img, encoding="rgb8"))
                if zed_depth_img is not None:
                    self.pubs["zed_depth"].publish(self.bridge.cv2_to_imgmsg(zed_depth_img, encoding="32FC1"))
            except Exception as e:
                og.log.debug(f"RosPolicy failed to publish images: {e}")
        except Exception as e:
            og.log.debug(f"RosPolicy failed to extract/publish sensor images: {e}")

    def get_action(self) -> np.ndarray:
        """Return the latest action as a numpy array (float32)."""
        with self._lock:
            if self._latest is None:
                return np.zeros(self.action_dim, dtype=np.float32)
            return self._latest.copy()


    def _cb(self, msg) -> None:
        try:
            data = np.array(msg.data, dtype=np.float32)
        except Exception:
            return
        if self.action_dim is None:
            return
        # pad or truncate
        if data.size != self.action_dim:
            if data.size < self.action_dim:
                pad = np.zeros(self.action_dim - data.size, dtype=np.float32)
                data = np.concatenate([data, pad])
            else:
                data = data[: self.action_dim]
        with self._lock:
            self._latest = data.copy()
        # signal that a new action has arrived
        try:
            self._new_action_event.set()
        except Exception:
            pass

    def act(self, obs: dict) -> th.Tensor:
        return self.forward(obs)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """Publish `obs` (if provided) and wait for an action to arrive from ROS.

        Call signature: `forward(obs=<state dict>, robot=<R1Pro>, timeout=seconds)`
        - If `robot` is provided and `obs` contains camera data, images are published.
        - The function blocks until a new ROS action message arrives or `timeout` elapses.
        - Returns a torch tensor of shape `(action_dim,)`.
        """
        # If an inner policy is set, delegate
        if self.policy is not None:
            return self.policy.act(obs).detach().cpu()
        # print(obs)
        timeout = kwargs.get("timeout", 1.0)
        robot = kwargs.get("robot", None)

        # attempt to publish images extracted from obs/state if possible
        if obs is not None and robot is not None:
            try:
                # If obs is the env state dict (mapping robot.name -> sensors), reuse publish_state
                self.publish_state(obs, robot)
            except Exception:
                pass

        # wait for a new action to arrive (clear event first)
        try:
            self._new_action_event.clear()
        except Exception:
            pass

        got = False
        try:
            got = self._new_action_event.wait(timeout=timeout)
        except Exception:
            got = False

        # return latest action (or zeros on timeout)
        with self._lock:
            if self._latest is None:
                vec = np.zeros(self.action_dim, dtype=np.float32)
            else:
                vec = self._latest.copy()

        if not got:
            og.log.debug("RosPolicy.forward: timeout waiting for action; returning latest/zeros")

        return th.from_numpy(vec).to(dtype=th.float32)

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        if self.action_dim is not None:
            with self._lock:
                self._latest = np.zeros(self.action_dim, dtype=np.float32)


def main(random_selection=False, headless=False, short_exec=False):
    """
    Generates a BEHAVIOR Task environment in an online fashion.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    print(f"当前脚本路径 __file__: {__file__}")
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)
    # Ask the user whether they want online object sampling or not
    sampling_options = {
        False: "Use a pre-sampled cached BEHAVIOR activity scene",
        True: "Sample the BEHAVIOR activity in an online fashion",
    }
    # should_sample = choose_from_options(
    #     options=sampling_options, name="online object sampling", random_selection=random_selection
    # )
    should_sample = False
    

    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["task"]["online_object_sampling"] = should_sample
    cfg["task"]["use_presampled_robot_pose"] = not should_sample

    # Load the environment
    env = og.Environment(configs=cfg)

    # Move camera to a good position
    og.sim.viewer_camera.set_position_orientation(
        position=[1.6, 6.15, 1.5], orientation=[-0.2322, 0.5895, 0.7199, -0.2835]
    )

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Run a ROS-enabled loop that publishes sensor images and accepts control commands
    max_iterations = 10 if not short_exec else 1
    robot: R1Pro = env.robots[0]
    
    # Instantiate RosPolicy (handles ROS node/subscriber/publishers internally)
    policy = RosPolicy(action_dim=23, topic="/behavior/control", rate_hz=1000)
    # Setup fixed publishers (no robot name) — use fixed `/robot/...` topics
    try:
        policy.setup_publishers_for_robot(robot)  # function now uses fixed topic names
    except Exception:
        pass
    # Use either rospy.Rate or fallback sleep
    if rospy is not None:
        try:
            rate_hz = rospy.get_param("~rate", policy.rate_hz)
            rate = rospy.Rate(rate_hz)
        except Exception:
            rate = None
    else:
        rate = None
    
    # Main loop: reset and then continuously publish and step
    for j in range(max_iterations):
        og.log.info("Resetting environment")
        reset_ret = env.reset()
        # env.reset() may return obs or (obs, info); handle both
        if isinstance(reset_ret, tuple) or isinstance(reset_ret, list):
            obs_state = reset_ret[0]
        else:
            obs_state = reset_ret

        step_count = 0
        while True:
            # If ROS is present, allow shutdown via rospy; otherwise run until episode ends
            if rospy is not None and rospy.is_shutdown():
                break

            # get action from policy by calling forward(obs=obs_state, robot=robot)
            try:
                robot_action_tensor = policy.forward(obs=obs_state, robot=robot, timeout=1.0)
            except Exception:
                # fallback to latest (non-blocking) behavior
                robot_action_tensor = policy.forward(obs=None, robot=None, timeout=0.0)

            action = robot_action_tensor.detach().cpu().numpy() * 0.1

            # step environment
            state, reward, terminated, truncated, info = env.step(action)
            state["task"] = env.task.get_obs(env)
            
            print(f"Step {step_count}: action={action}")
            # next obs_state is the returned state (for publishing in next forward)
            obs_state = _preprocess_obs(state, robot)

            step_count += 1
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(step_count))
                break

            # sleep to respect desired rate if rospy Rate not used
            if rate is not None:
                try:
                    rate.sleep()
                except Exception:
                    time.sleep(1.0 / max(1.0, policy.rate_hz))
            else:
                time.sleep(1.0 / max(1.0, policy.rate_hz))

    # Always close the environment at the end
    og.shutdown()


if __name__ == "__main__":
    main()
