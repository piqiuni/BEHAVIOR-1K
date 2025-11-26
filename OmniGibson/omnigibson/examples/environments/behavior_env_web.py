"""
WebSocket-Enabled BEHAVIOR Environment Demo

This script demonstrates how to control a BEHAVIOR robot environment via WebSocket communication.
The environment sends observations to a WebSocket server and receives actions in response.

Based on eval.py observation preprocessing and WebsocketPolicy implementation.

Usage:
    # Default settings (localhost:8000, assembling_gift_baskets task)
    python behavior_env_web.py
    
    # Custom WebSocket server
    python behavior_env_web.py --host 192.168.1.100 --port 9000
    
    # Different task
    python behavior_env_web.py --task cleaning_bathrooms
    
    # Headless mode for server deployment
    python behavior_env_web.py --headless
    
    # Quick test (1 iteration only)
    python behavior_env_web.py --short

Observation Structure:
    The preprocessed observations sent to the WebSocket server include:
    - RGB images from 3 cameras (head, left_wrist, right_wrist)
    - Proprioception data (joint positions, velocities, efforts, gripper states)
    - Camera relative poses (position + quaternion for each camera)
    - Task ID (integer index)
    
Expected Action Format:
    The WebSocket server should return actions as:
    {"action": numpy.ndarray}  # Shape: (23,) for R1Pro robot
"""

import os
import argparse
import yaml
import logging
import numpy as np
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.robots.r1pro import R1Pro
from omnigibson.learning.policies import WebsocketPolicy
from omnigibson.learning.utils.eval_utils import (
    flatten_obs_dict,
    ROBOT_CAMERA_NAMES,
    TASK_NAMES_TO_INDICES,
)
from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
import omnigibson.utils.transform_utils as T

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False

# Create logger
logger = logging.getLogger("behavior_env_web")
logger.setLevel(logging.INFO)


def preprocess_obs(obs: dict, robot: R1Pro, task_name: str) -> dict:
    """
    Preprocess the observation dictionary before sending to websocket.
    Based on eval.py's _preprocess_obs method.
    
    Args:
        obs (dict): Raw observation dictionary from environment
        robot (R1Pro): Robot instance
        task_name (str): Name of the current task
        
    Returns:
        dict: Preprocessed observation dictionary ready for network transmission
    """
    # Step 1: Flatten nested observation dictionary
    obs = flatten_obs_dict(obs)
    
    # Step 2: Compute camera relative poses
    base_pose = robot.get_position_orientation()
    cam_rel_poses = []
    
    for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
        camera = robot.sensors[camera_name.split("::")[1]]
        # Get camera parameters from the camera
        direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
        
        if np.allclose(direct_cam_pose, np.zeros(16)):
            # If camera parameters are not ready, use direct pose
            cam_pose = camera.get_position_orientation()
            cam_rel_poses.append(
                th.cat(T.relative_pose_transform(*cam_pose, *base_pose))
            )
        else:
            # Use camera view transform matrix
            cam_pose = T.mat2pose(
                th.tensor(
                    np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T),
                    dtype=th.float32
                )
            )
            cam_rel_poses.append(
                th.cat(T.relative_pose_transform(*cam_pose, *base_pose))
            )
    
    obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
    
    # Step 3: Add task ID
    if task_name in TASK_NAMES_TO_INDICES:
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[task_name]], dtype=th.int64)
    else:
        logger.warning(f"Task name '{task_name}' not found in TASK_NAMES_TO_INDICES, using default ID 0")
        obs["task_id"] = th.tensor([0], dtype=th.int64)
    
    # Step 4: Convert to numpy for network transmission
    obs = torch_to_numpy(obs)
    
    return obs


def main(random_selection=False, headless=False, short_exec=False, host="localhost", port=8000, task_name="assembling_gift_baskets"):
    """
    Generates a BEHAVIOR Task environment with WebSocket control.

    It connects to a WebSocket server, sends observations, receives actions,
    and steps the environment accordingly, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Use pre-sampled cached BEHAVIOR activity scene
    should_sample = False
    
    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["task"]["online_object_sampling"] = should_sample
    cfg["task"]["use_presampled_robot_pose"] = not should_sample
    
    # Update task name if specified
    if task_name is not None:
        cfg["task"]["activity_name"] = task_name
    
    # Get actual task name from config
    actual_task_name = cfg["task"]["activity_name"]
    logger.info(f"Using task: {actual_task_name}")
    
    # Ensure proper observation modalities are enabled
    cfg["robots"][0]["obs_modalities"] = ["depth", "rgb"]
    
    # Load the environment
    env = og.Environment(configs=cfg)
    
    # Get robot instance
    robot: R1Pro = env.robots[0]

    # Move camera to a good position
    og.sim.viewer_camera.set_position_orientation(
        position=[1.6, 6.15, 1.5], orientation=[-0.2322, 0.5895, 0.7199, -0.2835]
    )

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()
    
    # Initialize WebSocket policy
    logger.info(f"Connecting to WebSocket server at {host}:{port}")
    policy = WebsocketPolicy(host=host, port=port)
    logger.info("WebSocket policy initialized successfully")

    # Run a simple loop and reset periodically
    max_iterations = 10 if not short_exec else 1
    
    for j in range(max_iterations):
        og.log.info(f"Resetting environment (iteration {j+1}/{max_iterations})")
        
        # Reset environment and policy
        obs_raw, info = env.reset()
        policy.reset()
        
        # Preprocess initial observation
        obs = preprocess_obs(obs_raw, robot, actual_task_name)
        
        logger.info(f"Starting episode {j+1}")
        
        for i in range(100):
            # Get action from WebSocket policy
            try:
                action = policy.forward(obs=obs)
                action_np = action.numpy() if isinstance(action, th.Tensor) else action
            except Exception as e:
                logger.error(f"Error getting action from policy: {e}")
                logger.warning("Using zero action as fallback")
                action_np = np.zeros(robot.action_space.shape[0])
            
            # Step the environment
            obs_raw, reward, terminated, truncated, info = env.step(action_np)
            
            # Preprocess observation for next step
            obs = preprocess_obs(obs_raw, robot, actual_task_name)
            
            if terminated or truncated:
                og.log.info(f"Episode finished after {i + 1} timesteps")
                if info.get("done", {}).get("success", False):
                    logger.info("✓ Task completed successfully!")
                else:
                    logger.info("✗ Task failed or timed out")
                break

    # Always close the environment at the end
    logger.info("Shutting down environment")
    og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEHAVIOR environment with WebSocket control")
    parser.add_argument("--host", type=str, default="localhost", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket server port")
    parser.add_argument("--task", type=str, default="assembling_gift_baskets", 
                        help="Task name (e.g., assembling_gift_baskets, cleaning_bathrooms)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--short", action="store_true", help="Run only 1 iteration")
    
    args = parser.parse_args()
    
    # Set headless mode
    gm.HEADLESS = args.headless
    
    main(
        headless=args.headless,
        short_exec=args.short,
        host=args.host,
        port=args.port,
        task_name=args.task
    )
