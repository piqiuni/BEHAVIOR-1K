"""
Simple WebSocket Server for BEHAVIOR Environment

This is a minimal WebSocket server that receives observations from the BEHAVIOR environment
and returns zero actions. It serves as a template for building custom policy servers.

Usage:
    # Start server on default port (8000)
    python simple_websocket_server.py
    
    # Custom port
    python simple_websocket_server.py --port 9000
    
    # Custom host and port
    python simple_websocket_server.py --host 0.0.0.0 --port 8080
    
    # Enable visualization forwarding to ROS visualizer
    python simple_websocket_server.py --viz-host localhost --viz-port 8001

Then run the environment client:
    python OmniGibson/omnigibson/examples/environments/behavior_env_web.py --host localhost --port 8000
"""

import asyncio
import websockets
import msgpack
import numpy as np
import argparse
import logging
import functools
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("websocket_server")


# NumPy array support for msgpack (same as network_utils.py)
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


class VisualizationForwarder:
    """
    Optional forwarder that sends observations and actions to a ROS visualizer server.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8001):
        """
        Initialize the visualization forwarder.
        
        Args:
            host: Host of the visualization server
            port: Port of the visualization server
        """
        self.host = host
        self.port = port
        self._ws = None
        self._connected = False
        self._uri = f"ws://{host}:{port}"
        
    async def connect(self) -> bool:
        """Attempt to connect to the visualization server."""
        try:
            self._ws = await websockets.connect(
                self._uri,
                max_size=100 * 1024 * 1024,
            )
            # Receive metadata
            metadata = unpackb(await self._ws.recv())
            logger.info(f"Connected to visualizer at {self._uri}: {metadata}")
            self._connected = True
            return True
        except Exception as e:
            logger.warning(f"Could not connect to visualizer at {self._uri}: {e}")
            self._connected = False
            return False
    
    async def forward(self, obs: Dict[str, Any], action: np.ndarray) -> None:
        """
        Forward observation and action to the visualizer.
        
        Args:
            obs: Observation dictionary
            action: Action array
        """
        if not self._connected or self._ws is None:
            return
        
        try:
            data = {"obs": obs, "action": action}
            await self._ws.send(packb(data))
            # Receive acknowledgment
            await self._ws.recv()
        except Exception as e:
            logger.debug(f"Failed to forward to visualizer: {e}")
            self._connected = False
    
    async def close(self) -> None:
        """Close the connection to the visualizer."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._connected = False


class SimplePolicy:
    """
    A simple policy that returns zero actions.
    Replace this with your own policy implementation.
    """
    
    def __init__(self, action_dim: int = 23):
        """
        Initialize the policy.
        
        Args:
            action_dim (int): Dimension of action space (23 for R1Pro robot)
        """
        self.action_dim = action_dim
        self.step_count = 0
        logger.info(f"Initialized SimplePolicy with action_dim={action_dim}")
    
    def reset(self) -> None:
        """Reset policy state."""
        self.step_count = 0
        logger.info("Policy reset")
    
    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Compute action from observation.
        
        Args:
            obs (dict): Observation dictionary containing:
                - RGB images from cameras
                - Proprioception data
                - Camera relative poses
                - Task ID
        
        Returns:
            np.ndarray: Action array of shape (action_dim,)
        """
        self.step_count += 1
        
        # Log observation info every 10 steps
        if self.step_count % 10 == 1:
            logger.info(f"Step {self.step_count}: Received observation with keys: {list(obs.keys())}")
            
            # Log some observation details
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif self.step_count == 1:
            logger.info(f"First observation received with keys: {list(obs.keys())}")
        
        # Return zero action (robot will remain stationary)
        action = np.zeros(self.action_dim, dtype=np.float32)
        action = [-1]*self.action_dim  # For testing purposes, return -1 actions
        action = np.array(action, dtype=np.float32)
        
        
        return action


async def handle_client(websocket, policy: SimplePolicy, visualizer: VisualizationForwarder = None):
    """
    Handle WebSocket client connection.
    
    Args:
        websocket: WebSocket connection object
        policy: Policy instance to generate actions
        visualizer: Optional visualization forwarder
    """
    client_address = websocket.remote_address
    logger.info(f"Client connected from {client_address}")
    
    try:
        # Send server metadata as first message (expected by WebsocketClientPolicy)
        metadata = {"action_dim": policy.action_dim, "server": "simple_websocket_server"}
        metadata_bytes = packb(metadata)
        await websocket.send(metadata_bytes)
        logger.info("Sent server metadata to client")
        
        async for message in websocket:
            try:
                # Deserialize observation using msgpack with NumPy support
                obs = unpackb(message)
                print("get msg")
                # Check if this is a reset signal
                if isinstance(obs, dict) and obs.get("reset", False):
                    logger.info("Received reset signal")
                    policy.reset()
                    continue
                
                # Get action from policy
                action = policy.predict(obs)
                logger.debug(f"Generated action: {action.shape}")
                
                # Forward to visualizer if available
                if visualizer is not None:
                    await visualizer.forward(obs, action)
                
                # Prepare response
                response = {"action": action}
                
                # Serialize and send response with NumPy support
                response_bytes = packb(response)
                await websocket.send(response_bytes)
                
            except msgpack.exceptions.ExtraData as e:
                logger.error(f"msgpack ExtraData error: {e}")
                await websocket.send(f"Error: msgpack ExtraData - {str(e)}")
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await websocket.send(f"Error: {str(e)}")
    
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {client_address} disconnected normally")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client {client_address} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error with client {client_address}: {e}", exc_info=True)
    finally:
        logger.info(f"Connection closed with {client_address}")


async def main(host: str = "0.0.0.0", port: int = 8000, action_dim: int = 23, 
               viz_host: str = None, viz_port: int = 8001):
    """
    Start the WebSocket server.
    
    Args:
        host (str): Host address to bind to
        port (int): Port to listen on
        action_dim (int): Action space dimension
        viz_host (str): Optional visualization server host (enables forwarding)
        viz_port (int): Visualization server port
    """
    # Initialize policy
    policy = SimplePolicy(action_dim=action_dim)
    
    # Initialize visualization forwarder if configured
    visualizer = None
    if viz_host is not None:
        visualizer = VisualizationForwarder(host=viz_host, port=viz_port)
        await visualizer.connect()
    
    logger.info("=" * 60)
    logger.info("Starting Simple WebSocket Server for BEHAVIOR Environment")
    logger.info("=" * 60)
    logger.info(f"Server address: ws://{host}:{port}")
    logger.info(f"Action dimension: {action_dim}")
    logger.info(f"Policy: {policy.__class__.__name__} (returns zero actions)")
    if visualizer is not None:
        logger.info(f"Visualization forwarding: ws://{viz_host}:{viz_port}")
    logger.info("=" * 60)
    logger.info("Waiting for client connection...")
    logger.info("")
    
    # Helper to handle HTTP requests (like health checks)
    def process_request(connection, request):
        """
        Handle HTTP requests before WebSocket upgrade.
        - /healthz: Return 200 OK for health checks
        - Other non-WebSocket requests: Return 426 Upgrade Required
        - WebSocket upgrade requests: Allow to proceed (return None)
        """
        from websockets.http11 import Response
        from websockets.datastructures import Headers
        
        # Check if this is a health check endpoint
        if request.path == "/healthz":
            logger.info("Health check request received")
            return Response(
                status_code=200,
                reason_phrase="OK",
                headers=Headers([("Content-Type", "text/plain")]),
                body=b"OK\n"
            )
        
        # Check if this is a WebSocket upgrade request
        conn_hdr = request.headers.get("Connection", "")
        upgrade_hdr = request.headers.get("Upgrade", "")
        
        if "upgrade" in conn_hdr.lower() and "websocket" in upgrade_hdr.lower():
            # This is a valid WebSocket upgrade request, allow it to proceed
            return None
        
        # Non-WebSocket HTTP request - return 426
        logger.warning(f"Non-WebSocket request to {request.path}")
        return Response(
            status_code=426,
            reason_phrase="Upgrade Required",
            headers=Headers([("Content-Type", "text/plain")]),
            body=b"426 Upgrade Required: This endpoint expects a WebSocket connection.\n"
        )

    # Start WebSocket server
    async with websockets.serve(
        lambda ws: handle_client(ws, policy, visualizer),
        host,
        port,
        process_request=process_request,
        max_size=100 * 1024 * 1024,  # 100 MB max message size (for large images)
        ping_interval=20,             # Send ping every 20 seconds
        ping_timeout=10,              # Wait 10 seconds for pong
    ):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple WebSocket server for BEHAVIOR environment control"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0 for all interfaces)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=23,
        help="Action space dimension for R1Pro robot (default: 23)"
    )
    parser.add_argument(
        "--viz-host",
        type=str,
        default=None,
        help="Visualization server host (enables forwarding to ROS visualizer)"
    )
    parser.add_argument(
        "--viz-port",
        type=int,
        default=8001,
        help="Visualization server port (default: 8001)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run server
    try:
        asyncio.run(main(
            host=args.host, 
            port=args.port, 
            action_dim=args.action_dim,
            viz_host=args.viz_host,
            viz_port=args.viz_port
        ))
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
