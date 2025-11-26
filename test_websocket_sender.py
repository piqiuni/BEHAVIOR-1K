"""
Simple WebSocket Sender Test Script

This script continuously sends "hello" messages through WebsocketPolicy.
Based on behavior_env_web.py's WebSocket usage pattern.

Usage:
    # Default settings (localhost:8000)
    python test_websocket_sender.py
    
    # Custom server
    python test_websocket_sender.py --host 192.168.1.100 --port 9000
    
    # Custom message
    python test_websocket_sender.py --message "test message"
    
    # Send interval
    python test_websocket_sender.py --interval 2.0
"""

import argparse
import logging
import time
import numpy as np
import asyncio
import websockets
import msgpack
import functools

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("websocket_sender")


# NumPy array support for msgpack
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


async def send_messages(host: str, port: int, message: str, interval: float):
    """
    Connect to WebSocket server and continuously send messages.
    
    Args:
        host (str): WebSocket server host
        port (int): WebSocket server port
        message (str): Message to send
        interval (float): Seconds between messages
    """
    uri = f"ws://{host}:{port}"
    logger.info(f"Connecting to WebSocket server at {uri}")
    
    try:
        async with websockets.connect(uri, max_size=100 * 1024 * 1024) as websocket:
            logger.info("Connected successfully!")
            
            # Receive server metadata (first message)
            try:
                metadata_bytes = await websocket.recv()
                metadata = unpackb(metadata_bytes)
                logger.info(f"Server metadata: {metadata}")
            except Exception as e:
                logger.warning(f"Failed to receive metadata: {e}")
            
            # Send messages continuously
            count = 0
            while True:
                count += 1
                
                # Create message payload (as a dict with "message" key)
                payload = {
                    "message": message,
                    "count": count,
                    "timestamp": time.time()
                }
                
                # Serialize and send
                try:
                    payload_bytes = packb(payload)
                    await websocket.send(payload_bytes)
                    logger.info(f"[{count}] Sent: {message}")
                    
                    # Wait for response (if any)
                    try:
                        response_bytes = await asyncio.wait_for(
                            websocket.recv(), 
                            timeout=interval * 0.9  # Leave some time for sleep
                        )
                        response = unpackb(response_bytes)
                        logger.info(f"[{count}] Received response: {response}")
                    except asyncio.TimeoutError:
                        logger.debug(f"[{count}] No response (timeout)")
                    except Exception as e:
                        logger.debug(f"[{count}] Response error: {e}")
                    
                except Exception as e:
                    logger.error(f"Error sending message: {e}")
                
                # Wait before sending next message
                await asyncio.sleep(interval)
                
    except websockets.exceptions.ConnectionRefused:
        logger.error(f"Connection refused - is the server running at {uri}?")
    except websockets.exceptions.InvalidURI:
        logger.error(f"Invalid WebSocket URI: {uri}")
    except Exception as e:
        logger.error(f"Connection error: {e}", exc_info=True)


async def main(host: str, port: int, message: str, interval: float):
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("WebSocket Sender Test Script")
    logger.info("=" * 60)
    logger.info(f"Target: ws://{host}:{port}")
    logger.info(f"Message: '{message}'")
    logger.info(f"Interval: {interval}s")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop")
    logger.info("")
    
    await send_messages(host, port, message, interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebSocket message sender test script"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="WebSocket server host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="WebSocket server port (default: 8000)"
    )
    parser.add_argument(
        "--message",
        type=str,
        default="hello",
        help="Message to send (default: 'hello')"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between messages (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(
            host=args.host,
            port=args.port,
            message=args.message,
            interval=args.interval
        ))
    except KeyboardInterrupt:
        logger.info("\nStopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
