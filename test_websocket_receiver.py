"""
Simple WebSocket Receiver Test Script

This script listens for WebSocket connections and prints received messages.
Based on simple_websocket_server.py's server implementation.

Usage:
    # Default settings (0.0.0.0:8000)
    python test_websocket_receiver.py
    
    # Custom port
    python test_websocket_receiver.py --port 9000
    
    # Custom host
    python test_websocket_receiver.py --host 127.0.0.1
    
    # Echo back messages
    python test_websocket_receiver.py --echo
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
logger = logging.getLogger("websocket_receiver")


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


async def handle_client(websocket, echo: bool):
    """
    Handle WebSocket client connection and print received messages.
    
    Args:
        websocket: WebSocket connection object
        echo (bool): Whether to echo messages back to client
    """
    client_address = websocket.remote_address
    logger.info(f"[CONN] Client connected from {client_address}")
    
    try:
        # Send server metadata as first message
        metadata = {"server": "test_websocket_receiver", "echo": echo}
        metadata_bytes = packb(metadata)
        await websocket.send(metadata_bytes)
        logger.info(f"[META] Sent metadata to {client_address}")
        
        message_count = 0
        async for message in websocket:
            try:
                message_count += 1
                
                # Deserialize message using msgpack with NumPy support
                data = unpackb(message)
                
                # Print received message
                logger.info(f"[MSG {message_count}] From {client_address}:")
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, np.ndarray):
                            logger.info(f"  {key}: ndarray shape={value.shape}, dtype={value.dtype}")
                            logger.info(f"       preview: {value.flatten()[:5]}...")
                        else:
                            logger.info(f"  {key}: {value}")
                else:
                    logger.info(f"  Data: {data}")
                
                # Echo back if requested
                if echo:
                    try:
                        response = {
                            "echo": data,
                            "received_count": message_count,
                            "status": "ok"
                        }
                        response_bytes = packb(response)
                        await websocket.send(response_bytes)
                        logger.info(f"[ECHO {message_count}] Sent response to {client_address}")
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to echo: {e}")
                
            except msgpack.exceptions.ExtraData as e:
                logger.error(f"[ERROR] msgpack ExtraData error: {e}")
            except Exception as e:
                logger.error(f"[ERROR] Error processing message: {e}", exc_info=True)
    
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"[DISC] Client {client_address} disconnected normally")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"[ERROR] Client {client_address} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error with client {client_address}: {e}", exc_info=True)
    finally:
        logger.info(f"[CLOSE] Connection closed with {client_address} ({message_count} messages received)")


async def main(host: str = "0.0.0.0", port: int = 8000, echo: bool = False):
    """
    Start the WebSocket receiver.
    
    Args:
        host (str): Host address to bind to
        port (int): Port to listen on
        echo (bool): Whether to echo messages back
    """
    logger.info("=" * 60)
    logger.info("WebSocket Receiver Test Script")
    logger.info("=" * 60)
    logger.info(f"Server address: ws://{host}:{port}")
    logger.info(f"Echo mode: {'enabled' if echo else 'disabled'}")
    logger.info("=" * 60)
    logger.info("Waiting for connections...")
    logger.info("Press Ctrl+C to stop")
    logger.info("")
    
    # Helper to handle HTTP requests
    def process_request(connection, request):
        """Handle HTTP health checks."""
        from websockets.http11 import Response
        from websockets.datastructures import Headers
        
        if request.path == "/healthz":
            logger.info("[HTTP] Health check request received")
            return Response(
                status_code=200,
                reason_phrase="OK",
                headers=Headers([("Content-Type", "text/plain")]),
                body=b"OK\n"
            )
        
        conn_hdr = request.headers.get("Connection", "")
        upgrade_hdr = request.headers.get("Upgrade", "")
        
        if "upgrade" in conn_hdr.lower() and "websocket" in upgrade_hdr.lower():
            return None
        
        logger.warning(f"[HTTP] Non-WebSocket request to {request.path}")
        return Response(
            status_code=426,
            reason_phrase="Upgrade Required",
            headers=Headers([("Content-Type", "text/plain")]),
            body=b"426 Upgrade Required\n"
        )
    
    # Start WebSocket server
    async with websockets.serve(
        lambda ws: handle_client(ws, echo),
        host,
        port,
        process_request=process_request,
        max_size=100 * 1024 * 1024,  # 100 MB max message size
        ping_interval=20,
        ping_timeout=10,
    ):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebSocket message receiver test script"
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
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Echo received messages back to sender"
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
        asyncio.run(main(host=args.host, port=args.port, echo=args.echo))
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
