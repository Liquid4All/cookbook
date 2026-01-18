import asyncio
import json
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        # Track pending requests: request_id -> Future
        self.pending_requests: dict[int, asyncio.Future] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, websocket: WebSocket, message: dict):
        await websocket.send_text(json.dumps(message))

    async def handle_websocket_message(self, websocket: WebSocket, data: str):
        """Handle incoming WebSocket message - dispatch to appropriate handler"""
        try:
            message = json.loads(data)

            # Check if this is a response to a pending request
            if "id" in message and message["id"] in self.pending_requests:
                future = self.pending_requests.pop(message["id"])
                if "result" in message:
                    future.set_result(message["result"])
                elif "error" in message:
                    future.set_exception(Exception(f"RPC Error: {message['error']}"))
                else:
                    future.set_result(None)
            else:
                # This is a regular message, echo it back
                await websocket.send_text(data)

        except json.JSONDecodeError:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            await self.send_message(websocket, error_response)

    async def send_rpc_request(
        self,
        websocket: WebSocket,
        method: str,
        params: dict | None = None,
        request_id: int | None = None,
    ) -> Any:
        """Send a JSON-RPC request and wait for response"""
        if params is None:
            params = {}
        if request_id is None:
            # Generate a unique request ID
            request_id = id(params) + int(asyncio.get_event_loop().time() * 1000000)

        # Create a Future to wait for the response
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        try:
            await self.send_message(websocket, message)
            # Wait for the response with timeout
            result = await asyncio.wait_for(future, timeout=2.0)
            return result
        except asyncio.TimeoutError:
            # Clean up the pending request
            self.pending_requests.pop(request_id, None)
            raise Exception("Request timeout")
        except Exception as e:
            # Clean up the pending request
            self.pending_requests.pop(request_id, None)
            raise e
