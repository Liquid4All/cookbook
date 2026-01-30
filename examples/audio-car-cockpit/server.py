import base64
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from openai import AsyncOpenAI

from src.checklist import create_checklist_router
from src.connection_manager import ConnectionManager
from src.functions import create_functions_router
from src.llamacpp_inference import ToolCallingRuntime, function_to_args, spawn_server
from src.settings import p_env


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Setting up...")

    # Prepare inference runtimes
    with (
        spawn_server(file_name="LiquidAI/LFM2-1.2B-Tool-GGUF:Q8_0") as (_, port_lm),
    ):
        app.state.tcr = ToolCallingRuntime(port=port_lm)

        _url = p_env.DEMO_URL.unicode_string()
        print(f"Ready, opening: {_url}")
        webbrowser.open(_url, new=0, autoraise=True)
        yield


# Initialize FastAPI app and connection manager
app = FastAPI(lifespan=lifespan, title="Cockpit demo")
manager = ConnectionManager()

# Static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)


# Static file endpoints
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse((Path("static") / "favicon.ico"))


@app.get("/style.css", include_in_schema=False)
async def style():
    return FileResponse((Path("static") / "style.css"))


@app.get("/script.js", include_in_schema=False)
async def script():
    return FileResponse((Path("static") / "script.js"))


@app.get("/shader-logo.js", include_in_schema=False)
async def shader_logo():
    return FileResponse((Path("static") / "shader-logo.js"))


@app.get("/")
async def get_index():
    index_path = static_dir / "index.html"
    return HTMLResponse(index_path.read_text())


# Tool calling example endpoints
@app.get("/toolcall/single/{query}")
async def tool_calling_single_turn(query: str):
    tcr: ToolCallingRuntime = app.state.tcr

    tool_call, text = tcr.completion(query)

    if tool_call is not None:
        func_name, args = function_to_args(tool_call)

        if not manager.active_connections:
            print("No active cockpit connections")
        else:
            ws = manager.active_connections[0]
            result = await manager.send_rpc_request(ws, func_name, args)
            print(f"Function call result:\n{result}")

    return JSONResponse(content={"tool_call": tool_call, "text": text})


# Include routers
app.include_router(create_functions_router(manager))
app.include_router(create_checklist_router(manager))


# WebSocket endpoint for cockpit control
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.handle_websocket_message(websocket, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# WebSocket endpoint for audio (STT/TTS)
@app.websocket("/ws-audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_client = AsyncOpenAI(base_url=f"http://127.0.0.1:{p_env.AUDIO_SERVER_PORT}/v1", api_key="dummy")

    voice = "US female"

    try:
        while True:
            data = await websocket.receive_json()
            mode = data.get("mode", "asr")
            text = data.get("text")
            audio_b64 = data.get("audio")

            wav_data = base64.b64decode(audio_b64) if audio_b64 else None

            # Build messages based on mode
            if mode == "asr":
                print("\n[AUDIO] Starting ASR (Speech-to-Text)...")
                if wav_data is None:
                    continue
                messages = [
                    {"role": "system", "content": "Perform ASR."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64.b64encode(wav_data).decode("utf-8"),
                                    "format": "wav",
                                },
                            }
                        ],
                    },
                ]
            else:  # tts
                voice = data.get("voice", None) or voice
                print(f"\n[AUDIO] Starting TTS (Text-to-Speech) with voice '{voice}': '{text}'")
                messages = [
                    {
                        "role": "system",
                        "content": f"Perform TTS. Use the {voice} voice.",
                    },
                    {"role": "user", "content": text},
                ]

            # Stream response
            stream = await audio_client.chat.completions.create(
                model="",
                messages=messages,
                stream=True,
                max_tokens=512,
            )

            transcribed_text = ""

            async for chunk in stream:
                delta = chunk.choices[0].delta

                if delta.content:
                    _text_content = delta.content  # .rstrip("<|im_end|>")
                    transcribed_text += _text_content
                    await websocket.send_json({"type": "text", "data": _text_content})

                if hasattr(delta, "audio_chunk") and delta.audio_chunk:
                    chunk_data = delta.audio_chunk["data"]
                    # Send audio chunk immediately for low latency
                    await websocket.send_json({"type": "audio", "data": chunk_data, "sample_rate": 24000})

            # If ASR mode, process through tool calling and then TTS
            if mode == "asr" and transcribed_text:
                print(f"[AUDIO] Transcribed: {transcribed_text}")

                # Send User caption
                await websocket.send_json({"type": "caption", "role": "driver", "text": transcribed_text})

                # Process through tool calling runtime
                print("[AUDIO] Processing through tool calling model...")
                tcr: ToolCallingRuntime = app.state.tcr
                tool_call, response_text = tcr.completion(transcribed_text)

                formatted_tool_name = None
                tool_call_valid = True

                # Execute function call if present
                if tool_call is not None:
                    print(f"[AUDIO] Tool call detected: {tool_call}")
                    formatted_tool_name = tool_call
                    try:
                        func_name, args = function_to_args(tool_call)
                        formatted_tool_name = func_name

                        if not manager.active_connections:
                            print("[AUDIO] No active cockpit connections")
                            response_text = "Sorry, the cockpit is not connected."
                        else:
                            ws = manager.active_connections[0]
                            result = await manager.send_rpc_request(ws, func_name, args)
                            if result is not None:
                                tool_call_valid = result
                            print(f"[AUDIO] Function call result: {result}")
                    except Exception as e:
                        print(f"[AUDIO] Function call error: {e}")
                        # Override response with error message
                        response_text = f"Sorry, the model called the non-existing function: {tool_call}"
                        tool_call_valid = False
                else:
                    print("[AUDIO] No tool call detected")

                # Send caption
                await websocket.send_json(
                    {
                        "type": "caption",
                        "role": "model",
                        "text": response_text,
                        "tool": formatted_tool_name,
                        "tool_valid": tool_call_valid,
                    }
                )

                # Send the response text to TTS
                voice = data.get("voice", None) or voice
                print(f"[AUDIO] Sending to TTS with voice '{voice}': '{response_text}'")
                tts_messages = [
                    {
                        "role": "system",
                        "content": f"Perform TTS. Use the {voice} voice.",
                    },
                    {"role": "user", "content": response_text},
                ]

                tts_stream = await audio_client.chat.completions.create(
                    model="",
                    messages=tts_messages,
                    stream=True,
                    max_tokens=512,
                )

                async for chunk in tts_stream:
                    delta = chunk.choices[0].delta

                    if hasattr(delta, "audio_chunk") and delta.audio_chunk:
                        chunk_data = delta.audio_chunk["data"]
                        # Send audio chunk immediately for low latency
                        await websocket.send_json({"type": "audio", "data": chunk_data, "sample_rate": 24000})

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[AUDIO] Error: {e}")
        await websocket.send_json({"type": "error", "data": str(e)})


if __name__ == "__main__":
    import uvicorn

    host = p_env.DEMO_URL.unicode_host()
    port = p_env.DEMO_URL.port
    assert host is not None, f"Please set a valid host: {p_env.DEMO_URL=}"
    assert port is not None, f"Please set a valid port: {p_env.DEMO_URL=}"

    uvicorn.run(app, host=host, port=port)
