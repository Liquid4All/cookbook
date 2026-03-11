import json
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.agent import local_client, run_agent
from app.state import home_state

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

conversation_history: list[dict] = []
active_backend: str = "local"


@app.get("/")
def serve_index():
    return FileResponse("index.html")


@app.get("/model")
def get_model():
    global active_backend
    if active_backend == "openai":
        return JSONResponse({"name": "gpt-4o-mini"})
    try:
        models = local_client.models.list()
        name = models.data[0].id.split("_", 2)[-1] if models.data else "unknown"
    except Exception:
        name = "unknown"
    return JSONResponse({"name": name})


@app.get("/backend")
def get_backend():
    return JSONResponse({"backend": active_backend})


class BackendRequest(BaseModel):
    backend: str


@app.post("/backend")
def set_backend(req: BackendRequest):
    global active_backend
    if req.backend not in ("local", "openai"):
        return JSONResponse({"error": "invalid backend"}, status_code=400)
    active_backend = req.backend
    conversation_history.clear()
    return JSONResponse({"backend": active_backend})


@app.get("/state")
def get_state():
    return JSONResponse(home_state)


@app.post("/reset")
def reset():
    conversation_history.clear()
    return JSONResponse({"ok": True})


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    events = []

    def on_tool_call(name, args, result):
        events.append({"name": name, "args": args, "result": result})

    try:
        text = run_agent(req.message, history=conversation_history, backend=active_backend, on_tool_call=on_tool_call)
    except Exception as e:
        return JSONResponse({"text": f"Error: {e}", "tool_calls": events}, status_code=500)

    conversation_history.append({"role": "user",      "content": req.message})
    conversation_history.append({"role": "assistant", "content": text})

    return JSONResponse({"text": text, "tool_calls": events})
