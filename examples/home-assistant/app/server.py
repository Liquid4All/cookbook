import json
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.agent import client, run_agent
from app.state import home_state

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

conversation_history: list[dict] = []


@app.get("/")
def serve_index():
    return FileResponse("index.html")


@app.get("/model")
def get_model():
    try:
        models = client.models.list()
        name = models.data[0].id.split("_", 2)[-1] if models.data else "unknown"
    except Exception:
        name = "unknown"
    return JSONResponse({"name": name})


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

    text = run_agent(req.message, history=conversation_history, on_tool_call=on_tool_call)

    conversation_history.append({"role": "user",      "content": req.message})
    conversation_history.append({"role": "assistant", "content": text})

    return JSONResponse({"text": text, "tool_calls": events})
