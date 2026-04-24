#!/usr/bin/env python
"""
web/server.py — Servidor de Molinete AI

Ejecutar desde la raíz del proyecto:
    python web/server.py

Abrir en: http://localhost:7860

Dependencias:
    pip install fastapi uvicorn
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("\n❌  Faltan dependencias. Instálalas con:")
    print("       pip install fastapi uvicorn\n")
    sys.exit(1)

ROOT         = Path(__file__).parent.parent
EXAMPLES_DIR = ROOT / "ejemplos"
WEB_DIR      = Path(__file__).parent
DATA_DIR     = ROOT / "data"

app = FastAPI(title="Molinete AI")
app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")


@app.get("/")
async def serve_index():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/checkpoints")
async def list_checkpoints():
    results = []
    if DATA_DIR.exists():
        for sub in sorted(DATA_DIR.iterdir(), reverse=True):
            if sub.is_dir():
                bin_f = sub / "punto_control_mejor.bin"
                tok_f = sub / "tokenizador.json"
                csv_f = sub / "registro_entrenamiento.csv"
                if bin_f.exists():
                    results.append({
                        "name":          sub.name,
                        "path":          str(bin_f),
                        "has_tokenizer": tok_f.exists(),
                        "has_log":       csv_f.exists(),
                        "size_mb":       round(bin_f.stat().st_size / 1e6, 2),
                    })
    return {"checkpoints": results}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    process:     Optional[asyncio.subprocess.Process] = None
    stream_task: Optional[asyncio.Task]               = None

    async def send(data: dict) -> None:
        try:
            await websocket.send_text(json.dumps(data, ensure_ascii=False))
        except Exception:
            pass

    async def stream_output(proc: asyncio.subprocess.Process) -> None:
        try:
            while True:
                chunk = await proc.stdout.read(1024)
                if not chunk:
                    break
                await send({"type": "output", "text": chunk.decode("utf-8", errors="replace")})
            await proc.wait()
            await send({"type": "done", "code": proc.returncode})
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            await send({"type": "error", "message": str(exc)})

    async def kill_running() -> None:
        nonlocal process, stream_task
        if stream_task and not stream_task.done():
            stream_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(stream_task), timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        if process and process.returncode is None:
            try:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                pass
        process     = None
        stream_task = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            action = msg.get("action", "")

            if action == "run":
                await kill_running()

                example_id = str(msg.get("id", ""))
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"]  = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONPATH"] = (
                    str(EXAMPLES_DIR) + os.pathsep + env.get("PYTHONPATH", "")
                )

                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(WEB_DIR / "runner.py"),
                    example_id,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=str(EXAMPLES_DIR),
                    env=env,
                )
                stream_task = asyncio.create_task(stream_output(process))
                await send({"type": "started", "id": example_id})

            elif action == "input":
                if process and process.returncode is None and process.stdin:
                    value = msg.get("value", "")
                    try:
                        process.stdin.write((value + "\n").encode("utf-8"))
                        await process.stdin.drain()
                    except Exception as exc:
                        await send({"type": "error", "message": str(exc)})

            elif action == "stop":
                await kill_running()
                await send({"type": "stopped"})

            elif action == "ping":
                await send({"type": "pong"})

    except WebSocketDisconnect:
        pass
    finally:
        await kill_running()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n  🌬  Molinete AI — Interfaz Web")
    print(f"  → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
