# mainui.py
import os
import uuid
import json
import base64
import logging
import subprocess
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# ---------- Configuration ----------
RECORDINGS_DIR = "recordings"   # base recordings folder
os.makedirs(RECORDINGS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mainui")

# ---------- FastAPI App ----------
app = FastAPI(title="Meeting App", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Connection Manager ----------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.meeting_metadata: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, meeting_id: str):
        await websocket.accept()
        if meeting_id not in self.active_connections:
            self.active_connections[meeting_id] = []
            self.meeting_metadata[meeting_id] = {
                "created_at": datetime.now().isoformat(),
                "participants": 0
            }
        self.active_connections[meeting_id].append(websocket)
        self.meeting_metadata[meeting_id]["participants"] = len(self.active_connections[meeting_id])
        logger.info(f"✅ Client connected to meeting: {meeting_id}")
        logger.info(f"📊 Total clients in meeting {meeting_id}: {len(self.active_connections[meeting_id])}")

    def disconnect(self, websocket: WebSocket, meeting_id: str):
        if meeting_id in self.active_connections:
            if websocket in self.active_connections[meeting_id]:
                self.active_connections[meeting_id].remove(websocket)
                self.meeting_metadata.get(meeting_id, {})["participants"] = len(self.active_connections.get(meeting_id, []))
                logger.info(f"🔌 Client disconnected from meeting: {meeting_id}")
                logger.info(f"📊 Remaining clients in meeting {meeting_id}: {len(self.active_connections.get(meeting_id, []))}")

                if len(self.active_connections.get(meeting_id, [])) == 0:
                    # keep metadata for a short while or delete - here we delete metadata
                    del self.active_connections[meeting_id]
                    if meeting_id in self.meeting_metadata:
                        del self.meeting_metadata[meeting_id]
                    logger.info(f"🗑️  Meeting {meeting_id} removed (no clients)")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast_to_meeting(self, message: str, meeting_id: str, exclude_websocket: WebSocket = None):
        if meeting_id in self.active_connections:
            disconnected = []
            for conn in list(self.active_connections[meeting_id]):
                if conn != exclude_websocket:
                    try:
                        await conn.send_text(message)
                    except Exception as e:
                        logger.error(f"Error broadcasting to client: {e}")
                        disconnected.append(conn)
            for d in disconnected:
                self.disconnect(d, meeting_id)

    async def broadcast_participant_count(self, meeting_id: str):
        if meeting_id in self.active_connections:
            count = len(self.active_connections[meeting_id])
            message = json.dumps({
                "type": "participant_update",
                "count": count,
                "meeting_id": meeting_id
            })
            await self.broadcast_to_meeting(message, meeting_id)


manager = ConnectionManager()

# ---------- Utility: save base64 chunk to webm file ----------
def ensure_meeting_folder(meeting_id: str) -> str:
    folder = os.path.join(RECORDINGS_DIR, meeting_id)
    os.makedirs(folder, exist_ok=True)
    return folder

def append_base64_to_file(b64data: str, out_path: str):
    """
    Accepts base64 string (possibly a data URL like "data:audio/webm;codecs=opus;base64,AAAA..."),
    strips the prefix if present, decodes, and appends bytes to out_path.
    """
    if b64data.startswith("data:"):
        # split at first comma
        try:
            b64data = b64data.split(",", 1)[1]
        except Exception:
            raise ValueError("Invalid data URL base64")
    binary = base64.b64decode(b64data)
    with open(out_path, "ab") as f:
        f.write(binary)

def convert_webm_to_wav(webm_path: str, wav_path: str) -> bool:
    """
    Convert webm/opus file to WAV using ffmpeg.
    Returns True on success, False on failure.
    """
    # Example ffmpeg command:
    # ffmpeg -y -i audio.webm -ar 16000 -ac 1 -vn audio.wav
    cmd = [
        "ffmpeg",
        "-y",                # overwrite output
        "-i", webm_path,     # input file
        "-ar", "16000",      # set audio sampling rate
        "-ac", "1",          # mono
        "-vn",               # no video
        wav_path
    ]
    try:
        logger.info(f"🔁 Converting {webm_path} -> {wav_path} via ffmpeg")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.error(f"ffmpeg error ({result.returncode}): {result.stderr}")
            return False
        logger.info("✅ Conversion completed")
        return True
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg on the server.")
        return False
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        return False

# ---------- WebSocket endpoint ----------
@app.websocket("/ws/{meeting_id}")
async def websocket_endpoint(websocket: WebSocket, meeting_id: str):
    await manager.connect(websocket, meeting_id)
    meeting_folder = ensure_meeting_folder(meeting_id)
    webm_path = os.path.join(meeting_folder, "audio.webm")   # appended as chunks arrive
    wav_path = os.path.join(meeting_folder, "audio.wav")     # final converted file

    try:
        # send welcome message & broadcast join
        participant_count = len(manager.active_connections.get(meeting_id, []))
        await manager.send_personal_message(json.dumps({
            "type": "system",
            "text": f"✅ Connected to meeting: {meeting_id}",
            "participants": participant_count
        }), websocket)

        await manager.broadcast_to_meeting(json.dumps({
            "type": "system",
            "text": "👤 New participant joined the meeting",
            "participants": participant_count
        }), meeting_id, exclude_websocket=websocket)

        await manager.broadcast_participant_count(meeting_id)

        while True:
            raw = await websocket.receive_text()
            message = json.loads(raw)
            mtype = message.get("type", "unknown")
            logger.info(f"📨 Received {mtype} in meeting {meeting_id}")

            if mtype == "audio_chunk":
                # expected: message['data'] is base64 DataURL or base64 string
                b64 = message.get("data")
                if not b64:
                    logger.warning("Audio chunk without data")
                    continue

                try:
                    append_base64_to_file(b64, webm_path)
                except Exception as e:
                    logger.error(f"Error saving audio chunk: {e}")
                    await manager.send_personal_message(json.dumps({
                        "type": "error",
                        "text": f"Failed to save audio chunk: {str(e)}"
                    }), websocket)
                    continue

                # broadcast a short confirmation or transcript stub
                transcript_text = f"🎤 Speaker: audio chunk received (meeting: {meeting_id})"
                transcript_message = json.dumps({
                    "type": "transcript",
                    "text": transcript_text,
                    "timestamp": message.get("timestamp", datetime.now().isoformat()),
                    "meeting_id": meeting_id
                })
                await manager.broadcast_to_meeting(transcript_message, meeting_id, exclude_websocket=websocket)

                # send confirmation back to sender
                await manager.send_personal_message(json.dumps({
                    "type": "confirmation",
                    "text": "✅ Audio chunk saved",
                    "timestamp": message.get("timestamp", datetime.now().isoformat())
                }), websocket)

            elif mtype == "test":
                participant_count = len(manager.active_connections.get(meeting_id, []))
                await manager.send_personal_message(json.dumps({
                    "type": "test_response",
                    "text": f"✅ WebSocket test successful! {participant_count} participants in meeting",
                    "original_message": message.get("message", ""),
                    "meeting_id": meeting_id,
                    "participants": participant_count
                }), websocket)

            elif mtype == "chat":
                chat_message = json.dumps({
                    "type": "chat",
                    "text": message.get("text", ""),
                    "sender": "User",
                    "timestamp": message.get("timestamp", datetime.now().isoformat()),
                    "meeting_id": meeting_id
                })
                await manager.broadcast_to_meeting(chat_message, meeting_id)

            else:
                logger.warning(f"⚠️ Unknown message type: {mtype}")

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected for meeting: {meeting_id}")
        manager.disconnect(websocket, meeting_id)

        # Convert webm -> wav when the client leaves and there are no participants left
        # If you want conversion to happen per-client disconnect (e.g., finalize after every client leaves),
        # keep this here. Otherwise you could convert on schedule or via background task.
        # If the meeting still has participants, skip conversion until last leaves.
        if meeting_id not in manager.active_connections:
            if os.path.exists(webm_path):
                success = convert_webm_to_wav(webm_path, wav_path)
                if success:
                    logger.info(f"✅ WAV file available at: {wav_path}")
                else:
                    logger.error("❌ Failed to convert webm to wav.")
            else:
                logger.info("No webm file to convert.")

        # notify others
        participant_count = len(manager.active_connections.get(meeting_id, []))
        await manager.broadcast_to_meeting(json.dumps({
            "type": "system",
            "text": "👤 A participant left the meeting",
            "participants": participant_count
        }), meeting_id)

        await manager.broadcast_participant_count(meeting_id)

    except Exception as e:
        logger.error(f"❌ WebSocket error in meeting {meeting_id}: {e}")
        manager.disconnect(websocket, meeting_id)

# ---------- API routes ----------
@app.get("/create-meeting")
async def create_meeting():
    meeting_id = str(uuid.uuid4())[:8]
    logger.info(f"📝 New meeting created: {meeting_id}")
    # create meetings folder
    ensure_meeting_folder(meeting_id)
    return {"meeting_id": meeting_id, "message": "Meeting created successfully", "status": "success", "created_at": datetime.now().isoformat()}

@app.get("/meetings")
async def get_active_meetings():
    meetings_info = {}
    for meeting_id, clients in manager.active_connections.items():
        meetings_info[meeting_id] = {
            "participants": len(clients),
            "created_at": manager.meeting_metadata.get(meeting_id, {}).get('created_at', 'unknown')
        }
    return {"active_meetings": meetings_info, "total_meetings": len(meetings_info), "total_participants": sum(len(c) for c in manager.active_connections.values())}

@app.get("/meeting/{meeting_id}")
async def get_meeting_info(meeting_id: str):
    if meeting_id in manager.active_connections or os.path.isdir(os.path.join(RECORDINGS_DIR, meeting_id)):
        return {
            "meeting_id": meeting_id,
            "participants": len(manager.active_connections.get(meeting_id, [])),
            "created_at": manager.meeting_metadata.get(meeting_id, {}).get('created_at', 'unknown'),
            "status": "active" if meeting_id in manager.active_connections else "inactive"
        }
    else:
        return {"meeting_id": meeting_id, "status": "not_found", "message": "Meeting does not exist or has ended"}

@app.get("/recordings/{meeting_id}/audio.wav")
async def download_wav(meeting_id: str):
    meeting_folder = os.path.join(RECORDINGS_DIR, meeting_id)
    wav_path = os.path.join(meeting_folder, "audio.wav")
    if not os.path.exists(wav_path):
        # If wav doesn't exist but webm exists, try converting now
        webm_path = os.path.join(meeting_folder, "audio.webm")
        if os.path.exists(webm_path):
            success = convert_webm_to_wav(webm_path, wav_path)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to convert recording to WAV")
        else:
            raise HTTPException(status_code=404, detail="Recording not found")
    return FileResponse(wav_path, media_type="audio/wav", filename=f"{meeting_id}.wav")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "meeting-app", "active_meetings": len(manager.active_connections), "total_connections": sum(len(c) for c in manager.active_connections.values()), "timestamp": datetime.now().isoformat()}

# Serve static files (index.html, app.js, etc.)
static_dir = "static"
if os.path.exists(static_dir) and os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    logger.info("✅ Static files directory mounted")
else:
    logger.warning(f"⚠️ Static directory '{static_dir}' not found")
    @app.get("/")
    async def read_root_fallback():
        return {"message": "Meeting App API is running", "status": "Static files not found", "endpoints": { "create_meeting": "/create-meeting", "active_meetings": "/meetings", "health_check": "/health"}}

# ---------- Startup / Shutdown ----------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Meeting App starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Meeting App shutting down...")
    manager.active_connections.clear()
    manager.meeting_metadata.clear()

# ---------- Run server ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mainui:app", host="0.0.0.0", port=5091, reload=True, log_level="info")
