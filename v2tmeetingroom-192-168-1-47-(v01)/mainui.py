# mainui.py - FIXED: Continuous smooth audio recording without chunk loss
import os
import uuid
import json
import logging
import asyncio
import wave
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from collections import defaultdict, deque
import threading

# ---------- Configuration ----------
RECORDINGS_DIR = "recordings"
TRANSCRIPTIONS_DIR = "transcriptions"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)

SAMPLE_RATE = 16000  # samples per second
CHUNK_DURATION_MS = 100  # Process in 100ms chunks for real-time handling
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
MAX_RECENT_AUDIO = 300
MAX_USER_RECORDINGS_KEEP = 50

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mainui")

# ---------- Improved Audio Processing Utilities ----------
class AudioProcessor:
    @staticmethod
    def validate_audio_chunk(audio_data: list) -> bool:
        """Validate incoming audio chunk"""
        if not audio_data or not isinstance(audio_data, list):
            return False
        
        # Check for reasonable length
        if len(audio_data) < 100 or len(audio_data) > 10000:
            logger.warning(f"⚠️ Audio chunk length suspicious: {len(audio_data)}")
            return False
            
        # Check for valid float values
        try:
            audio_array = np.array(audio_data, dtype=np.float32)
            if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
                logger.warning("⚠️ Audio chunk contains NaN or Inf values")
                return False
                
            # Check for reasonable amplitude range
            max_val = np.max(np.abs(audio_array))
            if max_val > 10.0:  # Unusually loud
                logger.warning(f"⚠️ Audio chunk has unusually high amplitude: {max_val}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Audio chunk validation failed: {e}")
            return False
            
        return True

    @staticmethod
    def normalize_audio(audio_data: list) -> np.ndarray:
        """Normalize audio data to consistent format"""
        audio_array = np.array(audio_data, dtype=np.float32)
        
        # Remove DC offset
        audio_array = audio_array - np.mean(audio_array)
        
        # Gentle normalization (avoid over-compression)
        peak = np.max(np.abs(audio_array))
        if peak > 0:
            # Normalize to -1.0 to 1.0 range but don't over-compress quiet audio
            if peak > 1.0:
                audio_array = audio_array / peak
            # Don't amplify quiet audio too much to avoid noise
            elif peak < 0.1:
                audio_array = audio_array * (0.1 / peak) if peak > 0.01 else audio_array
        
        return audio_array

    @staticmethod
    def convert_to_int16(audio_array: np.ndarray) -> np.ndarray:
        """Convert float32 array to int16 for WAV file"""
        # Clip to safe range first
        audio_array = np.clip(audio_array, -1.0, 1.0)
        # Scale to int16 range
        return (audio_array * 32767).astype(np.int16)

# ---------- Continuous Audio Buffer ----------
class ContinuousAudioBuffer:
    """Maintains a continuous buffer of audio data for smooth recording"""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.is_recording = False
        self.start_time = None
        
    def start(self):
        """Start recording"""
        with self.buffer_lock:
            self.audio_buffer = []
            self.is_recording = True
            self.start_time = datetime.now()
            
    def stop(self):
        """Stop recording and return all audio data"""
        with self.buffer_lock:
            self.is_recording = False
            audio_data = self.audio_buffer.copy()
            self.audio_buffer = []
            return audio_data
            
    def add_chunk(self, audio_chunk: list):
        """Add audio chunk to buffer with validation"""
        if not self.is_recording:
            return
            
        if not AudioProcessor.validate_audio_chunk(audio_chunk):
            return
            
        try:
            normalized_audio = AudioProcessor.normalize_audio(audio_chunk)
            with self.buffer_lock:
                self.audio_buffer.extend(normalized_audio.tolist())
        except Exception as e:
            logger.error(f"❌ Error adding audio chunk to buffer: {e}")
            
    def get_duration(self) -> float:
        """Get current recording duration in seconds"""
        with self.buffer_lock:
            return len(self.audio_buffer) / self.sample_rate
            
    def get_sample_count(self) -> int:
        """Get total number of samples in buffer"""
        with self.buffer_lock:
            return len(self.audio_buffer)

# ---------- Transcription Manager ----------
class TranscriptionManager:
    def __init__(self):
        self.processing_meetings = set()

    async def process_meeting_audio(self, meeting_id: str, audio_file_path: str):
        if meeting_id in self.processing_meetings:
            logger.info(f"⚠️ Meeting {meeting_id} is already being processed")
            return None
        self.processing_meetings.add(meeting_id)

        try:
            logger.info(f"🎯 Starting transcription for meeting {meeting_id}")

            output_dir = os.path.join(TRANSCRIPTIONS_DIR, f"meeting_{meeting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(output_dir, exist_ok=True)

            result = await self.run_transcription_pipeline(audio_file_path, output_dir, meeting_id)
            if result:
                logger.info(f"✅ Transcription completed for meeting {meeting_id}")
                return result
            else:
                logger.error(f"❌ Transcription failed for meeting {meeting_id}")
                return None
        except Exception as e:
            logger.error(f"❌ Error in transcription process: {e}")
            return None
        finally:
            self.processing_meetings.discard(meeting_id)

    async def run_transcription_pipeline(self, audio_file: str, output_dir: str, meeting_id: str):
        try:
            # Try to use advanced pipeline (user-provided)
            from voice2text_pipeline import process_audio_file, create_mom_from_file  # optional
            diar_config = "nemo_diarization/config.yaml"
            speaker_transcript_file = process_audio_file(audio_file, output_dir, diar_config)

            if speaker_transcript_file and os.path.exists(speaker_transcript_file):
                mom_file = os.path.join(output_dir, "meeting_minutes.md")
                mom_content = create_mom_from_file(speaker_transcript_file, mom_file)
                with open(speaker_transcript_file, 'r', encoding='utf-8') as f:
                    transcript_content = f.read()

                return {
                    'transcript_file': speaker_transcript_file,
                    'mom_file': mom_file,
                    'transcript_content': transcript_content,
                    'mom_content': mom_content,
                    'output_dir': output_dir
                }
            else:
                logger.error("❌ Advanced transcription failed - using fallback")
                return await self.fallback_transcription(audio_file, output_dir, meeting_id)
        except Exception as e:
            logger.info(f"ℹ️ Advanced pipeline unavailable or failed: {e} — using fallback")
            return await self.fallback_transcription(audio_file, output_dir, meeting_id)

    async def fallback_transcription(self, audio_file: str, output_dir: str, meeting_id: str):
        try:
            import whisper
            import torch

            logger.info("🔄 Using fallback Whisper transcription")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🎛️ Whisper device set to {device}")

            model = whisper.load_model("base", device=device)
            result = model.transcribe(audio_file)

            transcript_content = result.get("text", "")
            transcript_file = os.path.join(output_dir, "transcript.txt")
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(transcript_content)

            mom_content = self.generate_simple_mom(transcript_content)
            mom_file = os.path.join(output_dir, "meeting_minutes.md")
            with open(mom_file, 'w', encoding='utf-8') as f:
                f.write(mom_content)

            return {
                'transcript_file': transcript_file,
                'mom_file': mom_file,
                'transcript_content': transcript_content,
                'mom_content': mom_content,
                'output_dir': output_dir
            }
        except Exception as e:
            logger.error(f"❌ Fallback transcription failed: {e}")
            return None

    def generate_simple_mom(self, transcript: str) -> str:
        return f"""# Meeting Minutes
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Transcript
{transcript}

## Summary
Automatically generated meeting summary.

## Action Items
- Review transcript
- Manually extract action items if required
"""

# ---------- Fixed Audio Recording Manager ----------
class AudioRecordingManager:
    def __init__(self):
        self.continuous_buffers: Dict[str, ContinuousAudioBuffer] = {}
        self.user_buffers: Dict[str, ContinuousAudioBuffer] = {}
        self.is_recording: Dict[str, bool] = {}
        self.recording_start_time: Dict[str, datetime] = {}
        self.transcription_manager = TranscriptionManager()
        self.audio_processor = AudioProcessor()

    def start_recording(self, meeting_id: str):
        """Start continuous recording for a meeting"""
        self.continuous_buffers[meeting_id] = ContinuousAudioBuffer(SAMPLE_RATE)
        self.continuous_buffers[meeting_id].start()
        self.is_recording[meeting_id] = True
        self.recording_start_time[meeting_id] = datetime.now()
        logger.info(f"🎙️ Started continuous recording for meeting: {meeting_id}")

    def stop_recording(self, meeting_id: str) -> Optional[str]:
        """Stop recording and save smooth continuous WAV file"""
        if meeting_id in self.continuous_buffers and self.is_recording.get(meeting_id, False):
            self.is_recording[meeting_id] = False

            # Get all continuous audio data
            continuous_buffer = self.continuous_buffers[meeting_id]
            all_audio = continuous_buffer.stop()
            
            duration = len(all_audio) / SAMPLE_RATE
            logger.info(f"📊 Continuous recording: {len(all_audio)} samples ({duration:.2f}s)")

            if not all_audio:
                logger.warning(f"❌ No audio data for meeting {meeting_id}")
                return None

            # Save as high-quality WAV
            filename = f"meeting_{meeting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            filepath = os.path.join(RECORDINGS_DIR, filename)

            self.save_high_quality_wav(all_audio, filepath, sample_rate=SAMPLE_RATE)
            logger.info(f"💾 Saved continuous recording: {filepath} ({len(all_audio)} samples, {duration:.2f}s)")

            # Cleanup
            if meeting_id in self.continuous_buffers:
                del self.continuous_buffers[meeting_id]
            if meeting_id in self.recording_start_time:
                del self.recording_start_time[meeting_id]

            return filepath
        return None

    def add_audio_chunk(self, meeting_id: str, user_id: str, user_name: str, audio_data: list):
        """Add audio chunk to continuous buffer"""
        if not self.is_recording.get(meeting_id, False):
            return

        # Validate audio chunk
        if not self.audio_processor.validate_audio_chunk(audio_data):
            logger.warning(f"⚠️ Invalid audio chunk from user {user_name} in meeting {meeting_id}")
            return

        # Add to continuous meeting buffer
        if meeting_id in self.continuous_buffers:
            self.continuous_buffers[meeting_id].add_chunk(audio_data)

        # Also maintain per-user buffer for individual recordings
        if user_id not in self.user_buffers:
            self.user_buffers[user_id] = ContinuousAudioBuffer(SAMPLE_RATE)
            self.user_buffers[user_id].start()
        
        self.user_buffers[user_id].add_chunk(audio_data)

    def save_high_quality_wav(self, audio_data: list, filename: str, sample_rate: int = SAMPLE_RATE):
        """Save continuous audio with high quality settings"""
        try:
            if len(audio_data) == 0:
                logger.warning("Attempted to save empty audio data")
                return

            # Convert to numpy array and process
            audio_array = np.array(audio_data, dtype=np.float32)
            
            # Apply gentle compression to reduce clipping
            audio_array = np.tanh(audio_array * 0.8)  # Soft clipping
            
            # Convert to int16 for WAV
            int16_audio = self.audio_processor.convert_to_int16(audio_array)

            # Save with WAV best practices
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.setcomptype('NONE', 'not compressed')  # No compression
                wav_file.writeframes(int16_audio.tobytes())

            file_size = os.path.getsize(filename)
            duration = len(audio_data) / sample_rate
            logger.info(f"✅ Continuous WAV saved: {filename} ({file_size/1024:.1f}KB, {duration:.1f}s)")

        except Exception as e:
            logger.error(f"❌ Error saving continuous WAV: {e}")

    def get_user_recording_file(self, user_id: str, meeting_id: str):
        """Get individual user recording from continuous buffer"""
        if user_id in self.user_buffers:
            user_buffer = self.user_buffers[user_id]
            user_audio = user_buffer.stop()  # Get all continuous audio
            
            if user_audio:
                filename = f"user_{user_id}_meeting_{meeting_id}.wav"
                filepath = os.path.join(RECORDINGS_DIR, filename)
                self.save_high_quality_wav(user_audio, filepath, sample_rate=SAMPLE_RATE)
                
                # Restart buffer for future recordings
                self.user_buffers[user_id].start()
                return filepath
        return None

    def get_meeting_recordings(self, meeting_id: str):
        recording_files = []
        if os.path.exists(RECORDINGS_DIR):
            for filename in os.listdir(RECORDINGS_DIR):
                if filename.startswith(f"meeting_{meeting_id}"):
                    filepath = os.path.join(RECORDINGS_DIR, filename)
                    try:
                        created_ts = os.path.getctime(filepath)
                    except Exception:
                        created_ts = datetime.now().timestamp()
                    recording_files.append({
                        'filename': filename,
                        'filepath': filepath,
                        'size': os.path.getsize(filepath),
                        'created_at': created_ts
                    })
        recording_files.sort(key=lambda x: x['created_at'], reverse=True)
        return recording_files

    async def stop_recording_and_transcribe(self, meeting_id: str, background_tasks: BackgroundTasks = None):
        recording_file = self.stop_recording(meeting_id)
        if recording_file and os.path.exists(recording_file):
            logger.info(f"🎯 Starting transcription for meeting {meeting_id}")
            if background_tasks:
                background_tasks.add_task(self.transcription_manager.process_meeting_audio, meeting_id, recording_file)
            else:
                asyncio.create_task(self.transcription_manager.process_meeting_audio(meeting_id, recording_file))
            return {
                "recording_file": recording_file,
                "transcription_started": True,
                "message": "Recording stopped and transcription started"
            }
        return {
            "recording_file": None,
            "transcription_started": False,
            "message": "Recording stopped but no file found for transcription"
        }

    def get_transcription_status(self, meeting_id: str):
        import glob
        transcription_pattern = os.path.join(TRANSCRIPTIONS_DIR, f"meeting_{meeting_id}_*")
        matches = glob.glob(transcription_pattern)

        if matches:
            latest_dir = max(matches, key=os.path.getctime)
            transcript_file = os.path.join(latest_dir, "speaker_transcript.txt")
            fallback_transcript = os.path.join(latest_dir, "transcript.txt")
            mom_file = os.path.join(latest_dir, "meeting_minutes.md")

            transcript_exists = os.path.exists(transcript_file) or os.path.exists(fallback_transcript)
            actual_transcript_file = transcript_file if os.path.exists(transcript_file) else fallback_transcript

            status = {
                "transcription_dir": latest_dir,
                "transcript_exists": transcript_exists,
                "transcript_file": actual_transcript_file,
                "mom_exists": os.path.exists(mom_file),
                "mom_file": mom_file,
                "is_processing": meeting_id in self.transcription_manager.processing_meetings
            }

            if status["transcript_exists"]:
                try:
                    with open(actual_transcript_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        status["transcript_preview"] = content[:500] + "..." if len(content) > 500 else content
                        status["transcript_length"] = len(content)
                except Exception as e:
                    status["transcript_preview"] = f"Unable to read transcript: {e}"
                    status["transcript_length"] = 0

            if status["mom_exists"]:
                try:
                    with open(mom_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        status["mom_preview"] = content[:500] + "..." if len(content) > 500 else content
                        status["mom_length"] = len(content)
                except Exception as e:
                    status["mom_preview"] = f"Unable to read meeting minutes: {e}"
                    status["mom_length"] = 0

            return status
        else:
            return {
                "transcript_exists": False,
                "mom_exists": False,
                "is_processing": meeting_id in self.transcription_manager.processing_meetings
            }

# ---------- Data Models ----------
class MeetingCreate(BaseModel):
    title: Optional[str] = "Untitled Meeting"
    host_name: Optional[str] = "Anonymous"

class RecordingRequest(BaseModel):
    meeting_id: str
    user_id: Optional[str] = None

# ---------- Connection Manager ----------
class ConnectionManager:
    def __init__(self):
        # meeting_id -> user_id -> websocket
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.meeting_metadata: Dict[str, Dict] = {}
        self.meeting_status: Dict[str, bool] = {}
        self.user_data: Dict[str, Dict] = {}
        self.audio_buffers: Dict[str, List] = {}
        self.audio_recorder = AudioRecordingManager()

        # For live echo - recent audio chunks per meeting
        self.recent_audio: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_RECENT_AUDIO))
        self.echo_enabled: Dict[str, bool] = {}

    async def connect(self, websocket: WebSocket, meeting_id: str, user_name: str = "Anonymous"):
        await websocket.accept()
        user_id = str(uuid.uuid4())[:8]

        if meeting_id not in self.active_connections:
            self.active_connections[meeting_id] = {}
            self.meeting_metadata[meeting_id] = {
                "created_at": datetime.now().isoformat(),
                "participants": 0,
                "title": f"Meeting {meeting_id}",
                "host": user_name
            }
            self.meeting_status[meeting_id] = False
            self.echo_enabled[meeting_id] = True

        self.active_connections[meeting_id][user_id] = websocket
        self.meeting_metadata[meeting_id]["participants"] = len(self.active_connections[meeting_id])

        self.user_data[user_id] = {
            "websocket": websocket,
            "meeting_id": meeting_id,
            "user_name": user_name,
            "joined_at": datetime.now().isoformat(),
            "is_speaking": False
        }

        self.audio_buffers[user_id] = []

        logger.info(f"✅ {user_name} connected to meeting: {meeting_id} (uid={user_id})")
        return user_id

    def disconnect(self, websocket: WebSocket, meeting_id: str, user_id: str):
        # Cleanly remove user from active connections and free memory
        try:
            if meeting_id in self.active_connections and user_id in self.active_connections[meeting_id]:
                try:
                    # Close websocket if still open
                    ws = self.active_connections[meeting_id][user_id]
                    # do not call ws.close() here (async) — rely on WebSocketDisconnect flow to close
                except Exception:
                    pass

                del self.active_connections[meeting_id][user_id]
                self.meeting_metadata[meeting_id]["participants"] = len(self.active_connections.get(meeting_id, {}))

                user_name = self.user_data.get(user_id, {}).get('user_name', 'Unknown')
                logger.info(f"🔌 {user_name} disconnected from meeting: {meeting_id} (uid={user_id})")

            if user_id in self.user_data:
                del self.user_data[user_id]

            if user_id in self.audio_buffers:
                del self.audio_buffers[user_id]

            # If no participants left, cleanup meeting-level state and save recording if active
            if meeting_id in self.active_connections and len(self.active_connections[meeting_id]) == 0:
                # Save recording when last user leaves
                if self.audio_recorder.is_recording.get(meeting_id, False):
                    recording_file = self.audio_recorder.stop_recording(meeting_id)
                    if recording_file:
                        logger.info(f"💾 Auto-saved continuous recording: {recording_file}")

                # remove meeting maps
                if meeting_id in self.active_connections:
                    del self.active_connections[meeting_id]
                if meeting_id in self.meeting_metadata:
                    del self.meeting_metadata[meeting_id]
                if meeting_id in self.meeting_status:
                    del self.meeting_status[meeting_id]
                if meeting_id in self.recent_audio:
                    del self.recent_audio[meeting_id]
                if meeting_id in self.echo_enabled:
                    del self.echo_enabled[meeting_id]
                logger.info(f"🗑️ Meeting {meeting_id} fully removed")
        except Exception as e:
            logger.error(f"Error during disconnect cleanup: {e}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast_to_meeting(self, message: str, meeting_id: str, exclude_user_id: str = None):
        if meeting_id in self.active_connections:
            disconnected = []
            for user_id, conn in list(self.active_connections[meeting_id].items()):
                if user_id != exclude_user_id:
                    try:
                        await conn.send_text(message)
                    except Exception as e:
                        logger.error(f"Error broadcasting to client {user_id}: {e}")
                        disconnected.append(user_id)

            # cleanup disconnected
            for user_id in disconnected:
                try:
                    conn = self.active_connections[meeting_id].get(user_id)
                    self.disconnect(conn, meeting_id, user_id)
                except Exception:
                    pass

    async def broadcast_audio_to_meeting(self, audio_data: list, meeting_id: str, sender_user_id: str, user_name: str):
        """Broadcast audio with continuous recording"""
        if meeting_id not in self.active_connections:
            return

        # Validate audio before processing
        if not AudioProcessor.validate_audio_chunk(audio_data):
            logger.warning(f"⚠️ Invalid audio chunk from {user_name}, skipping")
            return

        # Add to continuous recorder
        if self.audio_recorder.is_recording.get(meeting_id, False):
            self.audio_recorder.add_audio_chunk(meeting_id, sender_user_id, user_name, audio_data)

        # Store for echo buffer
        self.recent_audio[meeting_id].append({
            'data': audio_data,
            'user_id': sender_user_id,
            'user_name': user_name,
            'timestamp': datetime.now()
        })

        # Broadcast to others
        message = json.dumps({
            "type": "audio_data",
            "data": audio_data,
            "timestamp": datetime.now().isoformat(),
            "meeting_id": meeting_id,
            "user_id": sender_user_id,
            "user_name": user_name,
            "is_speaking": True
        })
        await self.broadcast_to_meeting(message, meeting_id, exclude_user_id=sender_user_id)

    async def send_audio_echo(self, websocket: WebSocket, meeting_id: str, user_id: str):
        """Send recent audio as echo to a new participant (helps them get context)."""
        if (meeting_id in self.recent_audio and self.recent_audio[meeting_id]
                and self.echo_enabled.get(meeting_id, True)):
            recent_chunks = list(self.recent_audio[meeting_id])[-3:]
            for chunk in recent_chunks:
                if chunk['user_id'] != user_id:
                    echo_message = json.dumps({
                        "type": "audio_echo",
                        "data": chunk['data'],
                        "original_speaker": chunk['user_name'],
                        "timestamp": chunk['timestamp'].isoformat()
                    })
                    try:
                        await websocket.send_text(echo_message)
                    except Exception as e:
                        logger.error(f"Error sending audio echo: {e}")

    async def broadcast_participant_count(self, meeting_id: str):
        if meeting_id in self.active_connections:
            count = len(self.active_connections[meeting_id])
            participants = self.get_meeting_participants(meeting_id)
            message = json.dumps({
                "type": "participant_update",
                "count": count,
                "participants": participants,
                "meeting_id": meeting_id
            })
            await self.broadcast_to_meeting(message, meeting_id)

    def get_meeting_participants(self, meeting_id: str) -> List[Dict]:
        participants = []
        for user_id, data in self.user_data.items():
            if data.get('meeting_id') == meeting_id:
                participants.append({
                    "user_id": user_id,
                    "user_name": data.get('user_name', 'Anonymous'),
                    "is_speaking": data.get('is_speaking', False)
                })
        return participants

    def start_meeting(self, meeting_id: str):
        self.meeting_status[meeting_id] = True
        self.audio_recorder.start_recording(meeting_id)
        logger.info(f"🎯 Meeting {meeting_id} started with continuous recording")

    async def stop_meeting(self, meeting_id: str, background_tasks: BackgroundTasks = None):
        self.meeting_status[meeting_id] = False
        result = await self.audio_recorder.stop_recording_and_transcribe(meeting_id, background_tasks)
        logger.info(f"🛑 Meeting {meeting_id} stopped - {result['message']}")
        return result

    def is_meeting_active(self, meeting_id: str) -> bool:
        return self.meeting_status.get(meeting_id, False)

    def update_user_speaking_status(self, user_id: str, is_speaking: bool):
        if user_id in self.user_data:
            self.user_data[user_id]['is_speaking'] = is_speaking

    def toggle_echo(self, meeting_id: str) -> bool:
        if meeting_id in self.echo_enabled:
            self.echo_enabled[meeting_id] = not self.echo_enabled[meeting_id]
            return self.echo_enabled[meeting_id]
        return False

# ---------- FastAPI App ----------
app = FastAPI(
    title="Meeting App",
    version="4.2.0",
    description="Real-time audio meeting application with continuous smooth audio recording"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()

# ---------- WebSocket endpoint ----------
@app.websocket("/ws/{meeting_id}")
async def websocket_endpoint(websocket: WebSocket, meeting_id: str):
    user_name = websocket.query_params.get("user_name", "Anonymous")
    user_id = await manager.connect(websocket, meeting_id, user_name)

    try:
        meeting_active = manager.is_meeting_active(meeting_id)
        participant_count = len(manager.active_connections.get(meeting_id, {}))
        participants = manager.get_meeting_participants(meeting_id)

        if meeting_active:
            await manager.send_audio_echo(websocket, meeting_id, user_id)

        # Send welcome to connected user
        await manager.send_personal_message(json.dumps({
            "type": "system",
            "text": f"✅ Connected to meeting as {user_name}",
            "participants": participant_count,
            "participant_list": participants,
            "meeting_active": meeting_active,
            "user_id": user_id,
            "recording": manager.audio_recorder.is_recording.get(meeting_id, False)
        }), websocket)

        # announce join
        await manager.broadcast_to_meeting(json.dumps({
            "type": "system",
            "text": f"👤 {user_name} joined the meeting",
            "participants": participant_count,
            "participant_list": participants
        }), meeting_id, exclude_user_id=user_id)

        await manager.broadcast_participant_count(meeting_id)

        while True:
            raw = await websocket.receive_text()
            message = json.loads(raw)
            mtype = message.get("type", "unknown")

            if mtype == "audio_data":
                # audio_data expected as list of floats; client should send small chunks
                if not manager.is_meeting_active(meeting_id):
                    await manager.send_personal_message(json.dumps({
                        "type": "error",
                        "text": "Meeting is not active"
                    }), websocket)
                    continue

                audio_chunk = message.get("data", [])
                # Enhanced audio validation
                if not AudioProcessor.validate_audio_chunk(audio_chunk):
                    logger.warning(f"⚠️ Invalid audio chunk from {user_name}, skipping")
                    continue

                await manager.broadcast_audio_to_meeting(audio_chunk, meeting_id, user_id, user_name)
                manager.update_user_speaking_status(user_id, True)

            elif mtype == "stop_speaking":
                manager.update_user_speaking_status(user_id, False)
                stop_message = json.dumps({
                    "type": "stop_speaking",
                    "user_id": user_id,
                    "user_name": user_name,
                    "meeting_id": meeting_id
                })
                await manager.broadcast_to_meeting(stop_message, meeting_id, exclude_user_id=user_id)

            elif mtype == "mic_on":
                # UI may send mic_on to indicate readiness
                manager.update_user_speaking_status(user_id, True)
                await manager.broadcast_to_meeting(json.dumps({
                    "type": "mic_status",
                    "user_id": user_id,
                    "user_name": user_name,
                    "mic": "on"
                }), meeting_id, exclude_user_id=user_id)

            elif mtype == "mic_off":
                manager.update_user_speaking_status(user_id, False)
                await manager.broadcast_to_meeting(json.dumps({
                    "type": "mic_status",
                    "user_id": user_id,
                    "user_name": user_name,
                    "mic": "off"
                }), meeting_id, exclude_user_id=user_id)

            elif mtype == "start_meeting":
                manager.start_meeting(meeting_id)
                await manager.broadcast_to_meeting(json.dumps({
                    "type": "meeting_status",
                    "status": "started",
                    "text": f"🎯 Meeting started by {user_name}! Continuous recording started...",
                    "meeting_id": meeting_id,
                    "recording": True
                }), meeting_id)
                logger.info(f"🎯 Meeting {meeting_id} started by {user_name}")

            elif mtype == "stop_meeting":
                # schedule transcription in background (non-blocking)
                result = await manager.stop_meeting(meeting_id)
                stop_message = {
                    "type": "meeting_status",
                    "status": "stopped",
                    "text": f"🛑 Meeting stopped by {user_name}. Transcription started...",
                    "meeting_id": meeting_id,
                    "recording": False,
                    "transcription_started": result["transcription_started"]
                }
                await manager.broadcast_to_meeting(json.dumps(stop_message), meeting_id)
                logger.info(f"🛑 Meeting {meeting_id} stopped by {user_name}, transcription: {result['transcription_started']}")

            elif mtype == "test":
                participant_count = len(manager.active_connections.get(meeting_id, {}))
                participants = manager.get_meeting_participants(meeting_id)
                await manager.send_personal_message(json.dumps({
                    "type": "test_response",
                    "text": f"✅ Test successful! {participant_count} participants",
                    "meeting_id": meeting_id,
                    "participants": participant_count,
                    "participant_list": participants,
                    "meeting_active": manager.is_meeting_active(meeting_id),
                    "recording": manager.audio_recorder.is_recording.get(meeting_id, False)
                }), websocket)

            elif mtype == "chat":
                chat_message = json.dumps({
                    "type": "chat",
                    "text": message.get("text", ""),
                    "sender": user_name,
                    "timestamp": message.get("timestamp", datetime.now().isoformat()),
                    "meeting_id": meeting_id
                })
                await manager.broadcast_to_meeting(chat_message, meeting_id, exclude_user_id=user_id)

            elif mtype == "raise_hand":
                hand_message = json.dumps({
                    "type": "raise_hand",
                    "user_id": user_id,
                    "user_name": user_name,
                    "meeting_id": meeting_id
                })
                await manager.broadcast_to_meeting(hand_message, meeting_id, exclude_user_id=user_id)

            elif mtype == "toggle_echo":
                echo_status = manager.toggle_echo(meeting_id)
                await manager.send_personal_message(json.dumps({
                    "type": "echo_status",
                    "enabled": echo_status,
                    "text": f"Echo feature {'enabled' if echo_status else 'disabled'}"
                }), websocket)

            elif mtype == "get_transcription_status":
                status = manager.audio_recorder.get_transcription_status(meeting_id)
                await manager.send_personal_message(json.dumps({
                    "type": "transcription_status",
                    "status": status
                }), websocket)

            else:
                await manager.send_personal_message(json.dumps({
                    "type": "error",
                    "text": f"Unknown message type: {mtype}"
                }), websocket)

    except WebSocketDisconnect:
        logger.info(f"🔌 {user_name} disconnected from meeting: {meeting_id}")
        manager.disconnect(websocket, meeting_id, user_id)

        participant_count = len(manager.active_connections.get(meeting_id, {}))
        participants = manager.get_meeting_participants(meeting_id)

        await manager.broadcast_to_meeting(json.dumps({
            "type": "system",
            "text": f"👤 {user_name} left the meeting",
            "participants": participant_count,
            "participant_list": participants
        }), meeting_id, exclude_user_id=user_id)
        await manager.broadcast_participant_count(meeting_id)

    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        # ensure cleanup
        try:
            manager.disconnect(websocket, meeting_id, user_id)
        except Exception:
            pass

# ---------- API routes ----------
@app.post("/create-meeting")
async def create_meeting(meeting_data: MeetingCreate):
    meeting_id = str(uuid.uuid4())[:8]
    logger.info(f"📝 New meeting: {meeting_id} by {meeting_data.host_name}")
    return {
        "meeting_id": meeting_id,
        "title": meeting_data.title,
        "host_name": meeting_data.host_name,
        "message": "Meeting created successfully",
        "status": "success",
        "created_at": datetime.now().isoformat()
    }

@app.post("/meeting/{meeting_id}/start")
async def start_meeting(meeting_id: str):
    manager.start_meeting(meeting_id)
    return {
        "meeting_id": meeting_id,
        "status": "started",
        "message": "Meeting started successfully with continuous recording",
        "recording": True
    }

@app.post("/meeting/{meeting_id}/stop")
async def stop_meeting(meeting_id: str, background_tasks: BackgroundTasks):
    result = await manager.stop_meeting(meeting_id, background_tasks)
    response = {
        "meeting_id": meeting_id,
        "status": "stopped",
        "message": result["message"],
        "recording": False,
        "transcription_started": result["transcription_started"]
    }
    if result.get("recording_file"):
        response["recording_file"] = result["recording_file"]
    return response

@app.get("/meeting/{meeting_id}/status")
async def get_meeting_status(meeting_id: str):
    participants = manager.get_meeting_participants(meeting_id)
    return {
        "meeting_id": meeting_id,
        "active": manager.is_meeting_active(meeting_id),
        "participants": len(manager.active_connections.get(meeting_id, {})),
        "participant_list": participants,
        "recording": manager.audio_recorder.is_recording.get(meeting_id, False),
        "echo_enabled": manager.echo_enabled.get(meeting_id, True)
    }

@app.get("/recordings/{meeting_id}")
async def get_meeting_recordings(meeting_id: str):
    recording_files = manager.audio_recorder.get_meeting_recordings(meeting_id)
    return {
        "meeting_id": meeting_id,
        "recordings": recording_files,
        "count": len(recording_files)
    }

@app.post("/download-recording")
async def download_recording(request: RecordingRequest):
    if request.user_id:
        filepath = manager.audio_recorder.get_user_recording_file(request.user_id, request.meeting_id)
    else:
        recordings = manager.audio_recorder.get_meeting_recordings(request.meeting_id)
        filepath = recordings[0]['filepath'] if recordings else None

    if filepath and os.path.exists(filepath):
        return FileResponse(path=filepath, filename=os.path.basename(filepath), media_type='audio/wav')
    else:
        raise HTTPException(status_code=404, detail="Recording not found")

@app.post("/meeting/{meeting_id}/toggle-echo")
async def toggle_echo(meeting_id: str):
    echo_status = manager.toggle_echo(meeting_id)
    return {
        "meeting_id": meeting_id,
        "echo_enabled": echo_status,
        "message": f"Echo feature {'enabled' if echo_status else 'disabled'}"
    }

@app.get("/meetings")
async def get_active_meetings():
    meetings_info = {}
    for meeting_id, clients in manager.active_connections.items():
        participants = manager.get_meeting_participants(meeting_id)
        meetings_info[meeting_id] = {
            "participants": len(clients),
            "participant_list": participants,
            "active": manager.is_meeting_active(meeting_id),
            "recording": manager.audio_recorder.is_recording.get(meeting_id, False),
            "echo_enabled": manager.echo_enabled.get(meeting_id, True),
            "title": manager.meeting_metadata.get(meeting_id, {}).get('title', 'Untitled Meeting')
        }
    return {
        "active_meetings": meetings_info,
        "total_meetings": len(meetings_info),
        "total_participants": sum(len(c) for c in manager.active_connections.values())
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "meeting-app",
        "active_meetings": len(manager.active_connections),
        "total_connections": sum(len(c) for c in manager.active_connections.values()),
        "timestamp": datetime.now().isoformat()
    }

# ---------- Transcription Routes ----------
@app.post("/meeting/{meeting_id}/transcribe")
async def transcribe_meeting(meeting_id: str, background_tasks: BackgroundTasks):
    recording_files = manager.audio_recorder.get_meeting_recordings(meeting_id)
    if not recording_files:
        raise HTTPException(status_code=404, detail="No recording found for this meeting")

    latest_recording = recording_files[0]  # get newest (we sorted descending)
    result = await manager.audio_recorder.transcription_manager.process_meeting_audio(meeting_id, latest_recording['filepath'])
    if result:
        return {
            "meeting_id": meeting_id,
            "status": "transcription_started",
            "transcript_file": result['transcript_file'],
            "mom_file": result['mom_file'],
            "output_dir": result['output_dir']
        }
    else:
        raise HTTPException(status_code=500, detail="Transcription failed")

@app.get("/meeting/{meeting_id}/transcription-status")
async def get_transcription_status(meeting_id: str):
    status = manager.audio_recorder.get_transcription_status(meeting_id)
    return {"meeting_id": meeting_id, "status": status}

@app.get("/meeting/{meeting_id}/transcript")
async def get_meeting_transcript(meeting_id: str):
    status = manager.audio_recorder.get_transcription_status(meeting_id)
    if not status["transcript_exists"]:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return FileResponse(path=status["transcript_file"], filename=f"meeting_{meeting_id}_transcript.txt", media_type='text/plain')

@app.get("/meeting/{meeting_id}/meeting-minutes")
async def get_meeting_minutes(meeting_id: str):
    status = manager.audio_recorder.get_transcription_status(meeting_id)
    if not status["mom_exists"]:
        raise HTTPException(status_code=404, detail="Meeting minutes not found")
    return FileResponse(path=status["mom_file"], filename=f"meeting_{meeting_id}_minutes.md", media_type='text/markdown')

@app.post("/meeting/{meeting_id}/stop-with-transcription")
async def stop_meeting_with_transcription(meeting_id: str, background_tasks: BackgroundTasks):
    result = await manager.stop_meeting(meeting_id, background_tasks)
    return {
        "meeting_id": meeting_id,
        "status": "stopped",
        "transcription_started": result["transcription_started"],
        "recording_file": result.get("recording_file"),
        "message": result["message"]
    }

# Serve static files
static_dir = "static"
if os.path.exists(static_dir) and os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    logger.info("✅ Static files directory mounted")
else:
    os.makedirs(static_dir, exist_ok=True)
    logger.info("✅ Created static directory")

@app.get("/")
async def read_root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Meeting app backend running. Add a frontend in /static to connect via WebSocket."}

@app.get("/favicon.ico")
async def favicon():
    icon_path = os.path.join(static_dir, "favicon.ico")
    if os.path.exists(icon_path):
        return FileResponse(icon_path)
    raise HTTPException(status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mainui:app", host="0.0.0.0", port=5050, reload=True, log_level="info")