#!/usr/bin/env python3
import os
import json
import logging
from collections import defaultdict
import librosa
import soundfile as sf
import torch
from whisperx import load_model
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer

# ---------------- CONFIG ----------------
CHUNK_SECONDS = 30
WHISPER_MODEL = "large-v2"
MIN_WORD_OVERLAP = 0.02
WORD_ASSIGN_TOL = 0.01
MERGE_GAP = 0.6

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("voice2text_pipeline")


def seconds_to_mmss(s: float) -> str:
    m = int(s // 60)
    sec = s - m * 60
    return f"{m:02d}:{sec:05.2f}"


def print_gpu_info():
    logger.info("CUDA available: %s", torch.cuda.is_available())
    try:
        logger.info("GPU count: %d", torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            logger.info("GPU name: %s", torch.cuda.get_device_name(0))
    except Exception:
        pass


def overlap(a_s, a_e, b_s, b_e):
    return max(0.0, min(a_e, b_e) - max(a_s, b_s))


def chunk_audio(input_file, chunk_dir, chunk_seconds):
    logger.info("Loading audio: %s", input_file)
    signal, sr = librosa.load(input_file, sr=None, mono=True)
    duration = librosa.get_duration(y=signal, sr=sr)
    logger.info("Duration: %.2fs | SR=%d", duration, sr)

    os.makedirs(chunk_dir, exist_ok=True)
    chunks = []
    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        chunk_sig = signal[int(start * sr):int(end * sr)]
        chunk_path = os.path.join(chunk_dir, f"chunk_{idx:04d}.wav")
        sf.write(chunk_path, chunk_sig, sr, subtype="PCM_16")
        logger.info("Created chunk %d: %.2f → %.2f", idx, start, end)
        chunks.append((chunk_path, start, end))
        start += chunk_seconds
        idx += 1
    return chunks


def transcribe_with_words(model, wav_path):
    logger.info("Transcribing: %s", wav_path)
    try:
        res = model.transcribe(wav_path, word_timestamps=True)
    except Exception:
        res = model.transcribe(wav_path)

    segments = res.get("segments", [])
    for seg in segments:
        norm = []
        for w in seg.get("words", []) or []:
            w_text = w.get("word") or w.get("text") or ""
            w_start = w.get("start", w.get("ts", None))
            w_end = w.get("end", w.get("te", None))
            if w_start is None or w_end is None:
                continue
            norm.append({"word": w_text.strip(), "start": float(w_start), "end": float(w_end)})
        seg["words"] = norm
    return segments


def create_manifest(audio_path: str, manifest_path: str):
    meta = {"audio_filepath": audio_path, "offset": 0, "duration": None,
            "label": "infer", "text": "-", "num_speakers": None,
            "rttm_filepath": None, "uem_filepath": None}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
        f.write("\n")


def find_rttm_in_dir(root_dir):
    candidates = []
    for root, dirs, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith(".rttm"):
                candidates.append(os.path.join(root, fn))
    for c in candidates:
        if os.path.sep + "pred_rttms" + os.path.sep in c or c.endswith("pred_rttms.rttm"):
            return c
    if candidates:
        return candidates[0]
    return None


def parse_rttm(rttm_path: str):
    segments = []
    if not rttm_path or not os.path.exists(rttm_path):
        logger.warning("RTTM path missing: %s", rttm_path)
        return segments
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            p = line.split()
            if p[0] != "SPEAKER": continue
            try:
                start = float(p[3])
                dur = float(p[4])
                spk = p[7]
                segments.append({"start": start, "duration": dur, "speaker": spk})
            except Exception:
                logger.warning("Failed to parse RTTM line: %s", line)
    return segments


def run_diarization(input_audio, diarizer_conf, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "input_manifest.json")
    create_manifest(input_audio, manifest_path)
    cfg = OmegaConf.load(diarizer_conf)
    cfg.diarizer.manifest_filepath = manifest_path
    cfg.diarizer.out_dir = output_dir
    try:
        cfg.diarizer.msdd_model.parameters.split_infer = True
    except Exception: pass

    logger.info("Loading NeMo ClusteringDiarizer...")
    diarizer = ClusteringDiarizer(cfg=cfg)
    logger.info("Running diarization...")
    diarizer.diarize()

    rttm_file = find_rttm_in_dir(output_dir)
    if rttm_file is None:
        logger.warning("No RTTM found.")
        return []
    diar_segments = parse_rttm(rttm_file)
    return diar_segments


def assign_words_to_speakers(word_list, diar_segments):
    diar_sorted = sorted(diar_segments, key=lambda x: x["start"])
    assigned = []
    for w in word_list:
        w_s, w_e, w_text = float(w["start"]), float(w["end"]), w["word"]
        best_spk, best_o = None, 0.0
        overlap_count = 0
        for d in diar_sorted:
            d_s, d_e = d["start"], d["start"] + d["duration"]
            if d_e + WORD_ASSIGN_TOL < w_s: continue
            if d_s - WORD_ASSIGN_TOL > w_e: break
            o = overlap(w_s, w_e, d_s, d_e)
            if o >= MIN_WORD_OVERLAP: overlap_count += 1
            if o > best_o: best_o = o; best_spk = d["speaker"]
        if best_spk is None: best_spk = "Unknown"
        assigned.append({"speaker": best_spk, "start": w_s, "end": w_e, "word": w_text, "overlap": best_o, "is_overlap_word": overlap_count > 1})
    return assigned


def group_assigned_words(assigned_words):
    from collections import defaultdict
    by_spk = defaultdict(list)
    for w in assigned_words: by_spk[w["speaker"]].append(w)

    lines = []
    for spk, arr in by_spk.items():
        arr_sorted = sorted(arr, key=lambda x: x["start"])
        if not arr_sorted: continue
        cur = {"speaker": spk, "start": arr_sorted[0]["start"], "end": arr_sorted[0]["end"],
               "text": arr_sorted[0]["word"], "overlap": arr_sorted[0]["is_overlap_word"]}
        for w in arr_sorted[1:]:
            if w["start"] - cur["end"] <= MERGE_GAP:
                cur["text"] = cur["text"].rstrip() + " " + w["word"].lstrip()
                cur["end"] = max(cur["end"], w["end"])
                cur["overlap"] = cur["overlap"] or w["is_overlap_word"]
            else:
                lines.append(cur)
                cur = {"speaker": spk, "start": w["start"], "end": w["end"], "text": w["word"], "overlap": w["is_overlap_word"]}
        lines.append(cur)
    return sorted(lines, key=lambda x: x["start"])


def canonicalize_speaker_names(diar_segments):
    totals = defaultdict(float)
    for d in diar_segments: totals[d["speaker"]] += d["duration"]
    sorted_spks = sorted(totals.items(), key=lambda x: -x[1])
    mapping = {}
    for idx, (spk, _) in enumerate(sorted_spks, start=1): mapping[spk] = f"Speaker_{idx}"
    return mapping


def process_audio_file(audio_file, output_dir, diar_config):
    os.makedirs(output_dir, exist_ok=True)
    print_gpu_info()
    chunk_dir = os.path.join(output_dir, "chunks")
    chunks = chunk_audio(audio_file, chunk_dir, CHUNK_SECONDS)
    model = load_model(WHISPER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

    all_asr_segments = []
    for chunk_path, offset, _ in chunks:
        segs = transcribe_with_words(model, chunk_path)
        for s in segs:
            seg_start = float(s.get("start", 0.0)) + offset
            seg_end = float(s.get("end", seg_start)) + offset
            words = [{"word": w["word"], "start": w["start"]+offset, "end": w["end"]+offset} for w in s.get("words", [])]
            all_asr_segments.append({"start": seg_start, "end": seg_end, "text": s.get("text",""), "words": words})

    with open(os.path.join(output_dir, "final_transcript.txt"), "w", encoding="utf-8") as f:
        for s in all_asr_segments: f.write(s["text"]+"\n")

    diar_out_dir = os.path.join(output_dir, "diarization")
    diar_segments = run_diarization(audio_file, diar_config, diar_out_dir)
    spk_map = canonicalize_speaker_names(diar_segments) if diar_segments else {}

    all_assigned_words = []
    debug_entries = []
    for seg in all_asr_segments:
        words = seg.get("words", [])
        if not words:
            a_s, a_e = seg["start"], seg["end"]
            best_spk = "Unknown"
            if diar_segments:
                best_o = 0.0
                for d in diar_segments:
                    o = overlap(a_s, a_e, d["start"], d["start"]+d["duration"])
                    if o > best_o: best_o = o; best_spk = spk_map.get(d["speaker"], d["speaker"])
            all_assigned_words.append({"speaker": best_spk, "start": a_s, "end": a_e, "word": seg["text"], "overlap": 0.0, "is_overlap_word": False})
            debug_entries.append({"segment": (a_s, a_e), "reason": "no words"})
            continue
        assigned = assign_words_to_speakers(words, diar_segments) if diar_segments else []
        for a in assigned: a["speaker"] = spk_map.get(a["speaker"], a["speaker"])
        all_assigned_words.extend(assigned)
        debug_entries.append({"segment": (seg["start"], seg["end"]), "assigned_count": len(assigned)})

    grouped_lines = group_assigned_words(all_assigned_words)

    # Save speaker transcript
    speaker_txt = os.path.join(output_dir, "speaker_transcript.txt")
    with open(speaker_txt, "w", encoding="utf-8") as f:
        for seg in grouped_lines:
            start_s, end_s = seconds_to_mmss(seg["start"]), seconds_to_mmss(seg["end"])
            overlap_tag = " (overlap)" if seg.get("overlap", False) else ""
            f.write(f"[{start_s} - {end_s}] {seg['speaker']}{overlap_tag}: {seg['text']}\n")

    # Save debug info
    debug_file = os.path.join(output_dir, "per_segment_word_assignment_debug.json")
    with open(debug_file, "w", encoding="utf-8") as jf:
        json.dump(debug_entries, jf, indent=2)

    logger.info("Audio processing completed. Outputs in %s", output_dir)
    return speaker_txt


#!/usr/bin/env python3
import os
import requests
import json
from datetime import datetime

# === Config ===
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ai.nexaois.com:4433/mistral")
# Removed OLLAMA_MODEL


def generate_meeting_minutes(transcript_text: str, circulation_date: str) -> str:
    """
    Generates structured Meeting Minutes (MoM) using Mistral model.
    """
    prompt = f"""
You are an expert Technical Meeting Minute Taker.
You are given a full transcript of a technical meeting with speaker labels (e.g., P1, P2, P3).

Your task is to create professional and highly accurate Meeting Minutes (MoM) with the following structure:

1. Meeting Subject / Topic: Extract from the discussion.
2. Date of Meeting: {circulation_date}
3. Attendees: List all participants mentioned in the transcript.
4. Detailed Discussion Points:
   - Include everything said by each participant, keeping speaker labels intact.
   - Preserve all technical details, system integration ideas, API calls, credentials, master data, and configurations.
   - Avoid summarizing important points; do not remove or hallucinate details.
   - Keep the order of discussion as it happened.
5. Decisions Made: Clearly list any concrete decisions, including approvals or rejections.
6. Action Items:
   - Include all action items mentioned.
   - Assign tasks to the responsible participant (use speaker labels if names not mentioned).
   - Include deadlines if mentioned.
7. Notes / Additional Information: Include clarifications, exceptions, or follow-up points.
8. API/Integration Details:
   - Capture all API names, endpoints, request types (GET/POST), data flows, and parameters discussed.
   - Include system data mentions (customer, item, branch) and read/write permissions.

Formatting instructions:
- Use clear headings and bullet points.
- Keep a professional, formal tone.
- Avoid filler words like "uh", "mm", "you know".
- Preserve acronyms, technical terms, and system names exactly.

Transcript:
{transcript_text}

Important:
- Only use information present in the transcript.
- Do not add any assumptions or content not discussed.
- The MoM must be technically accurate and ready to circulate to a project team.
"""

    # 🔥 Model hard-coded (no environment variable)
    payload = {
        "model": "mistral:latest",
        "prompt": prompt,
        "max_tokens": 2500,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_BASE_URL, json=payload)
        response.raise_for_status()

        # Response from your server should be JSON
        data = response.json()

        return data.get("output", "").strip()

    except Exception as e:
        print(f"❌ Error generating MoM: {e}")
        return ""


def create_mom_from_file(transcript_file: str, output_file: str) -> str:
    """
    Reads a speaker transcript file, generates MoM, and saves it to output_file.
    """
    if not os.path.exists(transcript_file):
        raise FileNotFoundError(f"Transcript not found: {transcript_file}")

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    circulation_date = datetime.now().strftime("%Y-%m-%d")
    print("📝 Generating Meeting Minutes...")
    mom_text = generate_meeting_minutes(transcript, circulation_date)

    if mom_text:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(mom_text)
        print(f"✅ MoM saved to: {output_file}")
    else:
        print("❌ No MoM generated. Check transcript or OLLAMA model.")

    return mom_text


if __name__ == "__main__":
    # Test the pipeline
    import sys
    if len(sys.argv) != 2:
        print("Usage: python voice2text_pipeline.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_dir = "test_output"
    diar_config = "nemo_diarization/config.yaml"
    
    result = process_audio_file(audio_file, output_dir, diar_config)
    print(f"Processing complete: {result}")