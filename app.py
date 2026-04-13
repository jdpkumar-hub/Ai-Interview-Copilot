# ==========================================
# AI INTERVIEW COPILOT - ADVANCED (AUTO + LIVE + STEALTH)
# ==========================================
# NOTE: sounddevice is NOT used. Browser mic via streamlit-webrtc only.


import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import numpy as np
import av
import tempfile
import os
from faster_whisper import WhisperModel
from openai import OpenAI
import time

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="AI Copilot Advanced", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return WhisperModel("tiny")

model = load_model()

# ==============================
# SESSION STATE
# ==============================
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# ==============================
# AUDIO PROCESSOR
# ==============================
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = []
        self.last_process_time = time.time()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.buffer.append(audio)
        return frame

# ==============================
# TRANSCRIBE
# ==============================
def transcribe(audio_chunks, fs=16000):
    audio_np = np.concatenate(audio_chunks, axis=0)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    from scipy.io.wavfile import write

# Normalize + convert to int16 (VERY IMPORTANT)
    audio_np = audio_np.astype(np.float32)

# Convert stereo → mono if needed
    if len(audio_np.shape) > 1:
        audio_np = np.mean(audio_np, axis=1)

# Normalize
    audio_np = audio_np / np.max(np.abs(audio_np) + 1e-9)

# Convert to int16 (FFmpeg friendly)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    write(temp_file.name, fs, audio_int16)

    segments, _ = model.transcribe(temp_file.name)
    text = " ".join([seg.text for seg in segments]).strip()

    os.remove(temp_file.name)
    return text

# ==============================
# STREAMING AI RESPONSE
# ==============================
def generate_answer_stream(question, mode, placeholder):
    if mode == "DBA":
        system_prompt = "You are a senior Oracle DBA. Give short, sharp interview answers."
    else:
        system_prompt = "You are an interview assistant. Give short, confident answers."

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        stream=True
    )

    full = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            full += chunk.choices[0].delta.content
            placeholder.markdown(f"**🤖 Answer:**\n\n{full}")

# ==============================
# UI
# ==============================

st.title("🕶️ AI Interview Copilot (Stealth Mode)")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("⚙️ Settings")
    mode = st.selectbox("Mode", ["General", "DBA"])
    auto_mode = st.toggle("⚡ Auto Mode (No Click)", True)

with col2:
    st.subheader("🎤 Live Assistant")

    ctx = webrtc_streamer(
    key="auto-mic",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    },
)
    question_placeholder = st.empty()
    answer_placeholder = st.empty()

# ==============================
# AUTO PROCESS LOOP
# ==============================

if ctx.audio_processor and auto_mode:
    audio_processor = ctx.audio_processor

    if len(audio_processor.buffer) > 20:
        audio_chunks = audio_processor.buffer.copy()
        audio_processor.buffer = []

        text = transcribe(audio_chunks)

        if text and text != st.session_state.last_text and len(text) > 5:
            st.session_state.last_text = text

            question_placeholder.markdown(f"**🗣 Interviewer:** {text}")

            generate_answer_stream(text, mode, answer_placeholder)

# ==============================
# STEALTH STYLE
# ==============================

st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
.stButton {display:none;}
h1, h2, h3 {color:#00ffcc;}
</style>
""", unsafe_allow_html=True)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("🕶️ Stealth AI | Auto Listening | Live Answers | Browser Mic")

# ==========================================
# FEATURES INCLUDED
# ==========================================
# ✅ Auto speech detection (no button)
# ✅ Live transcription updates
# ✅ Streaming AI answers
# ✅ Stealth dark UI
# ==========================================
