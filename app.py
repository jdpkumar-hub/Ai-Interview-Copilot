# ==========================================
# AI INTERVIEW COPILOT - PRO VERSION
# Real-time + Continuous + Streaming + Better UI
# ==========================================

import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import threading
import tempfile
import os
from faster_whisper import WhisperModel
from openai import OpenAI

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="AI Copilot PRO", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return WhisperModel("base")

model = load_model()

# ==============================
# SESSION STATE
# ==============================
if "running" not in st.session_state:
    st.session_state.running = False

# ==============================
# AUDIO STREAM (REAL-TIME)
# ==============================
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# ==============================
# TRANSCRIBE
# ==============================
def transcribe_audio(audio_np, fs=16000):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    from scipy.io.wavfile import write
    write(temp_file.name, fs, audio_np)

    segments, _ = model.transcribe(temp_file.name)
    text = " ".join([seg.text for seg in segments]).strip()

    os.remove(temp_file.name)
    return text

# ==============================
# AI RESPONSE (STREAMING)
# ==============================
def generate_answer_stream(question, mode="General"):
    if mode == "DBA":
        system_prompt = "You are a senior Oracle DBA with 20 years experience. Give short, powerful interview answers."
    else:
        system_prompt = "You are an expert interview assistant. Give short, confident answers."

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
            yield full

# ==============================
# CONTINUOUS LISTENING LOOP
# ==============================
def listen_loop(answer_placeholder, question_placeholder, mode):
    fs = 16000

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=fs):
        while st.session_state.running:
            audio_data = []

            for _ in range(20):  # ~2 seconds chunks
                audio_data.append(q.get())

            audio_np = np.concatenate(audio_data, axis=0)

            text = transcribe_audio(audio_np, fs)

            if text and len(text) > 5:
                question_placeholder.markdown(f"**🗣 Interviewer:** {text}")

                stream = generate_answer_stream(text, mode)
                for partial in stream:
                    answer_placeholder.markdown(f"**🤖 Answer:**\n\n{partial}")

# ==============================
# UI
# ==============================

st.title("🚀 AI Interview Copilot PRO")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Settings")

    mode = st.selectbox("Mode", ["General", "DBA"])

    if not st.session_state.running:
        if st.button("▶ Start Real-Time Listening"):
            st.session_state.running = True
    else:
        if st.button("⏹ Stop"):
            st.session_state.running = False

with col2:
    st.subheader("🧠 Live Copilot")

    question_placeholder = st.empty()
    answer_placeholder = st.empty()

# ==============================
# RUN THREAD
# ==============================

if st.session_state.running:
    threading.Thread(
        target=listen_loop,
        args=(answer_placeholder, question_placeholder, mode),
        daemon=True
    ).start()

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("🔥 Real-Time AI Copilot | Streaming | Continuous Listening")

# ==========================================
# INCLUDED FEATURES
# ==========================================
# ✅ Real-time listening (no button per question)
# ✅ Continuous transcription
# ✅ Streaming AI answers (word-by-word)
# ✅ DBA specialized mode
# ==========================================

# ==========================================
# NEXT (OPTIONAL)
# ==========================================
# - Zoom/Teams system audio capture
# - Voice answer (earpiece mode)
# - Resume-aware answers
# ==========================================
