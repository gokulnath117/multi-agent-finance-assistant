import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
from scipy.io.wavfile import write
from voice_io import transcribe_audio
from agent_logic import run_query

import pyttsx3

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

st.title("🎤 Voice Finance Assistant")
st.write("Click the button and speak into your mic.")

duration = 5  # seconds
fs = 16000  # sampling rate

if st.button("🎙️ Record"):
    st.write("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        write(f.name, fs, recording)
        audio_path = f.name

    st.write("Transcribing...")
    query = transcribe_audio(audio_path)
    st.write("You said:", query)

    with st.spinner("Processing with agent..."):
        response = run_query(query)
    
    st.success("Answer:")
    st.write(response)

    speak_text(response)