# voice_io.py
import speech_recognition as sr
from gtts import gTTS
import tempfile
from playsound import playsound
import os

def transcribe_audio(audio_file_path: str) -> str:
    r = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = r.record(source)
    return r.recognize_google(audio_data)

def speak_text(text: str):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        playsound(tmp.name)
        os.remove(tmp.name)