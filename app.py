# Purpose: Detect emotions based on 

from st_audiorec import st_audiorec
import streamlit as st
import io, soundfile as sf
import numpy as np
import librosa

# Set the title of the app
st.title("Voicy: Voice Emotion Detection")

# Brief description of what the app does
st.write("Voicy is an AI-powered voice emotion detection. It takes a speech recording as an input and classifies the emotion of the person talking. ")

# Get audio from user
st.subheader("🎤 Record your voice")
wav_audio = st_audiorec() # to show UI button

if wav_audio is not None:
    # Playback right away
    st.audio(wav_audio, format="audio/wav")

    # Convert WAV bytes to numpy waveform + sample rate
    data, sr = sf.read(io.BytesIO(wav_audio), dtype="float32", always_2d=False)

    # If stereo, average to mono now or later:
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Convert sample rate to 16,000 KHz
    target = 16000
    if sr != target:
        data = librosa.resample(data, orig_sr = sr, target_sr=target)
        sr = target

    st.success(f"Captured {len(data)/sr:.2f}s at {sr} Hz")

    