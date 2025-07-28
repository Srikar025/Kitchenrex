import os
import wave
import json
import streamlit as st
import pandas as pd
from vosk import Model, KaldiRecognizer

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("transcripts", exist_ok=True)

# Load Vosk model
@st.cache_resource
def load_model():
    return Model("models/vosk-model-small-en-us-0.15")

model = load_model()

# App title
st.title("🎙️ Voice-to-Text Transcriber")
st.write("Upload a 5–10 min audio file (in WAV format).")

# User input
username = st.text_input("Enter your name:")
question = st.text_input("Question you're answering:")
audio_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])

if audio_file and username and question:
    # Save uploaded file
    file_path = f"uploads/{username}_{audio_file.name}"
    with open(file_path, "wb") as f:
        f.write(audio_file.read())

    # Transcribe
    wf = wave.open(file_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text += result.get("text", "") + " "

    final_result = json.loads(rec.FinalResult())
    text += final_result.get("text", "")

    # Show result
    st.subheader("📄 Transcription Result:")
    st.text_area("Your Transcribed Answer:", text.strip(), height=250)

    # Save to CSV
    df = pd.DataFrame([{
        "username": username,
        "question": question,
        "answer": text.strip()
    }])

    csv_path = "transcripts/transcripts.csv"
    df.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))

    st.success("✅ Transcription saved to transcripts/transcripts.csv")
