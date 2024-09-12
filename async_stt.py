from faster_whisper import WhisperModel
import asyncio
import os
from dotenv import load_dotenv
import numpy as np
import wave
from scipy.io import wavfile
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from utils import printing
from utils.timer import timer
import time

printing.printgreen("Printing VAD model")
vad_model = load_silero_vad()
printing.printgreen("VAD model loaded")
printing.printblue("Loading STT model")
stt_model = WhisperModel(
    "Systran/faster-whisper-small",
    device="cuda",
    device_index=[1],
    compute_type="float16",
)
printing.printblue("STT model loaded")
now = time.time()


@timer
def process_stt(wav):
    # global full_text
    full_text = ""
    # Convert raw audio data to numpy array
    # Perform STT transcription (simulated with a mock)
    # global i
    # i += 1
    segments, info = stt_model.transcribe("audio_new.wav", language="en")
    # print("STT info:", info)
    # some_random_text = "texts/text_" + str(i) + ".txt"
    # with open(some_random_text, "w") as f:
    # print("Writing STT results to file:", some_random_text)
    for segment in segments:
        text = segment.text
        # print("Processed audio segment:", text)
        # logging.info("Processed audio segment: %s", text)
        # f.write(text + "\n")
        full_text += text
    printing.printred("Full text: " + full_text)
    return full_text


process_stt("audio_new.wav")
print("Time taken to process STT:", time.time() - now)


@timer
async def process_vad(filename):
    # read the audio fole
    wav = read_audio(filename)
    # get the speech timestamps
    speech_timestamps = get_speech_timestamps(vad_model, wav)
    return speech_timestamps


full_text = ""


@timer
async def process_audio_with_timeout(filename):
    # We assume the audio data is continuous, so we slice the first 1 second of data
    # To get 1 second of data:
    # 1 sample is int16 (2 bytes), so for mono, we need SAMPLE_RATE samples for 1 second.
    # For stereo, it's SAMPLE_RATE * 2 samples.
    # Multiply by 2 because it's 16-bit (2 bytes per sample)
    info, wav_arr = wavfile.read(filename)
    try:
        # Use asyncio timeout to ensure the processing stops after 1 second
        full_text = process_stt(wav_arr)
    except asyncio.TimeoutError:
        print("STT processing timed out after 1 second.")
    return full_text
