import asyncio
from livekit import rtc
import wave
from scipy.io import wavfile
import numpy as np
import random
from faster_whisper import WhisperModel
import logging
import time
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from groq import Groq
import os
from dotenv import load_dotenv
from async_tts import ttswrapper

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
vad_model = load_silero_vad()
stt_model = WhisperModel("small", device="cpu")

SAMPLE_RATE = 48000
NUM_CHANNELS = 1


async def publish_frames(source: rtc.AudioSource, frequency: int):
    print("Publishing audio frames")
    # amplitude = 32767  # for 16-bit audio
    samples_per_channel = 480  # 10ms at 48kHz

    print("Reading from file go.wav")

    sample_rate, data = wavfile.read("audios/go.wav")

    print("Sample rate: ", sample_rate)
    audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, samples_per_channel)
    audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
    if len(data.shape) > 1:
        data = data.flatten()
    print("Done Reading from file ")
    audio_data_mv = memoryview(data.astype(np.int16))  # Adjust dtype if necessary
    await source.capture_frame(audio_frame)
    for i in range(0, len(data), 480):
        np.copyto(audio_data, audio_data_mv[i : i + 480])
        print("Sending audio frame: ", +i)
        await source.capture_frame(audio_frame)


async def GenerateCompletion(user_input):
    global groq_client
    messages = [
        {"role": "system", "content": "Do whatever user says"},
        {"role": "user", "content": user_input},
    ]
    chat_completion_stream = groq_client.chat.completions.create(
        messages=messages, model="llama-3.1-8b-instant", stream=True
    )
    full_answer = ""
    temp = ""
    i = 0
    for chunk in chat_completion_stream:
        try:
            c = chunk.choices[0].delta.content
            # print(f"\033[94m completion: {c} \033[0m")
            full_answer += c
            temp += c
            if c == "." or c == "!" or c == "?" or c == ":" or c == "\n" or c == "":
                answer = temp
                # print("sending form here")
                temp = ""
                if len(answer) > 0:
                    # print()
                    print(f"\033[94m completion: {answer} \033[0m")
                    yield answer
            i += 1
        except Exception as e:
            PrintRed("Error occured while streaming: " + str(e))
            answer = temp
            if answer != None:
                yield answer
            # yield TokenUsage(prompt_tokens=0, completion_tokens=0, total_cost=0)
            break


async def send_audio_frames(audio_stream: rtc.AudioStream):
    # Open a .wav file to read the audio data
    input_filename = "audio.wav"
    with wave.open(input_filename, "rb") as wf:
        # Read the .wav file format parameters
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()

        # Read the raw audio data from the .wav file
        audio_data = wf.readframes(wf.getnframes())

        # Create an AudioFrame object with the audio data
        audio_frame = rtc.AudioFrame(
            data=audio_data,
            sample_rate=sample_rate,
            num_channels=num_channels,
            sample_width=sample_width,
        )

        # Send the audio frame to the audio stream


def PrintRed(s):
    print("\033[91m {}\033[00m".format(s))


i = 0


async def process_stt(wav):
    # Convert raw audio data to numpy array
    # Perform STT transcription (simulated with a mock)
    global i
    i += 1
    segments, info = stt_model.transcribe(wav.numpy(), language="uz")
    print("STT info:", info)
    some_random_text = "texts/text_" + str(i) + ".txt"
    full_text = ""
    with open(some_random_text, "w") as f:
        print("Writing STT results to file:", some_random_text)
        for segment in segments:
            text = segment.text
            print("Processed audio segment:", text)
            # logging.info("Processed audio segment: %s", text)
            f.write(text + "\n")
            full_text += text + "\n"

    PrintRed("Full text: " + full_text)


async def process_audio_with_timeout(wav):
    # We assume the audio data is continuous, so we slice the first 1 second of data
    # To get 1 second of data:
    # 1 sample is int16 (2 bytes), so for mono, we need SAMPLE_RATE samples for 1 second.
    # For stereo, it's SAMPLE_RATE * 2 samples.
    # Multiply by 2 because it's 16-bit (2 bytes per sample)

    try:
        # Use asyncio timeout to ensure the processing stops after 1 second
        await asyncio.wait_for(process_stt(wav), timeout=1.0)
    except asyncio.TimeoutError:
        print("STT processing timed out after 1 second.")


async def receive_audio_frames(audio_stream: rtc.AudioStream):
    # Open a .wav file to write the audio data
    output_filename = "audio_new.wav"
    i = 0
    # audio_data = bytearray()
    all_buffer = bytearray()
    with wave.open(output_filename, "wb") as wf:
        async for frame_event in audio_stream:
            buffer = frame_event.frame
            arr = np.frombuffer(buffer.data, dtype=np.int16)
            sample_rate = buffer.sample_rate
            num_channels = buffer.num_channels

            # If this is the first frame, set up the .wav file format parameters
            if wf.getnframes() == 0:
                wf.setnchannels(num_channels)
                wf.setsampwidth(2)  # 2 bytes for 16-bit PCM
                wf.setframerate(sample_rate)

            # Write the raw audio data to the .wav file

            wf.writeframes(buffer.data)
            i += 1
            if i % 1000 == 0:
                loop_time = time.time()
                print("Audio Frame received: ", i)
                print("Length of buffer: ", len(buffer.data))
                # await process_audio_with_timeout()
                # logging.info("VAD Processing")
                print("Reading audio")
                starting_time = time.time()
                wav = read_audio("audio_new.wav")
                # backend (sox, soundfile, or ffmpeg) required!
                print(type(wav))

                speech_timestamps = get_speech_timestamps(wav, vad_model)
                PrintRed("Time to process VAD: " + str(time.time() - starting_time))
                print("Speech timestamps: ", speech_timestamps)
                starting_time = time.time()
                j = 0
                async for token in GenerateCompletion(
                    "Hello, please write python code"
                ):
                    if j == 0:
                        PrintRed(
                            "Time to get first chunk: "
                            + str(time.time() - starting_time)
                        )
                        j += 1
                    print("Token: ", token)
                    # print(token)cd
                # do tts the streams
                starting_time = time.time()
                with wave.open(f"output/full.wav", "wb") as wf:
                    wf.setnchannels(1)
                    wf.setframerate(24000)
                    wf.setsampwidth(2)
                    async for audio in ttswrapper(
                        "Hello, please write python code", "en"
                    ):
                        with open(f"output/output_.wav", "wb") as f:
                            f.write(audio)
                        with wave.open(f"output/output_.wav", "rb") as rf:
                            wf.writeframes(rf.readframes(rf.getnframes()))
                        i += i

                print("Time to do tts: " + str(time.time() - starting_time))
                print("Audio TTS done and saved")
                # print("Audio: ", audio)

                # await process_audio_with_timeout(wav)
    # do tts and send stream
    # read audios/go.wav and send it to the room
