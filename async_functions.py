from utils import printing
import asyncio
from livekit import rtc
import wave
from scipy.io import wavfile
import numpy as np
import time
from groq import Groq
import os
from dotenv import load_dotenv
from async_tts import ttswrapper
from async_stt import process_audio_with_timeout, process_stt, process_vad
from llm.groq import GroqLLMHandlers
from datetime import datetime

load_dotenv()
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


async def receive_audio_frames(audio_stream: rtc.AudioStream):
    # Open a .wav file to write the audio data

    groq_client = GroqLLMHandlers(api_key=os.getenv("GROQ_API_KEY"))

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
                # get current time
                current_time = datetime.now()
                printing.printred("Time now is:" + str(current_time))
                loop_count = 0
                print("Audio Frame received: ", i)
                print("Length of buffer: ", len(buffer.data))
                # await process_audio_with_timeout()
                # logging.info("VAD Processing")
                print("Processing VAD")
                process_vad(output_filename)
                current_time = datetime.now()
                printing.printred("Time now is:" + str(current_time))
                # print("Speech timestamps: ", speech_timestamps)
                print("Processing STT")
                stt_text = process_stt(output_filename)
                starting_time = time.time()
                j = 0
                current_time = datetime.now()
                printing.printred("Time now is:" + str(current_time))
                with wave.open(f"output/full.wav", "wb") as wff:
                    wff.setnchannels(1)
                    wff.setframerate(24000)
                    wff.setsampwidth(2)
                    for token in groq_client.QueryStreams(
                        messages=groq_client.ConstructMessages(
                            context="{'username': 'John', 'debt': '100230'}",
                            messages=[
                                {
                                    "role": "user",
                                    "content": stt_text,
                                }
                            ],
                        )
                    ):
                        if j == 0:
                            printing.printred(
                                "Time to get first chunk: "
                                + str(time.time() - starting_time)
                            )
                        print("Token: ", token)
                        # print(token)cd
                        # do tts streams for tokens(sentences) generated
                        if len(token) < 5:
                            continue
                        starting_time = time.time()
                        async for audio in ttswrapper(token, "en"):
                            if loop_count == 0:
                                printing.printred(
                                    "#" * 50
                                    + "\n"
                                    + "To get first audio chunk from stt to tts: "
                                    + str(time.time() - loop_time)
                                    + "\n"
                                    + "#" * 50
                                    + "\n"
                                )
                                current_time = datetime.now()
                                printing.printred("Time now is:" + str(current_time))
                                loop_count += 1
                            with open(f"output/output_.wav", "wb") as f:
                                f.write(audio)
                            with wave.open(f"output/output_.wav", "rb") as rf:
                                wff.writeframes(rf.readframes(rf.getnframes()))
                            j += 1

                    print("Time to do tts: " + str(time.time() - starting_time))
                    print("Audio TTS done and saved")
                    # print("Audio: ", audio)
                    # print("To process")
                    return

                # await process_audio_with_timeout(wav)
    # do tts and send stream
    # read audios/go.wav and send it to the room
