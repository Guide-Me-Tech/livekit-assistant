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
from async_stt import (
    process_audio_with_timeout,
    process_stt,
    process_vad,
    process_stt_with_numpy,
)
from llm.groq import GroqLLMHandlers
from datetime import datetime
from livekit.plugins import silero
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from utils.timer import timer
import librosa
from scipy.signal import resample


load_dotenv(".env.production")
SAMPLE_RATE = 48000
NUM_CHANNELS = 1


async def publish_frames(source: rtc.AudioSource, frequency: int):
    print("Publishing audio frames")
    amplitude = 32767  # for 16-bit audio
    samples_per_channel = 480  # 10ms at 48kHz
    time = np.arange(samples_per_channel) / SAMPLE_RATE
    total_samples = 0

    print("Reading from file go.wav")

    sample_rate, data = wavfile.read("audios/go_new.wav")

    print("Sample rate: ", sample_rate)
    audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, samples_per_channel)
    audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
    resampled_data = resample(data, len(data) * SAMPLE_RATE // sample_rate)
    if sample_rate != SAMPLE_RATE:
        data = librosa.resample(
            data.astype(np.float32), orig_sr=sample_rate, target_sr=SAMPLE_RATE
        )

    if len(data.shape) > 1:
        data = data.flatten()
    print("Done Reading from file ")
    audio_data_mv = memoryview(data.astype(np.int16))  # Adjust dtype if necessary
    await source.capture_frame(audio_frame)
    print("Number of frames: ", len(data) // 480)
    for i in range(0, len(data) // 480 - 13000):
        np.copyto(audio_data, audio_data_mv[i * 480 : (i + 1) * 480])
        if i % 10000 == 0:
            print("Sending audio frame: ", +i / 480)
        await source.capture_frame(audio_frame)
    print("Done sending audio frames")

    return


async def send_audio_frames(source: rtc.AudioSource, filename: str):
    print("Publishing audio frames")
    amplitude = 32767  # for 16-bit audio
    samples_per_channel = 480  # 10ms at 48kHz
    time = np.arange(samples_per_channel) / SAMPLE_RATE
    total_samples = 0

    print("Reading from file go.wav")

    sample_rate, data = wavfile.read("audios/go_new.wav")

    print("Sample rate: ", sample_rate)
    audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, samples_per_channel)
    audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
    resampled_data = resample(data, len(data) * SAMPLE_RATE // sample_rate)
    if len(data.shape) > 1:
        data = data.flatten()
    print("Done Reading from file ")
    audio_data_mv = memoryview(
        resampled_data.astype(np.int16)
    )  # Adjust dtype if necessary
    # await source.capture_frame(audio_frame)
    print("Number of frames: ", len(audio_data_mv) // samples_per_channel)
    for i in range(0, len(audio_data_mv) // samples_per_channel):
        np.copyto(
            audio_data,
            audio_data_mv[i * samples_per_channel : (i + 1) * samples_per_channel],
        )
        if i % 1000 == 0:
            print(
                "Sending audio frame: [",
                str(i * samples_per_channel),
                ":",
                str((i + 1) * samples_per_channel),
                "] ------- ",
                str(i / 100),
                "seconds",
            )
        await source.capture_frame(audio_frame)
    print("Done sending audio frames")

    return


vad_model = load_silero_vad(onnx=True)


@timer
def check_vad(filename):
    """

    Returns True if the speech is stopped and the audio is less than 300ms
    Returns False if the speech is still ongoing
    """
    print("Doing vad")
    wav = read_audio(filename)
    full_wav_length = len(wav)
    # get the speech timestamps
    speech_timestamps = get_speech_timestamps(
        model=vad_model,
        audio=wav,
        min_silence_duration_ms=300,
        min_speech_duration_ms=300,
    )
    printing.printlightblue("Speech timestamps: " + str(speech_timestamps))
    if len(speech_timestamps) == 0:
        # no speech detected so keep listening
        return True, ""
    last_speech_timestamp = speech_timestamps[-1]
    printing.printlightblue("Full wav length: " + str(full_wav_length))
    printing.printlightblue("End of last speech: " + str(last_speech_timestamp["end"]))
    end_of_last_speech = last_speech_timestamp["end"]
    # the limit of the last speech timestamp
    # the limit is 300ms
    limit = 300
    k = 16
    if full_wav_length - end_of_last_speech >= limit * k:
        # speech has stopped
        stt_text = process_stt_with_numpy(wav.numpy())
        return False, stt_text
    return True, ""


def Send_Audio(source, audio_data):
    pass


async def receive_audio_frames(audio_stream: rtc.AudioStream, source: rtc.audio_frame):
    # Open a .wav file to write the audio data
    printing.printred("Started Receiving audio frames")
    groq_client = GroqLLMHandlers(api_key=os.getenv("GROQ_API_KEY"))

    output_filename = "audio_new.wav"
    i = 0
    # audio_data = bytearray()
    all_buffer = bytearray()

    # send audio frames first
    await send_audio_frames(source, "output/full.wav")
    return
    while True:
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
                print("Audio Frame received: ", i)
                continue
                if i % 100 == 0:
                    print("Now should process")

                    speech_ongoing, full_text = check_vad(output_filename)
                    if speech_ongoing:
                        printing.printpink("Speech is still ongoing")
                        continue
                    printing.printpink("Speech is stopped")

                    loop_time = time.time()
                    # get current time
                    current_time = datetime.now()
                    printing.printred("Time now is:" + str(current_time))
                    loop_count = 0
                    # print("Audio Frame received: ", i)
                    # print("Length of buffer: ", len(buffer.data))
                    # await process_audio_with_timeout()
                    # logging.info("VAD Processing")
                    # print("Processing VAD")

                    current_time = datetime.now()
                    printing.printred("Time now is:" + str(current_time))
                    # print("Speech timestamps: ", speech_timestamps)
                    # print("Processing STT")
                    # stt_text = process_stt(output_filename)
                    # wf.
                    stt_text = full_text
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
                            # starting_time = time.time()
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
                                    # current_time = datetime.now()
                                    # printing.printred("Time now is:" + str(current_time))
                                    loop_count += 1
                                with open(f"output/output_.wav", "wb") as f:
                                    f.write(audio)
                                with wave.open(f"output/output_.wav", "rb") as rf:
                                    wff.writeframes(rf.readframes(rf.getnframes()))
                                    await send_audio_frames(
                                        source, "output/output_.wav"
                                    )
                                j += 1
                            # break
                            return
                    # print("Time to do tts: " + str(time.time() - starting_time))
                    # print("Audio TTS done and saved")
                    # print("Audio: ", audio)
                    # print("To process")

                    # await process_audio_with_timeout(wav)
        # do tts and send stream
        # read audios/go.wav and send it to the room
