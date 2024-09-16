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
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from utils.timer import timer
import librosa
from scipy.signal import resample
import threading
import sys

#
#
#
new_speech_from_user_condition = False
sending_audio = False
#
#
#
load_dotenv(".env.production")
SAMPLE_RATE = 48000
NUM_CHANNELS = 1
# import queue
import queue

audio_frames_que = queue.Queue(maxsize=100)
# audio_frames_que = asyncio.Queue(10)
tokens_que = asyncio.Queue(10)


async def publish_frame_from_queue(
    source: rtc.AudioSource,
):
    # get thread id
    print("Thread info in publishing frame: ", threading.current_thread())

    global audio_frames_que
    global sending_audio
    samples_per_channel = 480  # 10ms at 48kHz
    audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, samples_per_channel)
    audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
    file_num = 0
    samples_per_channel = 480  # 10ms at 48kHz
    amplitude = 32767  # for 16-bit audio
    all_data = []

    while True:
        try:
            data = audio_frames_que.get_nowait()
        except queue.Empty:
            # audio sending is done because the queue is empty
            sending_audio = False
            await asyncio.sleep(0.1)
            continue
        printing.printyellow(f"GOT AUDIO DATA FROM QUEUE - {len(data)}")
        printing.printyellow(f"File number: {file_num}")
        # if file_num != 0:
        #     # time = np.arange(samples_per_channel) / SAMPLE_RATE
        #     # total_samples = 0
        #     sample_rate = 24000
        #     print("Sample rate: ", sample_rate)
        #     # audio_frame = rtc.AudioFrame.create(
        #     #     SAMPLE_RATE, NUM_CHANNELS, samples_per_channel
        #     # audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
        #     resampled_data = librosa.resample(
        #         data, orig_sr=24000, target_sr=SAMPLE_RATE
        #     )
        #     if resampled_data.dtype == np.float32 or resampled_data.dtype == np.float64:
        #         max_val = np.max(
        #             np.abs(resampled_data)
        #         )  # Get the maximum absolute value
        #         if max_val > 0:
        #             resampled_data = resampled_data / max_val  # Normalize to [-1, 1]
        #         resample_16k = np.int16(
        #             resampled_data * amplitude
        #         )  # Convert to 16-bit PCM
        #     with wave.open(f"output/output_{file_num}.wav", "wb") as f:
        #         f.setnchannels(1)
        #         f.setsampwidth(2)
        #         f.setframerate(48000)
        #         f.writeframes(resample_16k.tobytes())
        #         file_num += 1
        #     # if len(data.shape) > 1:
        #     #     resample_16k = resample_16k.flatten()
        #     audio_data_mv = memoryview(resample_16k.astype(np.int16))
        # else:
        #     resample_16k = data
        #     # if len(data.shape) > 1:
        #     #     resample_16k = resample_16k.flatten()
        #     audio_data_mv = memoryview(resample_16k.astype(np.int16))
        for i in range(0, (len(data) // samples_per_channel)):
            np.copyto(
                audio_data,
                data[i * samples_per_channel : (i + 1) * samples_per_channel],
            )
            await source.capture_frame(audio_frame)
            # if new_speech_from_user_condition:
            #     break
        file_num += 1

        # if new_speech_from_user_condition:
        #     # get all the remaining data
        #     for i in range(audio_frames_que.qsize()):
        #         audio_frames_que.get_nowait()
        # print("Send audio frame")
        # if not audio_frames_que.empty():
        #     np.copyto(audio_data, audio_frames_que.get())
        #     await source.capture_frame(audio_frame)
        #     print("Sending audio frame")
        # else:
        #     await asyncio.sleep(0.1)
        #     continue
        # print("No audio data in queue")
        # await asyncio.sleep(0.1)


async def send_audio_frames(
    source: rtc.AudioSource,
):
    global new_speech_from_user_condition
    file_num = 0
    previous_frames_set = 0
    previous_frames = []
    global audio_frames_que
    # asyncio.ensure_future(publish_frame_from_queue(source))

    while True:

        token, streaming = await tokens_que.get()
        printing.printred(f"GOT TOKEN FROM QUEUE: {token} ----- STREAMING: {streaming}")
        async for data in ttswrapper(token, "ru", streaming):

            # await audio_frames_que.put(data)
            printing.printyellow(f"PUT DATA TO QUEUE - {len(data)}")
            # continue

            print("Publishing audio frames")
            amplitude = 32767  # for 16-bit audio
            samples_per_channel = 480  # 10ms at 48kHz
            # time = np.arange(samples_per_channel) / SAMPLE_RATE
            # total_samples = 0

            # print("Reading from file")

            sample_rate = 24000
            # print("Sample rate: ", sample_rate)
            # audio_frame = rtc.AudioFrame.create(
            #     SAMPLE_RATE, NUM_CHANNELS, samples_per_channel
            # audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
            resampled_data = librosa.resample(
                data, orig_sr=24000, target_sr=SAMPLE_RATE
            )
            if resampled_data.dtype == np.float32 or resampled_data.dtype == np.float64:
                max_val = np.max(
                    np.abs(resampled_data)
                )  # Get the maximum absolute value
                if max_val > 0:
                    resampled_data = resampled_data / max_val  # Normalize to [-1, 1]
                resample_16k = np.int16(
                    resampled_data * amplitude
                )  # Convert to 16-bit PCM
            with wave.open(f"output/output_{file_num}.wav", "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(48000)
                f.writeframes(resample_16k.tobytes())
                file_num += 1

            # if len(data.shape) > 1:
            #     resample_16k = resample_16k.flatten()
            # print("Done Reading from file ")
            audio_data_mv = memoryview(
                resample_16k.astype(np.int16)
            )  # Adjust dtype if necessary
            # await source.capture_frame(audio_frame)
            # print("Number of frames: ", (len(audio_data_mv) // samples_per_channel))
            sending_audio = True
            audio_frames_que.put(audio_data_mv)

            # if previous_frames_set != 0:
            #     audio_data_mv = np.concatenate((previous_frames, audio_data_mv))
            #     previous_frames_set = 1


vad_model = load_silero_vad(onnx=True)


@timer
def check_vad(filename):
    """

    Returns True if the speech is stopped and the audio is less than 300ms
    Returns False if the speech is still ongoing
    """
    print("Doing vad")
    limit = 500
    try:
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
            raise Exception("No speech detected so delete the file")

            return True, ""
        last_speech_timestamp = speech_timestamps[-1]
        printing.printlightblue("Full wav length: " + str(full_wav_length))
        printing.printlightblue(
            "End of last speech: " + str(last_speech_timestamp["end"])
        )
        end_of_last_speech = last_speech_timestamp["end"]
    except Exception as e:
        print("Error: ", e)
        return True, ""
    # the limit of the last speech timestamp
    # the limit is 300ms

    k = 16
    if full_wav_length - end_of_last_speech >= limit * k:
        # speech has stopped
        stt_text = process_stt_with_numpy(wav.numpy())
        return False, stt_text
    return True, ""


async def SendOkeyAudio():
    print("Publishing audio frames")
    amplitude = 32767  # for 16-bit audio
    samples_per_channel = 480  # 10ms at 48kHz
    time = np.arange(samples_per_channel) / SAMPLE_RATE
    total_samples = 0

    # print("Reading from file")
    # sample_rate, data = wavfile.read("audios/go_new.wav")
    sample_rate, data = wavfile.read("okey_48k.wav")

    # data = data[6500000:]
    # print("Sample rate: ", sample_rate)
    # audio_frame = rtc.AudioFrame.create(
    #     SAMPLE_RATE, NUM_CHANNELS, samples_per_channel
    # audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
    if sample_rate != SAMPLE_RATE:
        resampled_data = librosa.resample(data, orig_sr=24000, target_sr=SAMPLE_RATE)
        if resampled_data.dtype == np.float32 or resampled_data.dtype == np.float64:
            max_val = np.max(np.abs(resampled_data))  # Get the maximum absolute value
            if max_val > 0:
                resampled_data = resampled_data / max_val  # Normalize to [-1, 1]
            resample_16k = np.int16(resampled_data * 32767)  # Convert to 16-bit PCM
    else:
        resample_16k = data
    # print("Done Reading from file ")
    audio_data_mv = memoryview(
        resample_16k.astype(np.int16)
    )  # Adjust dtype if necessary
    audio_frames_que.put(audio_data_mv)
    # await source.capture_frame(audio_frame)
    # print("Number of frames: ", (len(audio_data_mv) // samples_per_channel))
    # for i in range(0, (len(audio_data_mv) // samples_per_channel) - 12500):
    #     await audio_frames_que.put(
    #         audio_data_mv[i * samples_per_channel : (i + 1) * samples_per_channel]
    #     )


import requests
import subprocess
import json
import os

# Fetch the access token using gcloud command
# access_token = os.system("gclo")
access_token = "ya29.a0AcM612z36tcFo9swVg1R8KEFidhq9fVKNhOp4zr-ev0H3L8MFj0vsgkGfqEq7znzdvjQW2nJ4fVWUrEatF-eccmCdJuiihx6PhtKsX0sCxJ-O8iuBQ0FTy5qgpWk7YpPoBD0dalORhAt3dECfb0vudvZh7iOOQoEgGdbjN-miwIg8AaCgYKAa4SARMSFQHGX2MiITECjNJyx4AEl21gMyQ9XA0181"
# Define the headers
headers = {
    "Authorization": f"Bearer {access_token}",
    "x-goog-user-project": "chrome-axe-396907",
    "Content-Type": "application/json; charset=utf-8",
}


async def translate_text(text, language):
    data = {"q": text, "target": language}
    # Make the POST request
    url = "https://translation.googleapis.com/language/translate/v2"
    response = requests.post(url, headers=headers, json=data)
    return response.json()["data"]["translations"][0]["translatedText"]


async def receive_audio_frames(audio_stream: rtc.AudioStream, source: rtc.audio_frame):
    # Open a .wav file to write the audio data
    global new_speech_from_user_condition
    print("Thread info in receive audio frame: ", threading.current_thread())
    printing.printred("Started Receiving audio frames")
    groq_client = GroqLLMHandlers(api_key=os.getenv("GROQ_API_KEY"))

    output_filename = "audio_new.wav"

    # audio_data = bytearray()
    all_buffer = bytearray()

    # send audio frames first
    # await send_audio_frames(source, "output/full.wav")
    # return

    asyncio.ensure_future(send_audio_frames(source))
    should_break = False

    messages = []
    while True:
        if sending_audio:
            asyncio.sleep(0.1)
            continue
        print("Inside loop")
        try:
            os.remove("audio_new.wav")
        except Exception as e:
            print("Error: ", e)
            with open("audio_new.wav", "w") as f:
                f.write("Hello world")
            continue
        # wait queue to be empty
        while not audio_frames_que.empty():
            await asyncio.sleep(0.1)
        i = 0
        with wave.open(output_filename, "wb") as wf:
            async for frame_event in audio_stream:
                buffer = frame_event.frame
                arr = np.frombuffer(buffer.data, dtype=np.int16)
                sample_rate = buffer.sample_rate
                num_channels = buffer.num_channels

                # If this is the first frame, set up the .wav file format parameters
                # print("Number of frames: ", wf.getnframes())
                if wf.getnframes() == 0 or i == 0:
                    wf.setnchannels(num_channels)
                    wf.setsampwidth(2)  # 2 bytes for 16-bit PCM
                    wf.setframerate(sample_rate)

                # Write the raw audio data to the .wav file

                wf.writeframes(buffer.data)
                i += 1
                if sending_audio:
                    break

                if i > 0 and i % 30 == 0:
                    loop_time = time.time()
                    try:
                        speech_ongoing, full_text = check_vad(output_filename)
                        if speech_ongoing:
                            # printing.printpink("Speech is still ongoing")
                            continue
                        # printing.printpink("Speech is stopped")
                        should_break = True
                        new_speech_from_user_condition = True
                    except Exception as e:
                        # flush the file contents
                        continue
                    # send Okey Audio

                    # asyncio.create_task(SendOkeyAudio())

                    # loop_time = time.time()
                    # get current time
                    # current_time = datetime.now()
                    # printing.printred("Time now is:" + str(current_time))
                    loop_count = 0

                    # current_time = datetime.now()
                    # printing.printred("Time now is:" + str(current_time))

                    stt_text = await translate_text(full_text, "en")
                    messages.append(
                        {
                            "role": "user",
                            "content": stt_text,
                        }
                    )
                    starting_time = time.time()
                    j = 0
                    # current_time = datetime.now()
                    # printing.printred("Time now is:" + str(current_time))
                    # with wave.open(f"output/full.wav", "wb") as wff:
                    # wff.setnchannels(1)
                    # wff.setframerate(24000)
                    # wff.setsampwidth(2)
                    streaming = True
                    llm_text_full = ""
                    for token in groq_client.QueryStreams(
                        messages=groq_client.ConstructMessages(
                            context="""{
                            "user_id": "12345",
                            "name": "Aslon",
                            "phone": "+777 10 10",
                            "email": "aslon.hamidov@example.com",
                            "debt_info": {
                                "total_debt": 150000.00,
                                "currency": "SUMM",
                                "loan_type": "Personal Loan",
                                "interest_rate": 5.75,
                                "monthly_payment": 50000.00,
                                "next_payment_due_date": "2024-10-01",
                                "overdue_payments": 2,
                                "last_payment_date": "2024-08-01",
                                "last_payment_amount": 500.00,
                                "penalties": {
                                    "late_fee": 50.00,
                                    "total_penalties": 100.00
                                },
                                "remaining_balance": 14500.00,
                                "loan_start_date": "2022-01-01",
                                "loan_due_date": "2027-01-01"
                            },
                            "status": "Delinquent",
                            "contact_history": [
                                {
                                    "date": "2024-08-15",
                                    "method": "Phone",
                                    "result": "No Answer"
                                },
                                {
                                    "date": "2024-07-25",
                                    "method": "Email",
                                    "result": "Reminder Sent"
                                }
                            ],
                            "bank_details": {
                                "bank_name": "BRB bank",
                                "branch_name": "Main Branch",
                                "branch_code": "001",
                                "contact_number": "+888 55 44"
                            }
                            }
                        """,
                            messages=messages,
                        )
                    ):
                        if j == 0:
                            printing.printred(
                                "Time to get first chunk: "
                                + str(time.time() - starting_time)
                            )
                            printing.printred(
                                f"To get first chunk from vad: {time.time() -loop_time + 0.2}"
                            )
                            new_speech_from_user_condition = False

                        if len(token) < 5:
                            continue
                        llm_text_full += token
                        # trabslation
                        translated_text = await translate_text(token, sys.argv[2])
                        translated_text = translated_text.replace("&#39;", "'")
                        await tokens_que.put([translated_text, streaming])
                        printing.printred(
                            f"PUT TOKEN TO QUEUE: {token} ------ STREAMING: {streaming}"
                        )
                        j += 1
                        # print(token)cd
                        # do tts streams for tokens(sentences) generated

                        # starting_time = time.time()
                        # await send_audio_frames(source, token, streaming=streaming)
                        streaming = True
                        # async for audio_numpy in ttswrapper(token, "en"):

                        #     if loop_count == 0:
                        #         printing.printred(
                        #             "#" * 50
                        #             + "\n"
                        #             + "To get first audio chunk from stt to tts: "
                        #             + str(time.time() - loop_time)
                        #             + "\n"
                        #             + "#" * 50
                        #             + "\n"
                        #         )
                        #         # current_time = datetime.now()
                        #         # printing.printred("Time now is:" + str(current_time))
                        #         loop_count += 1
                        #         # with open(f"output/output_.wav", "wb") as f:
                        #         #     f.write(audio_numpy)
                        #         # with wave.open(f"output/output_.wav", "rb") as rf:
                        #         #     wff.writeframes(rf.readframes(rf.getnframes()))

                        #     j += 1
                    messages.append(
                        {
                            "role": "assistant",
                            "content": llm_text_full,
                        }
                    )
                if should_break:
                    # time.sleep(150)
                    print("Breaking point")
                    should_break = False
                    # time.sleep(5)
                    break
                # return
                # print("Time to do tts: " + str(time.time() - starting_time))
                # print("Audio TTS done and saved")
                # print("Audio: ", audio)
                # print("To process")

                # await process_audio_with_timeout(wav)
        # do tts and send stream
        # read audios/go.wav and send it to the room
