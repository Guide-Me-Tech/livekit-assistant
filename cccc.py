import asyncio
import logging
import os
from signal import SIGINT, SIGTERM

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from livekit import api, rtc
from dotenv import load_dotenv

load_dotenv()
# ensure LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are set

tasks = set()

# You can download a face landmark model file from https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models
# model_file = "face_landmarker.task"
# model_path = os.path.dirname(os.path.realpath(__file__)) + "/" + model_file

# BaseOptions = mp.tasks.BaseOptions
# FaceLandmarker = mp.tasks.vision.FaceLandmarker
# FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# options = FaceLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.VIDEO,
# )
from faster_whisper import WhisperModel

stt_model = WhisperModel("aslon1213/whisper-small-uz-with-uzbekvoice-ct2", device="cpu")


async def main(loop: asyncio.AbstractEventLoop) -> None:
    video_stream = None
    audio_stream = None
    print("Starting main")
    room = rtc.Room(loop=loop)
    token = (
        api.AccessToken()
        .with_identity("python-bot")
        .with_name("Python Bot")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room="my-room",
            )
        )
    )
    # print("connecting to room: " + os.getenv("LIVEKIT_URL"))
    print("Token: ", token.to_jwt())

    async def connect(room: rtc.Room, token: api.AccessToken):
        await room.connect(os.getenv("LIVEKIT_URL"), token.to_jwt())

    await connect(room, token)
    chat = rtc.ChatManager(room)

    @chat.on("message_received")
    def on_message_received(message: rtc.ChatMessage, *_):
        print(f"Message received: {message.message}")

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, *_):

        if track.kind == rtc.TrackKind.KIND_AUDIO:
            nonlocal audio_stream
            if video_stream is not None:
                # only process the first stream received
                return

            print("Subscribed to an Audio Track")
            print("Track info: ", track)
            # input("Press Enter to continue...")
            audio_stream = rtc.AudioStream(track)
            task = asyncio.create_task(audio_stream_handler_3(audio_stream))
            tasks.add(task)
            task.add_done_callback(tasks.remove)

    # chat = rtc.ChatManager(room)
    print("connected to room: " + room.name)


async def audio_stream_handler(audio_stream: rtc.AudioStream) -> None:
    async for frame_event in audio_stream:
        buffer = frame_event.frame
        arr = np.frombuffer(buffer.data, dtype=np.int16)

        sample_rate = buffer.sample_rate
        num_channels = buffer.num_channels
        print("Audio Frame received")
        print("Sample Rate: ", sample_rate)
        print("Num Channels: ", num_channels)
        # arr = arr.reshape((buffer.sample_rate, buffer.num_channels))
        arr = arr.reshape((-1, num_channels))  # Reshape based on the number of channels
        print("Audio Frame: ", arr.size)


async def audio_stream_handler_2(audio_stream: rtc.AudioStream) -> None:
    async for frame_event in audio_stream:
        buffer = frame_event.frame
        arr = np.frombuffer(buffer.data, dtype=np.int16)
        sample_rate = buffer.sample_rate
        num_channels = buffer.num_channels

        print("Audio Frame received")
        print("Sample Rate: ", sample_rate)
        print("Num Channels: ", num_channels)

        # Reshape array based on number of channels
        arr = arr.reshape((-1, num_channels))  # Now arr is [samples x channels]

        print("Audio Frame size: ", arr.shape)

        # Access individual channels
        left_channel = arr[:, 0]  # Assuming stereo, this is the left channel
        if num_channels > 1:
            right_channel = arr[:, 1]  # Right channel (if stereo)

        # Process the audio data here
        # Example: print the mean of the left channel
        print("Mean of left channel: ", np.mean(left_channel))
        # save audio data to a file
        with open("audio.wav", "wb") as f:
            f.write(buffer.data)


import wave

SAMPLE_RATE = 48000
NUM_CHANNELS = 1
import random


async def process_stt():
    # Convert raw audio data to numpy array
    # Perform STT transcription (simulated with a mock)
    segments, info = stt_model.transcribe("audio.wav", language="uz")
    print("STT info:", info)
    some_random_text = "texts/text_" + str(random.randint(1, 1000)) + ".txt"
    with open(some_random_text, "w") as f:
        print("Writing STT results to file:", some_random_text)
        for segment in segments:
            text = segment.text
            print("Processed audio segment:", text)
            logging.info("Processed audio segment: %s", text)
            f.write(text + "\n")


async def process_audio_with_timeout():
    # We assume the audio data is continuous, so we slice the first 1 second of data
    # To get 1 second of data:
    # 1 sample is int16 (2 bytes), so for mono, we need SAMPLE_RATE samples for 1 second.
    # For stereo, it's SAMPLE_RATE * 2 samples.
    # Multiply by 2 because it's 16-bit (2 bytes per sample)

    try:
        # Use asyncio timeout to ensure the processing stops after 1 second
        await asyncio.wait_for(process_stt(), timeout=1.0)
    except asyncio.TimeoutError:
        print("STT processing timed out after 1 second.")


audio_data = bytearray()


async def audio_stream_handler_3(audio_stream: rtc.AudioStream) -> None:
    # Open a .wav file to write the audio data
    output_filename = "audio.wav"
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
                print("Audio Frame received: ", i)
                # await process_audio_with_timeout()

            # print(f"Audio Frame received: {arr.size} samples")

    print(f"Audio saved to {output_filename}")


def draw_landmarks_on_image(rgb_image, detection_result):
    # from https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb
    face_landmarks_list = detection_result.face_landmarks

    # Loop through the detected faces to visualize.
    for face_landmarks in face_landmarks_list:
        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=rgb_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=rgb_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=rgb_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )


# async def frame_loop(video_stream: rtc.VideoStream) -> None:
#     landmarker = FaceLandmarker.create_from_options(options)
#     cv2.namedWindow("livekit_video", cv2.WINDOW_AUTOSIZE)
#     cv2.startWindowThread()
#     async for frame_event in video_stream:
#         buffer = frame_event.frame

#         arr = np.frombuffer(buffer.data, dtype=np.uint8)
#         arr = arr.reshape((buffer.height, buffer.width, 3))

#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
#         detection_result = landmarker.detect_for_video(
#             mp_image, frame_event.timestamp_us
#         )

#         draw_landmarks_on_image(arr, detection_result)

#         arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
#         cv2.imshow("livekit_video", arr)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     landmarker.close()
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("logs/cccc.log"), logging.StreamHandler()],
    )

    loop = asyncio.get_event_loop()

    # chat = rtc.ChatManager(room)

    async def cleanup():
        loop.stop()

    asyncio.ensure_future(main(loop))
    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()
