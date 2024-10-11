import asyncio
import logging
from signal import SIGINT, SIGTERM
import os
from dotenv import load_dotenv

load_dotenv(".env.production")
import numpy as np
from livekit import rtc, api

SAMPLE_RATE = 44100
NUM_CHANNELS = 2

# ensure LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are set


async def main(room: rtc.Room) -> None:
    @room.on("participant_disconnected")
    def on_participant_disconnect(participant: rtc.Participant, *_):
        logging.info("participant disconnected: %s", participant.identity)

    token = (
        api.AccessToken()
        .with_identity("python-publisher")
        .with_name("Python Publisher")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room="my-room",
            )
        )
        .to_jwt()
    )
    url = os.getenv("LIVEKIT_URL")

    logging.info("connecting to %s", url)
    try:
        await room.connect(
            url,
            token,
            options=rtc.RoomOptions(
                auto_subscribe=True,
            ),
        )
        logging.info("connected to room %s", room.name)
    except rtc.ConnectError as e:
        logging.error("failed to connect to the room: %s", e)
        return

    # publish a track
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("sinewave", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication = await room.local_participant.publish_track(track, options)
    logging.info("published track %s", publication.sid)

    asyncio.ensure_future(publish_frames(source, 440))


import wave


async def publish_frames(source: rtc.AudioSource, frequency: str):
    filepath = "go.wav"
    amplitude = 32767  # For 16-bit audio
    samples_per_channel = 480  # 10ms at 48kHz
    total_samples = 0

    # Open and read audio data from the file (e.g., go.wav)
    with wave.open(filepath, "rb") as wav_file:
        # Ensure the wav file parameters match the expected ones
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getnchannels() == NUM_CHANNELS
        assert wav_file.getsampwidth() == 2  # 16-bit audio

        # Read the full file's audio data
        audio_data = np.frombuffer(
            wav_file.readframes(wav_file.getnframes()), dtype=np.int16
        )

    # Split the audio data into chunks and publish each chunk
    while total_samples < len(audio_data):
        # Select a chunk of the audio data (10ms frame)
        start = total_samples
        end = start + samples_per_channel
        if end > len(audio_data):
            break  # Stop when we've processed all the audio data

        audio_chunk = audio_data[start:end]

        # Create an AudioFrame for each chunk
        audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, len(audio_chunk))

        # Now write the audio_chunk to the audio_frame
        audio_frame_buffer = audio_frame.get_buffer()  # Get the internal buffer
        audio_frame_buffer[:] = audio_chunk.tobytes()  # Write the chunk to the buffer

        # Capture the frame and send it to the AudioSource
        await source.capture_frame(audio_frame)

        # Move to the next chunk
        total_samples += samples_per_channel
        print("Total samples processed:", total_samples)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler("logs/publish_wave.log"),
            logging.StreamHandler(),
        ],
    )

    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup():
        await room.disconnect()
        loop.stop()

    asyncio.ensure_future(main(room))
    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()
