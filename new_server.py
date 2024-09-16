from livekit import rtc, api
import logging
import asyncio
import os
import wave
from dotenv import load_dotenv
import numpy as np
import time
import torch
from async_functions import receive_audio_frames, publish_frame_from_queue
import concurrent.futures
import threading


torch.set_num_threads(2)
load_dotenv(".env.production")


SAMPLE_RATE = 48000
NUM_CHANNELS = 1


def get_token():
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
        .to_jwt()
    )
    return token


async def main(loop: asyncio.AbstractEventLoop = None, room: rtc.Room = None):
    # make room only audio
    # room = rtc.Room(loop=loop)
    # # room_options = rtc.RoomOptions(
    # #     rtc_config=rtc.RtcConfiguration(

    # #     )
    # # )
    # await room.connect(
    #     url=os.getenv("LIVEKIT_URL"),
    #     token=get_token(),
    # )
    logging.info("connected to room %s", room.name)
    chat = rtc.ChatManager(room=room)
    logging.info("Chat manager created")
    # audio streams
    audio_stream = None

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logging.info(
            "participant connected: %s %s", participant.sid, participant.identity
        )

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info("track unsubscribed: %s", publication.sid)

    @room.on("track_muted")
    def on_track_muted(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logging.info("track muted: %s", publication.sid)

    @room.on("track_unmuted")
    def on_track_unmuted(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logging.info("track unmuted: %s", publication.sid)

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        logging.info("received data from %s: %s", data.participant.identity, data.data)

    @room.on("connection_quality_changed")
    def on_connection_quality_changed(
        participant: rtc.Participant, quality: rtc.ConnectionQuality
    ):
        logging.info("connection quality changed for %s", participant.identity)
        # write back

    @room.on("track_subscription_failed")
    def on_track_subscription_failed(
        participant: rtc.RemoteParticipant, track_sid: str, error: str
    ):
        logging.info("track subscription failed: %s %s", participant.identity, error)

    @room.on("connection_state_changed")
    def on_connection_state_changed(state: rtc.ConnectionState):
        logging.info("connection state changed: %s", state)

    @room.on("connected")
    def on_connected() -> None:
        logging.info("connected")

    @room.on("disconnected")
    def on_disconnected() -> None:
        logging.info("disconnected")

    @room.on("reconnecting")
    def on_reconnecting() -> None:
        logging.info("reconnecting")

    @room.on("reconnected")
    def on_reconnected() -> None:
        logging.info("reconnected")

    # track_subscribed is emitted whenever the local participant is subscribed to a new track
    # async def publish_audio():
    # source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    # track = rtc.LocalAudioTrack.create_audio_track("sinewave", source)
    # options = rtc.TrackPublishOptions()
    # publication = await room.local_participant.publish_track(track, options)
    # print("published track %s", publication.sid)
    # asyncio.ensure_future(publish_frames(source, 440))

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            nonlocal audio_stream
            if audio_stream is not None:
                return

            print("Subscribed to an Audio Track")
            print("Track info: ", track)
            audio_stream = rtc.AudioStream(track)

            # publish a track
            # source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
            # track_output = rtc.LocalAudioTrack.create_audio_track("sinewave", source)
            # options = rtc.TrackPublishOptions()
            # options.source = rtc.TrackSource.SOURCE_MICROPHONE

            # # Run the async function in the event loop
            # loop_2 = asyncio.get_event_loop()
            # print("Publishing track")
            # publish_track_task = asyncio.run_coroutine_threadsafe(
            #     room.local_participant.publish_track(track_output, options), loop_2
            # )
            # print("Track published: ", track_output)
            # Create a thread pool executor
            # executor = concurrent.futures.ThreadPoolExecutor()
            # Start the receive_audio_frames task
            asyncio.ensure_future(receive_audio_frames(audio_stream, source))
            # asyncio.ensure_future(publish_frame_from_queue(source))
            # loop_2.run_in_executor(executor, publish_frame_from_queue, source)
            # open new thread and run publish_frame_from_queue in a separate thread

            # asyncio.ensure_future(publish_frames(source, 440))

    print("Particapants: ", room.remote_participants)
    print("Tracks: ", room)

    # By default, autosubscribe is enabled. The participant will be subscribed to
    # all published tracks in the room

    # participants and tracks that are already available in the room
    # participant_connected and track_published events will *not* be emitted for them
    # for participant in room.participants.items():
    #     for publication in participant.track_publications.items():
    #         print("track publication: %s", publication.sid)

    @chat.on("message_received")
    def on_message(msg: rtc.ChatMessage):
        print("Message received: ", msg)
        starting_time = time.time()
        task = asyncio.ensure_future(chat.send_message(f"You message: {msg.message}"))
        ending_time = time.time()
        if task.done():
            print("Task done")
            print(f"Time taken to send message: {ending_time - starting_time}")


def run_publish_in_thread(source):
    """This function will run in a separate thread to handle publish_frame_from_queue."""
    loop = asyncio.new_event_loop()  # Create a new event loop for this thread
    asyncio.set_event_loop(loop)  # Set it as the current thread's event loop

    # Now we can safely run the coroutine in this new event loop
    loop.run_until_complete(publish_frame_from_queue(source))
    loop.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("logs/cccc.log"), logging.StreamHandler()],
    )
    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)
    task = loop.create_task(
        room.connect(url=os.getenv("LIVEKIT_URL"), token=get_token())
    )
    loop.run_until_complete(task)
    asyncio.ensure_future(main(room=room, loop=loop))

    print("Connected to room")

    # Publish audio track
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track_output = rtc.LocalAudioTrack.create_audio_track("sinewave", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    print("Publishing track")
    publish_track_task = asyncio.run_coroutine_threadsafe(
        room.local_participant.publish_track(track_output, options), loop
    )
    print("Track published: ", track_output)

    # Schedule the main logic

    # Run the publish_frame_from_queue in a separate thread
    publish_thread = threading.Thread(target=run_publish_in_thread, args=(source,))
    publish_thread.start()

    try:
        # Run the event loop forever
        loop.run_forever()
    finally:
        loop.close()
