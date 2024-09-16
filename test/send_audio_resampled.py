from livekit import rtc, api
import logging
import asyncio
import os
import wave
from dotenv import load_dotenv
import numpy as np
import time
import torch
from scipy.io import wavfile
from scipy.signal import resample

torch.set_num_threads(1)
load_dotenv(".env.production")


SAMPLE_RATE = 48000
NUM_CHANNELS = 1


async def receive_audio_frames(audio_stream: rtc.AudioStream, source: rtc.audio_frame):
    # Open a .wav file to write the audio data
    # printing.printred("Started Receiving audio frames")
    # groq_client = GroqLLMHandlers(api_key=os.getenv("GROQ_API_KEY"))

    output_filename = "audio_new_new_new_new.wav"
    i = 0
    # audio_data = bytearray()
    all_buffer = bytearray()

    # send audio frames first
    # await send_audio_frames(source, "output/full.wav")
    # return
    while True:
        with wave.open(output_filename, "wb") as wf:
            async for frame_event in audio_stream:
                buffer = frame_event.frame
                # arr = np.frombuffer(buffer.data, dtype=np.int16)
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


def get_token():
    token = (
        api.AccessToken()
        .with_identity("python-bot - sender")
        .with_name("Python Bot Sender")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room="my-room",
            )
        )
        .to_jwt()
    )
    return token


async def send_audio_frames(source: rtc.AudioSource, filename: str):
    print("Publishing audio frames")
    amplitude = 32767  # for 16-bit audio
    samples_per_channel = 480  # 10ms at 48kHz
    time = np.arange(samples_per_channel) / SAMPLE_RATE
    total_samples = 0

    print("Reading from file go.wav")

    sample_rate, data = wavfile.read("roma.wav")

    print("Sample rate: ", sample_rate)
    print("Length of data: ", len(data))
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
    print("Length of audio data: ", len(audio_data_mv) - 13000)
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


async def main(loop: asyncio.AbstractEventLoop = None):
    room = rtc.Room(loop=loop)
    await room.connect(url=os.getenv("LIVEKIT_URL"), token=get_token())
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

            print("Track published")

            # Start the receive_audio_frames task
            asyncio.ensure_future(receive_audio_frames(audio_stream, source))
            # asyncio.ensure_future(publish_frames(source, 440))

    print("Particapants: ", room.remote_participants)
    print("Tracks: ", room)
    # publish a track
    print("Sending audio")
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track_output = rtc.LocalAudioTrack.create_audio_track("sinewave", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    # Run the async function in the event loop
    loop_2 = asyncio.get_event_loop()
    print("Publishing track")
    publish_track_task = asyncio.run_coroutine_threadsafe(
        room.local_participant.publish_track(track_output, options), loop_2
    )

    asyncio.ensure_future(send_audio_frames(source, filename="output/full.wav"))
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("logs/cccc.log"), logging.StreamHandler()],
    )
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(main(loop))
    try:
        loop.run_forever()
    finally:
        loop.close()
