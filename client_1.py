import asyncio
from livekit import RTCClient, RoomEvent, TrackPublication
from aiortc import MediaStreamTrack


class AudioProcessor(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track: MediaStreamTrack):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()  # Receive the audio frame
        audio_data = frame.to_ndarray()  # Convert frame to numpy array for processing

        # Process the audio data here (e.g., audio analysis, filtering, etc.)
        print(f"Audio frame received: {audio_data}")

        return frame  # Return the frame (optional, depending on your processing needs)


async def main():
    # Create a LiveKit client and connect to the room
    client = RTCClient()
    await client.connect()

    @client.on(RoomEvent.TrackSubscribed)
    async def on_track_subscribed(publication: TrackPublication):
        if publication.kind == "audio":
            audio_track = publication.track

            if isinstance(audio_track, MediaStreamTrack):
                # Wrap the track with our custom AudioProcessor
                audio_processor = AudioProcessor(audio_track)

                # Now process the audio
                while True:
                    await audio_processor.recv()  # Process the incoming audio frames

    await client.join(room="your-room-id")


if __name__ == "__main__":
    asyncio.run(main())
