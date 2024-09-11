from livekit import api
import dotenv

dotenv.load_dotenv()
import os

# will automatically use the LIVEKIT_API_KEY and LIVEKIT_API_SECRET env vars
# token = (
#     api.AccessToken()
#     .with_identity("python-bot")
#     .with_name("Python Bot")
#     .with_grants(
#         api.VideoGrants(
#             room_join=True,
#             room="my-room",
#         )
#     )
#     .to_jwt()
# )


from livekit import api
import asyncio


async def main():
    lkapi = api.LiveKitAPI(
        "wss://consultant-ai-weia2b76.livekit.cloud",
    )
    room_info = await lkapi.room.create_room(
        api.CreateRoomRequest(name="my-room"),
    )
    print(room_info)
    results = await lkapi.room.list_rooms(api.ListRoomsRequest())
    print(results)
    await lkapi.aclose()


asyncio.get_event_loop().run_until_complete(main())
