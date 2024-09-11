# server for TTS microservice
from concurrent import futures
import logging
import grpc

# from tts_pb2 import
# import tts_pb2_grpc under utils_grpc d
import xtts_funcs as xtts
import os
import datetime

print("Loading default model", flush=True)
model, gpt_cond_latent, speaker_embedding = xtts.load_model_for_inference(
    model_path=xtts.downloaded_model_path + "/",
    speaker_audi_paths=["./female.wav"],
)
print("XTTS Model loaded", flush=True)


async def ttswrapper(text, language):

    return xtts.ttsStreamEnglishRus(
        model=model,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        text=text,
        language=language,
    )
