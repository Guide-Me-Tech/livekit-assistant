# server for TTS microservice
from concurrent import futures
import logging
import grpc

# from tts_pb2 import
# import tts_pb2_grpc under utils_grpc d
import xtts_funcs as xtts
import os
import datetime
import torch
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, split_sentence
from utils import printing
from utils.timer import timer


MAX_TOKEN_LENGTH = 350
tokenizer = VoiceBpeTokenizer(xtts.downloaded_model_path + "/vocab.json")

printing.printred(f"Loading default model ---- {xtts.downloaded_model_path}")
model, gpt_cond_latent, speaker_embedding = xtts.load_model_for_inference(
    model_path=xtts.downloaded_model_path + "/",
    speaker_audi_paths=["./roma.wav"],
)
printing.printred("XTTS Model loaded")


@timer
def calculate_the_token_length(text, language="en"):
    token_length = (
        torch.IntTensor(tokenizer.encode(text, lang=language))
        .unsqueeze(0)
        .to("cuda")
        .shape[-1]
    )
    print("Token length: ", token_length)
    return token_length


async def ttswrapper(text, language):
    token_length = calculate_the_token_length(text, language)
    chunking_required = False
    chunks = []
    if token_length > MAX_TOKEN_LENGTH:
        num_chunks = token_length // MAX_TOKEN_LENGTH
        chunking_required = True
        chunks = split_sentence(text, lang=language, text_split_length=MAX_TOKEN_LENGTH)
    if not chunking_required:
        gens = xtts.ttsStreamEnglishRus(
            model=model,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            text=text,
            language=language,
        )
        for gen in gens:
            yield gen
    else:
        for chunk in chunks:
            gens = xtts.ttsStreamEnglishRus(
                model=model,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                text=chunk,
                language=language,
            )
            for gen in gens:
                yield gen
