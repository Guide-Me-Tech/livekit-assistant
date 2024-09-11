# server for TTS microservice
from concurrent import futures
import logging
import grpc

# from tts_pb2 import
# import tts_pb2_grpc under utils_grpc d
import tts_pb2_grpc
import tts_pb2
import xtts_funcs as xtts
from redis_funcs import RedisConnection
import os
import initializers
import datetime


initializers.load_envs()

print("Loading default model", flush=True)
model, gpt_cond_latent, speaker_embedding = xtts.load_model_for_inference(
    model_path=xtts.downloaded_model_path + "/",
    speaker_audi_paths=["./female.wav"],
)
print("XTTS Model loaded", flush=True)


print("Connecting to Redis")
redisClient = RedisConnection(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    decode_responses=True,
    password=os.getenv("REDIS_PASSWORD"),
)


_SIGNATURE_HEADER_KEY = "api-key"


def MiddlewareForApiKey(api_key):
    return redisClient.CheckIfKeyExists(api_key)


def PushUsageToRedis(api_key, usage_info):
    redisClient.UpdateUsageTTS(api_key, datetime.datetime.now(), usage_info=usage_info)


class TextToSpeechServicer(tts_pb2_grpc.TextToSpeechServicer):
    def __init__(self):
        self.xtts = None
        self.connections = []
        print("Started Server on port 50052")

    def Synthesize(self, request, context):

        # auth process
        print(context.invocation_metadata())
        metadata = dict(context.invocation_metadata())
        try:
            api_key = metadata[_SIGNATURE_HEADER_KEY]
        except KeyError:
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "API Key not present in metadata"
            )
        print("API_KEY:", api_key)
        if not MiddlewareForApiKey(api_key):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid API Key")
        else:
            print("Valid API Key")
        usage_info = {
            "bytes_length": 0,
            "tokens_used": 0,
            "language": "",
            "text_length": 0,
            "function": "Synthesize",
        }
        # print(request.audio)
        text = request.text
        language = request.language_code
        xtts.CheckLanguageSupport(language)
        # language = "en"
        audio = xtts.ttsFull(
            model=model,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            text=text,
            language=language,
        )
        token_length = xtts.calculate_the_token_length(text)
        usage_info["bytes_length"] = len(audio)
        usage_info["tokens_used"] = token_length
        usage_info["text_length"] = len(text)
        usage_info["language"] = language
        PushUsageToRedis(api_key=api_key, usage_info=usage_info)
        # print("audio", audio)
        with open("audio.wav", "wb") as f:
            f.write(audio)
        return tts_pb2.SynthesizeResponse(audio=audio)

    def SynthesizeStream(self, request, context):
        # output is the stream of SynthesizeResponse
        # auth process
        print(context.invocation_metadata())
        metadata = dict(context.invocation_metadata())
        try:
            api_key = metadata[_SIGNATURE_HEADER_KEY]
        except KeyError:
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "API Key not present in metadata"
            )
        print("API_KEY:", api_key)
        if not MiddlewareForApiKey(api_key):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid API Key")
        else:
            print("Valid API Key")
        usage_info = {
            "bytes_length": 0,
            "tokens_used": 0,
            "language": "",
            "text_length": 0,
            "function": "SynthesizeStream",
        }
        text = request.text
        language = request.language_code

        # language = "en"
        xtts.CheckLanguageSupport(language)
        print("Text: ", text)
        token_length = xtts.calculate_the_token_length(text)
        usage_info["tokens_used"] = token_length
        usage_info["language"] = language
        usage_info["text_length"] = len(text)
        #### token length is greater than 400 then split the audio and then process the chunks separately
        if token_length < 400:
            print("Token length is less than 400")
            for audio in xtts.ttsStreamEnglishRus(
                model=model,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                text=text,
                language=language,
            ):
                # print("audio", audio)
                usage_info["bytes_length"] += len(audio)
                yield tts_pb2.SynthesizeResponse(audio=audio)
        else:
            print("Token length is greater than 400")
            # split the sentence to smaller chunks and send them
            chunks = xtts.split_text_to_chunks(text, token_length)
            for chunk in chunks:
                for audio in xtts.ttsStreamEnglishRus(
                    model=model,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    text=chunk,
                    language=language,
                ):
                    usage_info["bytes_length"] += len(audio)
                    yield tts_pb2.SynthesizeResponse(audio=audio)
        PushUsageToRedis(api_key=api_key, usage_info=usage_info)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tts_pb2_grpc.add_TextToSpeechServicer_to_server(TextToSpeechServicer(), server)
    server.add_insecure_port("[::]:50052")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    print("Starting server")
    device = xtts.get_device()
    logging.basicConfig()
    serve()
