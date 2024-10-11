from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, split_sentence
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
import time
import torch
import torchaudio
import io
import soundfile as sf
import subprocess
import os


model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

if os.path.exists("./tts_models--multilingual--multi-dataset--xtts_v2"):
    downloaded_model_path = "./tts_models--multilingual--multi-dataset--xtts_v2"
else:
    downloaded_model_path = os.path.join(
        get_user_data_dir("tts"), model_name.replace("/", "--")
    )

#

# English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko) Hindi (hi).
SUPPORTED_LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "ja",
    "hu",
    "ko",
    "hi",
]

print("Model Found on ", downloaded_model_path)

MAX_TOKEN_LENGTH = 400
tokenizer = VoiceBpeTokenizer(downloaded_model_path + "/vocab.json")


def calculate_the_token_length(text, language="en"):
    token_length = (
        torch.IntTensor(tokenizer.encode(text, lang=language))
        .unsqueeze(0)
        .to("cuda")
        .shape[-1]
    )
    print("Token length: ", token_length)
    return token_length


def get_device():
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    return device


def load_model_for_inference(
    model_path,
    speaker_audi_paths=["female.wav"],
    use_deepspeed=False,
    device_id: int = 0,
):
    print("Loading model...")
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, checkpoint_dir=model_path, use_deepspeed=use_deepspeed
    )
    model.cuda(device=device_id)
    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=speaker_audi_paths
    )
    return model, gpt_cond_latent, speaker_embedding


# print("Loading model...")
# config = XttsConfig()
# config.load_json("./tts_models--multilingual--multi-dataset--xtts_v2/config.json")
# model = Xtts.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="./tts_models--multilingual--multi-dataset--xtts_v2/", use_deepspeed=False)
# model.cuda()


# print("Computing speaker latents...")
# gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["female.wav"])
# aslon 1213
def CheckLanguageSupport(language):
    if language in SUPPORTED_LANGUAGES:
        return
    else:
        raise ValueError(
            "Language not supported - supported languages: ", SUPPORTED_LANGUAGES
        )


def split_text_to_chunks(text, token_length):
    chunks_num = token_length // MAX_TOKEN_LENGTH + 1
    # split str with length of MAX_TOKEN_LENGTH
    chunks = []
    for i in range(chunks_num):
        if (i + 1) * MAX_TOKEN_LENGTH > len(text):
            chunks.append(text[i * MAX_TOKEN_LENGTH :])
        else:
            chunks.append(text[i * MAX_TOKEN_LENGTH : (i + 1) * MAX_TOKEN_LENGTH])
    return chunks


def ttsFull(
    model,
    gpt_cond_latent,
    speaker_embedding,
    text="Today I was given a right to ask and noone is gonna take it off",
    language="en",
):
    print("Inference Full...")
    t0 = time.time()
    output = model.inference(text, language, gpt_cond_latent, speaker_embedding)
    f = io.BytesIO()
    sf.write(f, output["wav"], 24000, format="WAV", subtype="PCM_16")
    print("Time to inference: ", time.time() - t0)
    return f.getvalue()


def ttsStreamEnglishRus(
    model,
    gpt_cond_latent,
    speaker_embedding,
    text="Today I was given a right to ask and noone is gonna take it off",
    language="en",
):
    print("Inference Stream...")
    t0 = time.time()
    chunks = model.inference_stream(text, language, gpt_cond_latent, speaker_embedding)
    # wav_chunks_bytes = []
    # wav_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"Time to first chunck: {time.time() - t0}")
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        # play each chunk

        # torchaudio.save(f" chunk_{i}.wav", chunk.squeeze().unsqueeze(0).cpu(), 24000)
        # IPython.display.display(IPython.display.Audio(f"chunk_{i}.wav"))
        b = io.BytesIO()
        j = chunk.squeeze().cpu().numpy()
        sf.write(b, j, 24000, format="WAV", subtype="PCM_16")
        # wav_chunks.append(j)
        # wav_chunks_bytes.append(b.getvalue())
        yield b.getvalue()


def check_cuda_smi() -> bool:
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True  # CUDA available
    except FileNotFoundError:
        return False  # nvidia-smi not found
    except subprocess.CalledProcessError:
        return False  # nvidia-smi execution failed


def check_cuda_availability() -> bool:
    return torch.cuda.is_available()
