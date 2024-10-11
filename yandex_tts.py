from argparse import ArgumentParser
import os
from speechkit import model_repository, configure_credentials, creds
from dotenv import load_dotenv
from scipy.io.wavfile import read

load_dotenv()

# Authentication via an API key.
configure_credentials(
    yandex_credentials=creds.YandexCredentials(
        iam_token=os.getenv("YANDEX_OAUTH_TOKEN")
    )
)


def synthesize(text, export_path):
    model = model_repository.synthesis_model()

    # Set the synthesis settings.
    model.voice = "nigora"
    model.role = "good"

    # Performing speech synthesis and creating an audio file with results.
    result = model.synthesize(text, raw_format=False)
    result.export(export_path, "wav")

    return read(export_path)
