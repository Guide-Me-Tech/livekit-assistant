import docker
import json
import os
import sys
import time

client = docker.from_env()
while True:
    # continue
    print("[ ", end="", flush=True)
    for container in client.containers.list(all=True):
        print(container.attrs["Name"], end=", ", flush=True)
        if (
            container.attrs["Name"] == "/classifier"
            and container.attrs["State"]["Status"] != "running"
        ):
            print("Classifier is stopped should be restarted")
            os.system(
                "sh ~/aslon/consultant_ai_sentence_classifier/utils/shell_scripts/restart.sh"
            )
    print("]")
    # print()
    time.sleep(10)
