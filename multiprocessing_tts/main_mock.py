from multiprocessing import Process, Queue, shared_memory
import os
import torch
import sys
import json
from utils.kafka_utils import KafkaQueueProcessor
from utils.shared_memory_queue import SharedMemoryWithIndexes
import time
from utils.xtts_funcs import (
    ttsStreamEnglishRus,
    ttsFull,
    downloaded_model_path,
    load_model_for_inference,
)
import numpy as np
import librosa
from scipy.io import wavfile


def processor(worker_processes_queue: Queue, worker_id: int):
    # tts worker here
    # load model here
    print(f"Loading model for worker {worker_id}")
    # number of gpus
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    print(f"Number of GPUs: {num_gpus}    ------ Number of CPUs: {num_cpus}")
    if num_gpus > 0:
        device = "cuda"
        id_to_use = worker_id % num_gpus
    else:
        device = "cpu"
        id_to_use = 0
    print(f"Using device: {device}")
    print(f"ID to use: {id_to_use}")
    # model, gpt_cond_latent, speaker_embedding = load_model_for_inference(
    #     downloaded_model_path,
    #     speaker_audi_paths=["female.wav"],
    #     device_id=id_to_use,
    # )

    while True:
        # read from kafka or any message queue
        work = worker_processes_queue.get(block=True)
        print(
            f"Took work from queue by worker {worker_id} ---- {work['id']} ------- time stamped: {int(time.time() * 1000) - work['timestamp']}"
        )
        print("WORK:", work)
        shared_memory_name = work["shm_name"]
        shm_queue = SharedMemoryWithIndexes(shared_memory_name)
        # for audio_chunk in ttsStreamEnglishRus(
        #     model=model,
        #     gpt_cond_latent=gpt_cond_latent,
        #     speaker_embedding=speaker_embedding,
        #     text=work["text"],
        #     language=work["language"],
        # ):
        # # open the audio with numpy array

        rate, audio_chunk = wavfile.read("../audios/roma.wav")
        # print(audio_chunk[0:10])
        # get only first 48000 samples
        # audio_chunk = audio_chunk[:96000]
        for i in range(2):
            print(f"Enqueuing message: {i} ----- SHM: {shared_memory_name}")
            start = i * shm_queue.item_size
            ending = (i + 1) * shm_queue.item_size
            print(start, ending)
            temp = audio_chunk[start:ending]
            # temp = audio_chunk[
            #     i * shm_queue.item_size : (i + 1) * shm_queue.item_size
            # ]
            shm_queue.enqueue(temp)
            # time.sleep(1)
        shm_queue.close()
        # time.sleep(100)


def KafkaConsumeProcessor(queue: Queue):
    kafka_thing = KafkaQueueProcessor(
        topic="tts_messages",
        bootstrap_servers="localhost:9092",
        queue=queue,
    )
    kafka_thing.consume()


def main():
    processes = []
    # sys.argv[1] is the number of processes to create
    #
    worker_processes_queue = Queue()
    print("Created KafkaQueueProcessor")

    for i in range(int(sys.argv[1])):
        process = Process(target=processor, args=(worker_processes_queue, i))
        process.start()
        processes.append(process)
    print(f"Started all worker processes --- {len(processes)}")
    kafka_consumer_process = Process(
        target=KafkaConsumeProcessor, args=(worker_processes_queue,)
    )
    kafka_consumer_process.start()
    processes.append(kafka_consumer_process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
