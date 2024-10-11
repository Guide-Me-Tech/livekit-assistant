from kafka import TopicPartition, KafkaConsumer
from kafka import KafkaProducer
from kafka.errors import KafkaTimeoutError
from kafka.admin import KafkaAdminClient, NewTopic
import json
from multiprocessing import Queue

conf = {
    "bootstrap.servers": "host1:9092,host2:9092",
    "group.id": "foo",
    "auto.offset.reset": "smallest",
}


class KafkaQueueProcessor:
    def __init__(self, topic: str, bootstrap_servers: str, queue: Queue):
        self.consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="latest",
            enable_auto_commit=False,
        )
        topic_partition = TopicPartition(topic, 0)  # Assign partition 0
        self.consumer.assign([topic_partition])
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        self.queue = queue

    def consume(self):
        for msg in self.consumer:
            # deserialize the message
            message = json.loads(msg.value)
            message["timestamp"] = msg.timestamp
            # print(message)
            self.queue.put(message)

    def produce(self, topic, value):
        # serialize and convert to bytes
        val = json.dumps(value).encode("utf-8")
        try:
            self.producer.send(topic, value=val, partition=0)
        except KafkaTimeoutError:
            raise Exception("Failed to produce message to Kafka")
        finally:
            self.producer.flush()

    def create_topic_with_retention(
        self,
        topic_name: str,
        num_partitions: int = 1,
        replication_factor: int = 1,
        retention_ms: int = 604800000,
    ):
        admin_client = KafkaAdminClient(
            bootstrap_servers=self.consumer.config["bootstrap_servers"]
        )
        topic_configs = {"retention.ms": str(retention_ms)}
        new_topic = NewTopic(
            name=topic_name,
            num_partitions=num_partitions,
            replication_factor=replication_factor,
            topic_configs=topic_configs,
        )
        try:
            admin_client.create_topics([new_topic])
            print(
                f"Topic '{topic_name}' created successfully with retention period of {retention_ms} ms."
            )
        except Exception as e:
            print(f"Failed to create topic: {e}")
        finally:
            admin_client.close()
