from kafka import KafkaConsumer, KafkaProducer
from typing import List
from utils.bytes_and_json import BytesAndJson


class STTWorkMessagesProducer:
    def __init__(self, kafka_servers: List[str], kafka_topic: str):
        self.kafka_servers = kafka_servers
        self.kafka_topic = kafka_topic
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=BytesAndJson.dict_to_bytes,
            client_id="stt_work_messages_client",
        )

    def produce(
        self,
        work_id: str,
        file_id: str,
        on_local: bool,
        language: str,
        file_contents: str,  # base64 encoded file contents
    ):
        if on_local:
            message_dict = {
                "work_id": work_id,
                "file_id": file_id,
                "on_local": on_local,
                "language": language,
                "file_contents": file_contents,
            }
        else:
            message_dict = {
                "work_id": work_id,
                "file_id": file_id,
                "on_local": on_local,
                "language": language,
            }
        self.producer.send(self.kafka_topic, value=message_dict)


class STTWorkMessagesConsumer:
    def __init__(self, kafka_servers: List[str], kafka_topic: str):
        self.kafka_servers = kafka_servers
        self.kafka_topic = kafka_topic
        self.consumer = KafkaConsumer(
            self.kafka_topic,
            bootstrap_servers=self.kafka_servers,
            auto_offset_reset="latest",
            enable_auto_commit=False,
            client_id="stt_work_messages_client",
            value_deserializer=BytesAndJson.bytes_to_dict,
        )

    def consume(self):
        for message in self.consumer:
            message_dict = message.value
            yield message_dict


class STTResultMessagesProducer:
    def __init__(self, kafka_servers: List[str], kafka_topic: str):
        self.kafka_servers = kafka_servers
        self.kafka_topic = kafka_topic
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=BytesAndJson.dict_to_bytes,
            client_id="stt_result_messages_client",
        )

    def produce(self, transcription: str, work_id: str, language: str):
        message_dict = {
            "work_id": work_id,
            "transcription": transcription,
            "language": language,
        }
        self.producer.send(self.kafka_topic, value=message_dict)


# Positional Arguments for KafkaConsumer:
# 	1.	*topics (str):
# 	•	This is an optional list of topics the consumer will subscribe to. If not provided during initialization, you can subscribe to topics later by calling KafkaConsumer.subscribe() or assign partitions directly using KafkaConsumer.assign().
# Keyword Arguments:
# 	2.	bootstrap_servers:
# 	•	Specifies the Kafka brokers to connect to, in the format 'host[:port]'. It could be a single string or a list of broker addresses. This list is used to initialize the connection to the Kafka cluster. It doesn’t need to contain all the brokers, just at least one that can respond to metadata requests. Default is localhost:9092.
# 	3.	client_id (str):
# 	•	A unique identifier for the Kafka client. It is used by the Kafka server for logging and monitoring purposes. If not provided, the default is kafka-python-{version}.
# 	4.	group_id (str or None):
# 	•	Defines the consumer group the consumer belongs to. Consumers in the same group share load and each reads from different partitions. If this is set to None, the consumer will not commit offsets automatically, and auto-partition assignment is disabled. Default is None.
# 	5.	key_deserializer (callable):
# 	•	A callable function to deserialize the key of the message. For example, if the key is in bytes and needs to be converted to a string, you can provide a custom deserializer.
# 	6.	value_deserializer (callable):
# 	•	Similar to the key deserializer, this deserializes the message value from bytes into an appropriate format (e.g., JSON, string, etc.).
# 	7.	fetch_min_bytes (int):
# 	•	The minimum amount of data the consumer will wait to accumulate from Kafka before returning records. If there’s not enough data, the consumer waits up to the timeout specified by fetch_max_wait_ms. Default is 1 byte.
# 	8.	fetch_max_wait_ms (int):
# 	•	The maximum amount of time (in milliseconds) the consumer will wait for new data before returning records, even if fetch_min_bytes isn’t satisfied. Default is 500 ms.
# 	9.	fetch_max_bytes (int):
# 	•	The maximum data size for a fetch request. This is a global limit across partitions but can be exceeded if the first partition fetches a message larger than this size. Default is 50 MB.
# 	10.	max_partition_fetch_bytes (int):
# 	•	The maximum amount of data returned per partition in a fetch request. Larger messages than this could cause the consumer to get stuck if they cannot be fetched. Default is 1 MB.
# 	11.	request_timeout_ms (int):
# 	•	The client’s timeout for requests. If a request to the Kafka broker takes longer than this time, the consumer will throw an error. Default is 305000 ms (5 minutes).
# 	12.	retry_backoff_ms (int):
# 	•	The time to wait before retrying a request if an error occurs. Default is 100 ms.
# 	13.	reconnect_backoff_ms (int):
# 	•	The amount of time to wait before trying to reconnect to a broker after a failed connection. Default is 50 ms.
# 	14.	reconnect_backoff_max_ms (int):
# 	•	The maximum backoff time for reconnection attempts. If connection failures continue, the time between each reconnection increases exponentially up to this maximum value. Default is 1000 ms.
# 	15.	max_in_flight_requests_per_connection (int):
# 	•	This controls how many requests the client can send to the Kafka broker without receiving responses for previous ones. Default is 5.
# 	16.	auto_offset_reset (str):
# 	•	Determines what to do when there is no initial offset for the consumer or if the current offset is out of range. Options are:
# 	•	'earliest': Moves the offset to the earliest message.
# 	•	'latest': Moves the offset to the most recent message.
# 	•	Any other value will raise an error. Default is 'latest'.
# 	17.	enable_auto_commit (bool):
# 	•	When True, the consumer will automatically commit its offset at regular intervals. Default is True.
# 	18.	auto_commit_interval_ms (int):
# 	•	Defines the interval (in milliseconds) at which offsets are automatically committed when enable_auto_commit is set to True. Default is 5000 ms.
# 	19.	default_offset_commit_callback (callable):
# 	•	A callback function that can be executed after an offset commit. It can trigger custom logic on successful or failed commit operations.
# 	20.	check_crcs (bool):
# 	•	Enables CRC32 checking on consumed records to verify data integrity. Disabling this can slightly improve performance. Default is True.
# 	21.	metadata_max_age_ms (int):
# 	•	The maximum time (in milliseconds) before forcing a metadata refresh. This helps discover new brokers or partitions. Default is 300000 ms (5 minutes).
# 	22.	partition_assignment_strategy (list):
# 	•	A list of strategies for partition assignment between consumer instances in a group. Default includes RangePartitionAssignor and RoundRobinPartitionAssignor.
# 	23.	max_poll_records (int):
# 	•	The maximum number of records returned in a single poll() call. Default is 500.
# 	24.	max_poll_interval_ms (int):
# 	•	The maximum delay between calls to poll() before the consumer is considered failed and removed from the group. This is important for consumer group rebalancing. Default is 300000 ms.
# 	25.	session_timeout_ms (int):
# 	•	Timeout used by the Kafka group management to detect consumer failures. If no heartbeat is received within this time, the consumer is removed from the group. Default is 10000 ms.
# 	26.	heartbeat_interval_ms (int):
# 	•	Defines how frequently the consumer sends heartbeats to the Kafka broker to signal its liveliness when using consumer group management. Default is 3000 ms.
# 	27.	receive_buffer_bytes (int):
# 	•	Size of the TCP receive buffer used when reading data from Kafka. By default, this relies on system defaults.
# 	28.	send_buffer_bytes (int):
# 	•	Size of the TCP send buffer. By default, this relies on system defaults.
# 	29.	socket_options (list):
# 	•	Custom socket options, such as disabling Nagle’s algorithm (TCP_NODELAY), to improve performance in specific network configurations. Default is to disable Nagle’s algorithm.
# 	30.	consumer_timeout_ms (int):
# 	•	The amount of time the consumer will block when calling poll() before raising StopIteration if no data is available. Default is to block indefinitely.
# 	31.	security_protocol (str):
# 	•	Specifies the communication protocol used to connect with Kafka brokers. Supported values include:
# 	•	PLAINTEXT: unencrypted communication.
# 	•	SSL: encrypted communication with SSL.
# 	•	SASL_PLAINTEXT: SASL authentication without encryption.
# 	•	SASL_SSL: SASL authentication with SSL encryption.
# Default is PLAINTEXT.
# 	32.	ssl_context (ssl.SSLContext):
# 	•	Pre-configured SSL context for creating secure connections. If set, other SSL-related configurations like ssl_cafile will be ignored.
# 	33.	ssl_check_hostname (bool):
# 	•	If True, SSL handshakes will verify that the broker’s certificate matches its hostname. Default is True.
# 	34.	ssl_cafile (str):
# 	•	Filepath for the Certificate Authority (CA) file used to verify broker certificates. This is needed when SSL is enabled.
# 	35.	ssl_certfile (str):
# 	•	Filepath to the client’s SSL certificate in PEM format.
# 	36.	ssl_keyfile (str):
# 	•	Filepath to the client’s SSL private key in PEM format.
# 	37.	ssl_password (str):
# 	•	Password used to decrypt the client’s SSL private key.
# 	38.	ssl_crlfile (str):
# 	•	Filepath for the Certificate Revocation List (CRL), which is used to check for revoked certificates during SSL handshakes.
# 	39.	ssl_ciphers (str):
# 	•	Optional list of allowed SSL ciphers.
# 	40.	api_version (tuple):
# 	•	Specifies the version of the Kafka API to use. If not set, the client will probe the broker to determine the version. Certain features are only available in later versions (e.g., group coordination was introduced in Kafka 0.9.0.0).
# 	41.	api_version_auto_timeout_ms (int):
# 	•	Timeout for the API version probing process, used only when api_version is set to None.
# 	42.	connections_max_idle_ms (int):
# 	•	Time in milliseconds before closing idle connections. Helps avoid unexpected disconnections due to broker timeouts. Default is 540000 ms.
# 	43.	metric_reporters (list):
# 	•	A list of classes used to report Kafka client metrics. Allows integration with custom or third-party monitoring systems.
# 	44.	metrics_num_samples (int):
# 	•	Number of samples maintained for calculating metrics. Default
