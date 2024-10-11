from kafka import KafkaProducer
# config = {
#     "bootstrap_servers": "localhost:9092",
#     # 'client.id': socket.gethostname()
# }

producer = KafkaProducer(bootstrap_servers="localhost:9092")
print(producer.bootstrap_connected())
for _ in range(100):
    producer.send("foobar", b"some_message_bytes")
