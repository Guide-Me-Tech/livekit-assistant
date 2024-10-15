import json


class BytesAndJson:
    @staticmethod
    def bytes_to_json(bytes_data: bytes) -> dict:
        return bytes_data.decode("utf-8")

    @staticmethod
    def json_to_bytes(json_data: str) -> bytes:
        return json_data.encode("utf-8")

    @staticmethod
    def dict_to_bytes(dict_data: dict) -> bytes:
        return json.dumps(dict_data).encode("utf-8")

    @staticmethod
    def bytes_to_dict(bytes_data: bytes) -> dict:
        return json.loads(bytes_data.decode("utf-8"))
