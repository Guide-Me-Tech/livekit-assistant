dependencies_list = [
    "langchain",
    "openai",
    "python-dotenv",
    "chromadb",
    "orjson",
    "grpcio-tools",
    "tensorflow",
    "bs4",
    "nltk",
    "sentence_transformers",
    "langchain_openai",
]

import os

for dependency in dependencies_list:
    os.system("pip install " + dependency)

# pip install langchain
# 10487  pip install openai
# 10488  pip install dotenv
# 10489  pip install python-dotenv
# 10490  pip install chromadb
# 10491  pip install orjson
# 10492  pip install grpcio-tools
# 10494  pip install tensorflow
# 10495  pip install bs4 nlyk
# 10496  pip install bs4 nltk
# 10498  pip install sentence_transformers
# 10500  pip install langchain_openai
# 10509  pip install langchainhub
