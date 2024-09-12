from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions

sentence_transformer_ef_huggingface = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
)

sentence_transformer_ef_chroma = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)
