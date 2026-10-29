import torch
from sentence_transformers import SentenceTransformer
from . import config
from .data_classes import embedded_chunk, knowledge_chunk


embedding_model = SentenceTransformer(
    config.EMBEDDING_MODEL_NAME,
    revision=config.EMBEDDING_MODEL_REVISION,
    # CPU and float32 give the same vectors on every run and machine
    device="cpu",
    model_kwargs={"dtype": torch.float32},
)


def embed_texts(texts, is_query=False):
    """Turn a list of texts into a (len(texts), 640) array of unit-length vectors."""

    # Queries get the task instruction in front; chunks are embedded as they are
    prompt = config.EMBEDDING_QUERY_INSTRUCTION if is_query else None
    return embedding_model.encode(
        texts,
        prompt=prompt,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def embed_chunks(chunks: list[knowledge_chunk]) -> list[embedded_chunk]:
    """Add an 'embedding' field to each chunk, containing its vector representation."""

    # Embed all texts in one call (faster than one call per chunk); row i belongs to chunks[i]
    embeddings = embed_texts([chunk.text for chunk in chunks])
    return [
        embedded_chunk(**chunk.model_dump(), embedding=embedding.tolist())
        for chunk, embedding in zip(chunks, embeddings)
    ]