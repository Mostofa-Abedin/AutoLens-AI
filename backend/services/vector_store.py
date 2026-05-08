import os
from pathlib import Path

COLLECTION_NAME = "vehicle_reference_images"
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/chroma")

_client = None
_collection = None


def get_client():
    global _client
    if _client is None:
        import chromadb
        Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_images(embeddings: list, metadatas: list, ids: list):
    get_collection().add(embeddings=embeddings, metadatas=metadatas, ids=ids)


def find_similar(embedding: list, top_k: int = 5) -> list[dict]:
    col = get_collection()
    if col.count() == 0:
        return []
    results = col.query(query_embeddings=[embedding], n_results=min(top_k, col.count()))
    out = []
    for i, meta in enumerate(results["metadatas"][0]):
        distance = results["distances"][0][i]
        # ChromaDB cosine distance: 0 = identical, 2 = opposite. Normalise to [0,1].
        similarity = 1.0 - (distance / 2.0)
        out.append({
            "path":             meta.get("path", ""),
            "class_id":         meta.get("class_id", ""),
            "angle":            meta.get("angle", ""),
            "filename":         meta.get("filename", ""),
            "similarity_score": round(similarity, 4),
        })
    return out


def count() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0


def clear():
    global _collection
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    get_collection()  # recreate
